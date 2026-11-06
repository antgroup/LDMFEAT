import math
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

MAX_ATTN_ELEMENTS = 5_000_000_000


class FEATImputer:
    """
    Zero-training-cost robust missing-value imputation pipeline for FEAT.
    
    This imputer implements linear boundary-clipping with RobustScale (for continuous)
    and Frequency Encoding (for categoricals) to correctly align data with the
    normal/linear prior expected by the pre-trained NanEncoder and NormalizationEncoder
    layers inside FEAT, avoiding catastrophic NRMSE long-tail explosions.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = torch.device(device)
        self.amp_dtype = torch.bfloat16 if (self.device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
        self.model.eval()

    def _compute_max_rows(self, num_features):
        """Return the maximum number of rows that can be processed in one pass."""
        features_per_group = getattr(self.model, 'features_per_group', 2)
        nhead = getattr(self.model, 'nhead', 8)
        num_groups = math.ceil(num_features / features_per_group)
        attn_per_row = nhead * num_groups * num_groups
        max_rows = max(1, int(MAX_ATTN_ELEMENTS // max(1, attn_per_row)))
        return max_rows

    def _feat_infer_chunk(self, X_chunk):
        """Run FEAT inference on one chunk and return denormalized predictions."""
        x_tensor = torch.from_numpy(X_chunk).float().to(self.device)
        y_dummy = torch.zeros(X_chunk.shape[0], dtype=torch.float32, device=self.device)
        y_type_dummy = torch.zeros(X_chunk.shape[0], dtype=torch.float32, device=self.device)

        eval_pos = X_chunk.shape[0]
        x_batch = x_tensor.unsqueeze(0)
        y_batch = y_dummy.unsqueeze(0)
        y_type_batch = y_type_dummy.unsqueeze(0)

        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type == 'cuda',
            dtype=self.amp_dtype,
        ):
            out = self.model(x=x_batch, y=y_batch, eval_pos=eval_pos, y_type=y_type_batch, task_type='cls')

            if not (isinstance(out, dict) and 'feature_pred' in out):
                raise KeyError("Model output does not contain 'feature_pred'!")

            feature_pred = out['feature_pred']
            process_config = out['process_config']

            num_features_cfg = process_config['features_per_group']
            num_used = process_config['num_used_features']
            if num_used is not None and num_features_cfg is not None:
                feature_pred = feature_pred / torch.sqrt(
                    num_features_cfg / num_used.to(self.device)
                )

            mean_norm = process_config['mean_for_normalization']
            std_norm = process_config['std_for_normalization']
            if mean_norm is not None and std_norm is not None:
                feature_pred = feature_pred * std_norm + mean_norm

            b, s, g, fpg = feature_pred.shape
            pred_np = feature_pred.permute(1, 0, 2, 3).reshape(s, b, g * fpg).squeeze(1).float().cpu().numpy()

            n_x_padding = process_config['n_x_padding']
            if n_x_padding > 0:
                pred_np = pred_np[:, :-n_x_padding]

        return pred_np

    def impute(self, masked_df, gt_df=None, mask_bool=None):
        """
        Perform zero-training-cost missing-value imputation using FEAT.
        
        Args:
            masked_df: DataFrame with missing values (np.nan).
            gt_df: (Optional) Ground truth DataFrame, used solely for type resolution. 
                   If not provided, columns with object/category strings are parsed automatically.
            mask_bool: (Optional) Boolean mask indicating which cells were missing. 
                       If not provided, inferred automatically from masked_df.isna().
                       
        Returns:
            pd.DataFrame with exactly the same shape and format as masked_df, where NaNs are filled.
        """
        X_df = masked_df.copy()
        
        if mask_bool is None:
            mask_bool = X_df.isna().values

        actual_cat_cols = []
        cat_mappings = {}
        
        # 1. Frequency Encoding Mapping for Categoricals (Preserves Mode-Seeking Prior)
        for i, col in enumerate(X_df.columns):
            is_cat = False
            if gt_df is not None:
                is_cat = gt_df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(gt_df[col])
            else:
                is_cat = X_df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X_df[col])
                
            if is_cat:
                actual_cat_cols.append(i)
                # Frequency statistics must come only from observed input cells.
                # gt_df is intentionally used for type resolution only; using its
                # hidden values here would leak benchmark ground truth.
                valid_series = X_df[col].dropna()
                val_counts = valid_series.value_counts(dropna=True)
                uniques = val_counts.index.values
                if len(uniques) == 0:
                    raise ValueError(
                        f"Categorical column '{col}' has no observed value to infer its vocabulary."
                    )
                cat_mappings[i] = uniques

                value_to_code = {v: idx for idx, v in enumerate(uniques)}
                new_col = np.full(len(X_df), np.nan, dtype=np.float32)
                for j, row_val in enumerate(X_df[col]):
                    if pd.notna(row_val) and row_val in value_to_code:
                        new_col[j] = float(value_to_code[row_val])
                X_df[col] = new_col

        X_np = X_df.values.astype(np.float32)
        finite_mask = np.isfinite(X_np)
        X_np[finite_mask] = np.clip(X_np[finite_mask], -1e6, 1e6)

        num_rows, num_features = X_np.shape

        # 2. Robust Linear Scaling with 50x Bounds for Continuous Targets
        scaler_dict = {}
        for i in range(num_features):
            if i not in actual_cat_cols:
                col_data = X_np[:, i]
                valid_mask = ~np.isnan(col_data)
                n_valid = valid_mask.sum()
                if n_valid > 1:
                    scaler = RobustScaler()
                    valid_vals = col_data[valid_mask].reshape(-1, 1)
                    scaled_vals = scaler.fit_transform(valid_vals)
                    X_np[valid_mask, i] = np.clip(scaled_vals, -50.0, 50.0).squeeze()
                    scaler_dict[i] = scaler

        # 3. Secure Chunked CCMM Block Inference
        max_rows = self._compute_max_rows(num_features)
        if num_rows <= max_rows:
            pred_np = self._feat_infer_chunk(X_np)
        else:
            num_chunks = (num_rows + max_rows - 1) // max_rows
            print(f"  [FEATImputer] Splitting {num_rows} rows into {num_chunks} chunks.")
            chunks = []
            for start in range(0, num_rows, max_rows):
                end = min(start + max_rows, num_rows)
                chunks.append(self._feat_infer_chunk(X_np[start:end]))
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            pred_np = np.concatenate(chunks, axis=0)

        actual_num_features = masked_df.shape[1]
        pred_np = pred_np[:, :actual_num_features]

        # 4. Inverse Robust Linear Scaling + Absolute Clipping
        for i in range(actual_num_features):
            if i not in actual_cat_cols and i in scaler_dict:
                scaler = scaler_dict[i]
                pred_col = pred_np[:, i].reshape(-1, 1)
                pred_col = np.clip(pred_col, -50.0, 50.0)
                pred_np[:, i] = scaler.inverse_transform(pred_col).squeeze()

        # 5. Composite Final Predictions
        pred_df = masked_df.copy()
        for i, col in enumerate(pred_df.columns):
            if i in actual_cat_cols:
                uniques = cat_mappings.get(i, [])
                if len(uniques) > 0:
                    pred_codes = np.round(pred_np[:, i])
                    pred_codes = np.nan_to_num(pred_codes, nan=0.0)
                    pred_codes = np.clip(pred_codes, 0, len(uniques) - 1).astype(int)
                    col_mask = mask_bool[:, i]
                    pred_df.loc[col_mask, col] = np.array(uniques)[pred_codes[col_mask]]
            else:
                col_mask = mask_bool[:, i]
                pred_df.loc[col_mask, col] = pred_np[col_mask, i]

        return pred_df
