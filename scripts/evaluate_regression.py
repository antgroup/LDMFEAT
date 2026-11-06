"""
FEAT Regression Evaluation (DDP Supported)
===========================================
The evaluator improves prediction stability through:
1. Multi-pass inference with median aggregation
2. Context shuffling for diversity
3. Robust outlier handling

Usage:
    # Single GPU
    python scripts/evaluate_regression.py --model_path your_model.pt

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 scripts/evaluate_regression.py --model_path your_model.pt

"""

# Fix OpenBLAS thread allocation issue - MUST be before any numpy/scipy imports
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import sys
import argparse
import time
from pathlib import Path
from datetime import timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.metrics import r2_score, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feat.inference.data_distillation import distill_training_data
from feat.utils.loading import load_model
from scipy import stats


DEFAULT_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "FEAT-38M.pt"


# ============================================================================
# DDP Utilities
# ============================================================================

def setup_distributed() -> Tuple[int, int, bool]:
    """Initialize distributed environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # Extended timeout for long-running evaluations
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            timeout=timedelta(hours=3)
        )

        torch.cuda.set_device(local_rank)
        return rank, world_size, True
    else:
        return 0, 1, False


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """Check if current process is rank 0."""
    return rank == 0


def print_rank0(msg: str, rank: int):
    """Print only on rank 0."""
    if is_main_process(rank):
        print(msg)


def distribute_datasets(datasets: List[str], rank: int, world_size: int) -> List[str]:
    """Distribute datasets across ranks using round-robin."""
    if world_size == 1:
        return datasets
    return [d for i, d in enumerate(datasets) if i % world_size == rank]


def gather_results(local_results: List[Dict], rank: int, world_size: int) -> List[Dict]:
    """Gather results from all ranks to rank 0."""
    if world_size == 1:
        return local_results

    # Use all_gather_object for simplicity
    gathered_list = [None] * world_size
    dist.all_gather_object(gathered_list, local_results)

    if rank == 0:
        all_results = []
        for results in gathered_list:
            if results:
                all_results.extend(results)
        return all_results
    else:
        return []


# ============================================================================
# Data Loading
# ============================================================================

def load_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset from directory."""
    dataset_dir = Path(dataset_path)

    train_file = dataset_dir / 'train.csv'
    test_file = dataset_dir / 'test.csv'

    if train_file.exists() and test_file.exists():
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
    else:
        train_files = list(dataset_dir.glob('*_train.csv'))
        test_files = list(dataset_dir.glob('*_test.csv'))

        if train_files and test_files:
            train_df = pd.read_csv(train_files[0])
            test_df = pd.read_csv(test_files[0])
        else:
            raise FileNotFoundError(f"No train/test files found in {dataset_path}")

    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    return X_train, y_train, X_test, y_test


def preprocess_features(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess features."""
    X_train_df = X_train_df.copy()
    X_test_df = X_test_df.copy()

    actual_cat_cols = []
    for col in X_train_df.columns:
        train_numeric = pd.to_numeric(X_train_df[col], errors='coerce')
        test_numeric = pd.to_numeric(X_test_df[col], errors='coerce')

        if train_numeric.isna().all() and test_numeric.isna().all():
            actual_cat_cols.append(col)
        else:
            X_train_df[col] = train_numeric
            X_test_df[col] = test_numeric

    for col in actual_cat_cols:
        X_train_df[col] = X_train_df[col].fillna('__MISSING__')
        X_test_df[col] = X_test_df[col].fillna('__MISSING__')
        codes, _ = pd.factorize(pd.concat([X_train_df[col], X_test_df[col]]))
        X_train_df[col] = codes[:len(X_train_df)].astype(np.float32)
        X_test_df[col] = codes[len(X_train_df):].astype(np.float32)

    X_train_np = X_train_df.values.astype(np.float32)
    X_test_np = X_test_df.values.astype(np.float32)

    mean_X = np.nanmean(X_train_np, axis=0)
    std_X = np.nanstd(X_train_np, axis=0) + 1e-6
    mean_X = np.nan_to_num(mean_X, nan=0.0)
    std_X = np.nan_to_num(std_X, nan=1.0)

    X_train_norm = (X_train_np - mean_X) / std_X
    X_test_norm = (X_test_np - mean_X) / std_X

    X_train_norm = np.nan_to_num(X_train_norm, nan=0.0)
    X_test_norm = np.nan_to_num(X_test_norm, nan=0.0)

    return X_train_norm, X_test_norm


# ============================================================================
# Target Transform (Y值变换) - 针对偏斜分布的数据
# ============================================================================

class TargetTransformer:
    """
    对 Y 值进行变换，使其更适合 ICL 模型学习。
    支持多种变换方式，并提供逆变换以还原预测结果。
    """

    def __init__(self, method: str = 'auto'):
        """
        Args:
            method: 变换方法
                - 'auto': 自动检测最佳变换
                - 'none': 不变换
                - 'log': log(y + shift) 变换（适合正偏态）
                - 'sqrt': sqrt(y + shift) 变换（适合正偏态但程度较轻）
                - 'boxcox': Box-Cox 变换（自动优化 lambda）
                - 'yeo-johnson': Yeo-Johnson 变换（支持负值）
                - 'quantile': 分位数变换到正态分布
                - 'rank': 转换为排名（0-1范围）
        """
        self.method = method
        self.fitted_method = None
        self.shift = 0
        self.lmbda = None  # Box-Cox / Yeo-Johnson lambda
        self.quantile_transformer = None
        self.y_min = None
        self.y_max = None
        self.y_mean = None
        self.y_std = None

    def _detect_best_transform(self, y: np.ndarray) -> str:
        """自动检测最佳变换方法"""
        skewness = stats.skew(y)
        has_negative = np.any(y <= 0)
        y_range = np.max(y) - np.min(y)
        y_std = np.std(y)
        cv = y_std / (np.abs(np.mean(y)) + 1e-8)  # 变异系数

        # 如果分布已经比较对称，不需要变换
        if abs(skewness) < 0.5:
            return 'none'

        # 如果有负值，使用 Yeo-Johnson
        if has_negative:
            return 'yeo-johnson'

        # 强正偏态用 log
        if skewness > 2:
            return 'log'

        # 中等正偏态用 sqrt
        if skewness > 1:
            return 'sqrt'

        # 如果变异系数很大，用分位数变换
        if cv > 2:
            return 'quantile'

        # 默认使用 Yeo-Johnson（最通用）
        return 'yeo-johnson'

    def fit(self, y: np.ndarray) -> 'TargetTransformer':
        """拟合变换参数"""
        y = np.asarray(y).flatten()
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        self.y_min = np.min(y)
        self.y_max = np.max(y)

        if self.method == 'auto':
            self.fitted_method = self._detect_best_transform(y)
        else:
            self.fitted_method = self.method

        if self.fitted_method == 'log':
            # 确保所有值为正
            self.shift = max(0, -self.y_min + 1)

        elif self.fitted_method == 'sqrt':
            self.shift = max(0, -self.y_min + 0.01)

        elif self.fitted_method == 'boxcox':
            self.shift = max(0, -self.y_min + 1)
            y_shifted = y + self.shift
            _, self.lmbda = stats.boxcox(y_shifted)

        elif self.fitted_method == 'yeo-johnson':
            _, self.lmbda = stats.yeojohnson(y)

        elif self.fitted_method == 'quantile':
            from sklearn.preprocessing import QuantileTransformer
            self.quantile_transformer = QuantileTransformer(
                output_distribution='normal', random_state=42
            )
            self.quantile_transformer.fit(y.reshape(-1, 1))

        elif self.fitted_method == 'rank':
            pass  # rank 变换不需要拟合参数

        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """变换 Y 值"""
        y = np.asarray(y).flatten()

        if self.fitted_method == 'none':
            return y

        elif self.fitted_method == 'log':
            return np.log1p(y + self.shift)

        elif self.fitted_method == 'sqrt':
            return np.sqrt(y + self.shift)

        elif self.fitted_method == 'boxcox':
            y_shifted = y + self.shift
            return stats.boxcox(y_shifted, lmbda=self.lmbda)

        elif self.fitted_method == 'yeo-johnson':
            return stats.yeojohnson(y, lmbda=self.lmbda)

        elif self.fitted_method == 'quantile':
            return self.quantile_transformer.transform(y.reshape(-1, 1)).flatten()

        elif self.fitted_method == 'rank':
            # 转换为 0-1 范围的排名
            ranks = stats.rankdata(y, method='average')
            return (ranks - 1) / (len(ranks) - 1 + 1e-8)

        return y

    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """逆变换预测值"""
        y_transformed = np.asarray(y_transformed).flatten()

        if self.fitted_method == 'none':
            return y_transformed

        elif self.fitted_method == 'log':
            return np.expm1(y_transformed) - self.shift

        elif self.fitted_method == 'sqrt':
            return np.square(np.maximum(y_transformed, 0)) - self.shift

        elif self.fitted_method == 'boxcox':
            from scipy.special import inv_boxcox
            y_inv = inv_boxcox(y_transformed, self.lmbda)
            return y_inv - self.shift

        elif self.fitted_method == 'yeo-johnson':
            return self._inv_yeojohnson(y_transformed, self.lmbda)

        elif self.fitted_method == 'quantile':
            return self.quantile_transformer.inverse_transform(
                y_transformed.reshape(-1, 1)
            ).flatten()

        elif self.fitted_method == 'rank':
            # rank 变换的逆变换需要原始值范围
            return y_transformed * (self.y_max - self.y_min) + self.y_min

        return y_transformed

    def _inv_yeojohnson(self, y: np.ndarray, lmbda: float) -> np.ndarray:
        """Yeo-Johnson 逆变换"""
        y = np.asarray(y)
        result = np.zeros_like(y)

        pos_mask = y >= 0
        neg_mask = ~pos_mask

        if lmbda == 0:
            result[pos_mask] = np.expm1(y[pos_mask])
        elif lmbda != 2:
            result[pos_mask] = np.power(y[pos_mask] * lmbda + 1, 1 / lmbda) - 1

        if lmbda != 2:
            result[neg_mask] = 1 - np.power(-(2 - lmbda) * y[neg_mask] + 1, 1 / (2 - lmbda))
        else:
            result[neg_mask] = 1 - np.exp(-y[neg_mask])

        return result

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """拟合并变换"""
        return self.fit(y).transform(y)


class FallbackRegressor:
    """
    Simple regressor as fallback when FEAT model performs poorly.
    Uses lightweight methods that work well on difficult distributions.
    """

    def __init__(self, method='ridge', seed=42):
        self.method = method
        self.seed = seed
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the fallback model."""
        if self.method == 'ridge':
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=1.0, random_state=self.seed)
        elif self.method == 'knn':
            from sklearn.neighbors import KNeighborsRegressor
            n_neighbors = min(10, len(X_train) // 2)
            self.model = KNeighborsRegressor(n_neighbors=max(1, n_neighbors))
        elif self.method == 'mean':
            self.mean_y = np.mean(y_train)
            return self
        else:
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=1.0, random_state=self.seed)

        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict using fallback model."""
        if self.method == 'mean':
            return np.full(len(X_test), self.mean_y)
        return self.model.predict(X_test)


# ============================================================================
# Boosted Regressor
# ============================================================================

class BoostedRegressor:
    """
    Boosted regression predictor with multiple enhancement strategies.

    Strategies:
    - shuffle: Shuffle context order (original)
    - subsample: Random context subsampling
    - knn: KNN-based context selection per test sample

    Features:
    - Auto-detection of problematic predictions
    - Fallback to simple regressor when model is unreliable
    """

    def __init__(self, model, device, max_cells=500000, seed=42):
        self.model = model
        self.device = device
        self.max_cells = max_cells
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.use_fallback = False  # Will be set if model seems unreliable

    def predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        n_passes: int = 5,
        use_shuffle: bool = True,
        aggregation: str = 'median',
        strategy: str = 'shuffle',
        context_fraction: float = 0.8
    ) -> Tuple[np.ndarray, float, float]:
        """
        Multi-pass regression with robust aggregation.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            n_passes: Number of forward passes
            use_shuffle: Whether to shuffle training data order
            aggregation: 'median' (robust) or 'mean' or 'trimmed'
            strategy: 'shuffle' (reorder), 'subsample' (random subset), 'knn' (nearest neighbors)
            context_fraction: Fraction of training data to use (for subsample strategy)
        """
        y_train_float = y_train.astype(np.float32)

        # Global normalization stats (fixed across all passes)
        # IMPORTANT: Keep simple z-score to match training distribution
        mean_y = np.nanmean(y_train_float)
        std_y = np.nanstd(y_train_float)
        if std_y < 1e-6:
            std_y = 1e-6

        y_train_norm = (y_train_float - mean_y) / std_y

        all_preds = []

        if strategy == 'knn':
            # KNN-based context selection: for each test sample, use K nearest neighbors
            pred = self._knn_predict(X_train, y_train_norm, X_test, mean_y, std_y, n_neighbors=min(500, len(X_train)))
            if not np.any(np.isnan(pred)) and not np.any(np.isinf(pred)):
                return pred, mean_y, std_y
            # Fallback to shuffle if KNN fails
            strategy = 'shuffle'

        for i in range(n_passes):
            if strategy == 'subsample':
                # Random context subsampling
                n_ctx = min(len(X_train), max(10, int(len(X_train) * context_fraction)))
                indices = self.rng.choice(len(X_train), size=n_ctx, replace=False)
                X_train_pass = X_train[indices]
                y_norm_pass = y_train_norm[indices]
            elif use_shuffle and i > 0:
                # Shuffle training data order (but keep same normalization)
                perm = self.rng.permutation(len(X_train))
                X_train_pass = X_train[perm]
                y_norm_pass = y_train_norm[perm]
            else:
                X_train_pass = X_train
                y_norm_pass = y_train_norm

            pred = self._single_forward(
                X_train_pass, y_norm_pass, X_test, mean_y, std_y
            )

            if not np.any(np.isnan(pred)) and not np.any(np.isinf(pred)):
                all_preds.append(pred)

        if len(all_preds) == 0:
            pred = self._single_forward(X_train, y_train_norm, X_test, mean_y, std_y)
            return pred, mean_y, std_y

        all_preds = np.array(all_preds)

        if aggregation == 'median':
            y_pred = np.median(all_preds, axis=0)
        elif aggregation == 'trimmed':
            # Trimmed mean: remove top/bottom 20% of predictions per sample
            y_pred = self._trimmed_mean(all_preds, trim_fraction=0.2)
        else:
            y_pred = np.mean(all_preds, axis=0)

        # Store prediction variance for reliability assessment
        self.pred_variance = np.var(all_preds, axis=0) if len(all_preds) > 1 else np.zeros(len(y_pred))

        return y_pred, mean_y, std_y

    def predict_with_fallback(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        n_passes: int = 5,
        use_shuffle: bool = True,
        aggregation: str = 'median',
        strategy: str = 'shuffle',
        context_fraction: float = 0.8,
        fallback_method: str = 'ridge',
        blend_threshold: float = 0.3
    ) -> Tuple[np.ndarray, float, float, str]:
        """
        Predict with automatic fallback for unreliable predictions.

        This method:
        1. Runs FEAT model prediction
        2. Detects if predictions are unreliable (high variance, out of range)
        3. Computes fallback prediction using simple regressor
        4. Blends or switches based on reliability

        Args:
            ... (same as predict)
            fallback_method: 'ridge', 'knn', or 'mean'
            blend_threshold: variance threshold for blending (normalized)

        Returns:
            y_pred, mean_y, std_y, method_used ('feat', 'fallback', 'blend')
        """
        y_train_float = y_train.astype(np.float32)
        mean_y = np.nanmean(y_train_float)
        std_y = np.nanstd(y_train_float)
        if std_y < 1e-6:
            std_y = 1e-6

        # Get FEAT model predictions
        y_pred_feat, _, _ = self.predict(
            X_train, y_train, X_test,
            n_passes=n_passes,
            use_shuffle=use_shuffle,
            aggregation=aggregation,
            strategy=strategy,
            context_fraction=context_fraction
        )

        # Compute reliability metrics
        pred_var_norm = np.mean(self.pred_variance) / (std_y ** 2 + 1e-8)

        # Check if predictions are reasonable
        y_min, y_max = np.min(y_train_float), np.max(y_train_float)
        y_range = y_max - y_min + 1e-8
        out_of_range_ratio = np.mean(
            (y_pred_feat < y_min - 0.5 * y_range) |
            (y_pred_feat > y_max + 0.5 * y_range)
        )

        # Decision: use FEAT, fallback, or blend
        unreliable = (pred_var_norm > blend_threshold) or (out_of_range_ratio > 0.2)

        if unreliable:
            # Compute fallback prediction
            fallback = FallbackRegressor(method=fallback_method, seed=self.seed)
            fallback.fit(X_train, y_train_float)
            y_pred_fallback = fallback.predict(X_test)

            # Blend based on variance (per-sample)
            if len(self.pred_variance) == len(y_pred_feat):
                # Normalize variance to [0, 1] weight for fallback
                var_normalized = np.clip(self.pred_variance / (std_y ** 2 + 1e-8), 0, 1)
                blend_weight = np.clip(var_normalized / blend_threshold, 0, 0.5)
                y_pred = (1 - blend_weight) * y_pred_feat + blend_weight * y_pred_fallback
                method_used = 'blend'
            else:
                # Fallback entirely if can't compute per-sample weights
                y_pred = y_pred_fallback
                method_used = 'fallback'
        else:
            y_pred = y_pred_feat
            method_used = 'feat'

        return y_pred, mean_y, std_y, method_used

    def predict_with_distillation(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        n_passes: int = 5,
        use_shuffle: bool = True,
        aggregation: str = 'median',
        max_context_samples: int = None
    ) -> Tuple[np.ndarray, float, float, bool]:
        """
        使用数据蒸馏的预测方法。

        当训练数据量超过 max_cells 限制时，使用智能采样选择最有代表性的训练样本。
        基于 query (测试集) 子空间投影，选择与测试集相关的训练样本。

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
            n_passes: 推理次数
            use_shuffle: 是否打乱顺序
            aggregation: 聚合方法
            max_context_samples: 最大上下文样本数，默认根据 max_cells 计算

        Returns:
            y_pred: 预测值
            mean_y: 训练标签均值
            std_y: 训练标签标准差
            used_distillation: 是否使用了蒸馏
        """
        n_train = len(X_train)
        n_features = X_train.shape[1]

        # 计算最大可用的上下文样本数
        if max_context_samples is None:
            # 基于 max_cells 和特征数计算
            max_context_samples = max(100, int(self.max_cells / max(1, n_features)) - len(X_test))
            max_context_samples = min(max_context_samples, 5000)  # 上限

        # 判断是否需要蒸馏
        need_distillation = n_train > max_context_samples

        if need_distillation:
            # 转换为 torch tensor 进行蒸馏
            X_train_t = torch.from_numpy(X_train.astype(np.float32)).to(self.device)
            y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(self.device)
            X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(self.device)

            # 使用 v14_reg 进行智能采样
            X_distilled, y_distilled, _ = distill_training_data(
                X_train_t, y_train_t, X_test_t,
                max_samples=max_context_samples,
                method='v14_reg'
            )

            # 转回 numpy
            X_train_used = X_distilled.cpu().numpy()
            y_train_used = y_distilled.cpu().numpy()
        else:
            X_train_used = X_train
            y_train_used = y_train

        # 正常预测
        y_pred, mean_y, std_y = self.predict(
            X_train_used, y_train_used, X_test,
            n_passes=n_passes,
            use_shuffle=use_shuffle,
            aggregation=aggregation,
            strategy='shuffle'
        )

        return y_pred, mean_y, std_y, need_distillation

    def _knn_predict(
        self,
        X_train: np.ndarray,
        y_train_norm: np.ndarray,
        X_test: np.ndarray,
        mean_y: float,
        std_y: float,
        n_neighbors: int = 500
    ) -> np.ndarray:
        """
        KNN-based prediction: for each test sample, use its K nearest training samples as context.
        This ensures the model sees the most relevant context for each prediction.
        """
        from sklearn.neighbors import NearestNeighbors

        n_test = len(X_test)
        n_train = len(X_train)
        k = min(n_neighbors, n_train)

        # Fit KNN on training data
        knn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
        knn.fit(X_train)

        # Find K nearest neighbors for each test sample
        distances, indices = knn.kneighbors(X_test)

        all_preds = []

        # Process in batches to avoid memory issues
        batch_size = 100
        for batch_start in range(0, n_test, batch_size):
            batch_end = min(batch_start + batch_size, n_test)
            batch_indices = indices[batch_start:batch_end]
            batch_X_test = X_test[batch_start:batch_end]

            # For each test sample in batch, use its own context
            batch_preds = []
            for i, (test_x, ctx_idx) in enumerate(zip(batch_X_test, batch_indices)):
                X_ctx = X_train[ctx_idx]
                y_ctx = y_train_norm[ctx_idx]

                # Single test sample prediction
                pred = self._single_forward(
                    X_ctx, y_ctx, test_x.reshape(1, -1), mean_y, std_y
                )
                batch_preds.append(pred[0])

            all_preds.extend(batch_preds)

        return np.array(all_preds)

    def _trimmed_mean(self, all_preds: np.ndarray, trim_fraction: float = 0.2) -> np.ndarray:
        """Compute trimmed mean along axis 0 (across passes)."""
        n_passes = all_preds.shape[0]
        n_trim = max(1, int(n_passes * trim_fraction))

        if n_passes <= 2 * n_trim:
            # Not enough passes to trim, use median
            return np.median(all_preds, axis=0)

        # Sort predictions for each test sample
        sorted_preds = np.sort(all_preds, axis=0)
        # Remove top and bottom n_trim predictions
        trimmed = sorted_preds[n_trim:-n_trim, :]
        return np.mean(trimmed, axis=0)

    def _single_forward(
        self,
        X_ctx: np.ndarray,
        y_ctx_norm: np.ndarray,
        X_test: np.ndarray,
        mean_y: float,
        std_y: float
    ) -> np.ndarray:
        """Single forward pass."""
        n_ctx = len(X_ctx)
        n_test = len(X_test)
        n_features = X_ctx.shape[1]

        ctx_mean = np.mean(y_ctx_norm)
        ctx_std = np.std(y_ctx_norm)
        if ctx_std < 1e-6:
            ctx_std = 1e-6

        X_all = np.concatenate([X_ctx, X_test], axis=0)
        x_tensor = torch.from_numpy(X_all.astype(np.float32)).to(self.device)
        y_tensor = torch.from_numpy(y_ctx_norm.astype(np.float32)).to(self.device)

        seq_len_limit = max(1000, int(self.max_cells / max(1, n_features)))
        seq_len_limit = max(seq_len_limit, 2048)

        amp_dtype = torch.bfloat16 if (self.device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type == 'cuda',
            dtype=amp_dtype,
        ), torch.inference_mode():
            n_total = n_ctx + n_test

            if n_total <= seq_len_limit:
                x_batch = x_tensor.unsqueeze(0)
                y_batch = y_tensor.unsqueeze(0)
                out = self.model(x=x_batch, y=y_batch, eval_pos=n_ctx, task_type='reg')
                pred = out['reg_output'] if isinstance(out, dict) else out
                pred = pred.squeeze(0)
                if pred.dim() == 2:
                    pred = pred.squeeze(-1)
            else:
                target_ctx_len = int(seq_len_limit * 0.9)
                actual_ctx = min(n_ctx, target_ctx_len)

                x_ctx_eff = x_tensor[:n_ctx][-actual_ctx:]
                y_ctx_eff = y_tensor[-actual_ctx:]

                y_ctx_eff_np = y_ctx_norm[-actual_ctx:]
                ctx_mean = np.mean(y_ctx_eff_np)
                ctx_std = np.std(y_ctx_eff_np)
                if ctx_std < 1e-6:
                    ctx_std = 1e-6

                space_for_query = max(1, seq_len_limit - actual_ctx)
                x_test_tensor = x_tensor[n_ctx:]
                chunk_outputs = []

                for j in range(0, n_test, space_for_query):
                    chunk_size = min(space_for_query, n_test - j)
                    x_chunk = x_test_tensor[j:j + chunk_size]

                    batch_x = torch.cat([x_ctx_eff, x_chunk], dim=0).unsqueeze(0)
                    batch_y = y_ctx_eff.unsqueeze(0)

                    out = self.model(x=batch_x, y=batch_y, eval_pos=actual_ctx, task_type='reg')
                    out = out['reg_output'] if isinstance(out, dict) else out
                    out = out.squeeze(0)
                    if out.dim() == 2:
                        out = out.squeeze(-1)

                    if out.shape[0] < chunk_size:
                        pad = torch.zeros(chunk_size - out.shape[0], device=out.device, dtype=out.dtype)
                        out = torch.cat([out, pad], dim=0)
                    else:
                        out = out[:chunk_size]
                    chunk_outputs.append(out)

                pred = torch.cat(chunk_outputs, dim=0) if chunk_outputs else torch.zeros(n_test, device=self.device)

        pred_np = pred.float().cpu().numpy()

        # Denormalize
        pred_first_norm = pred_np * ctx_std + ctx_mean
        pred_original = pred_first_norm * std_y + mean_y

        return pred_original


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_regression(y_true, y_pred, mean_y, std_y):
    """Calculate regression metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    y_true_norm = (y_true - mean_y) / std_y
    y_pred_norm = (y_pred - mean_y) / std_y
    rmse_norm = np.sqrt(mean_squared_error(y_true_norm, y_pred_norm))

    return {'r2': r2, 'rmse': rmse, 'rmse_norm': rmse_norm}


def get_all_datasets(base_path: str) -> List[str]:
    """Get all regression dataset directories."""
    datasets = []
    base = Path(base_path)

    if not base.exists():
        return datasets

    for sub_benchmark in base.iterdir():
        if sub_benchmark.is_dir() and not sub_benchmark.name.startswith('.'):
            for dataset_dir in sub_benchmark.iterdir():
                if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
                    has_train = (dataset_dir / 'train.csv').exists() or list(dataset_dir.glob('*_train.csv'))
                    if has_train:
                        datasets.append(str(dataset_dir))

    return sorted(datasets)


def evaluate_single_dataset(
    predictor: BoostedRegressor,
    dataset_path: str,
    n_passes: int,
    use_shuffle: bool,
    aggregation: str,
    strategy: str = 'shuffle',
    context_fraction: float = 0.8,
    use_hybrid: bool = False,
    fallback_method: str = 'ridge',
    use_distillation: bool = False,
    max_context_samples: int = None,
    target_transform: str = 'none'
) -> Dict[str, Any]:
    """Evaluate a single dataset."""
    dataset_name = os.path.basename(dataset_path)
    benchmark_name = os.path.basename(os.path.dirname(dataset_path))

    result = {
        'dataset': dataset_name,
        'benchmark': benchmark_name,
        'status': 'success',
        'error': None
    }

    try:
        X_train, y_train, X_test, y_test = load_dataset(dataset_path)
        X_train_norm, X_test_norm = preprocess_features(
            pd.DataFrame(X_train) if isinstance(X_train, np.ndarray) else X_train,
            pd.DataFrame(X_test) if isinstance(X_test, np.ndarray) else X_test
        )

        y_train_np = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

        # Apply target transform if specified
        transformer = None
        if target_transform != 'none':
            transformer = TargetTransformer(method=target_transform)
            y_train_transformed = transformer.fit_transform(y_train_np).astype(np.float32)
            result['target_transform'] = transformer.fitted_method
        else:
            y_train_transformed = y_train_np.astype(np.float32)

        result['n_train'] = len(y_train_np)
        result['n_test'] = len(y_test_np)
        result['n_features'] = X_train_norm.shape[1]

        start_time = time.time()

        if use_distillation:
            # 使用数据蒸馏进行预测
            y_pred, mean_y, std_y, used_distill = predictor.predict_with_distillation(
                X_train_norm, y_train_transformed, X_test_norm,
                n_passes=n_passes,
                use_shuffle=use_shuffle,
                aggregation=aggregation,
                max_context_samples=max_context_samples
            )
            result['method_used'] = 'distill' if used_distill else 'feat'
        elif use_hybrid:
            # Use hybrid method with automatic fallback
            y_pred, mean_y, std_y, method_used = predictor.predict_with_fallback(
                X_train_norm, y_train_transformed, X_test_norm,
                n_passes=n_passes,
                use_shuffle=use_shuffle,
                aggregation=aggregation,
                strategy=strategy,
                context_fraction=context_fraction,
                fallback_method=fallback_method
            )
            result['method_used'] = method_used
        else:
            y_pred, mean_y, std_y = predictor.predict(
                X_train_norm, y_train_transformed, X_test_norm,
                n_passes=n_passes,
                use_shuffle=use_shuffle,
                aggregation=aggregation,
                strategy=strategy,
                context_fraction=context_fraction
            )

        # Inverse transform predictions if transformer was used
        if transformer is not None:
            y_pred = transformer.inverse_transform(y_pred)
            mean_y = np.mean(y_train_np)
            std_y = np.std(y_train_np)

        metrics = evaluate_regression(y_test_np.astype(np.float32), y_pred, mean_y, std_y)
        result['inference_time'] = time.time() - start_time
        result.update(metrics)

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)

    return result


def print_results_table(results: List[Dict], rank: int):
    """Print results summary (only on rank 0)."""
    if not is_main_process(rank):
        return

    if not results:
        print("\nNo results to display.\n")
        return

    benchmarks = defaultdict(list)
    for r in results:
        benchmarks[r['benchmark']].append(r)

    print(f"\n{'='*120}")
    print(f"  FEAT REGRESSION RESULTS")
    print(f"{'='*120}")

    all_benchmark_summaries = []

    for benchmark_name in sorted(benchmarks.keys()):
        benchmark_results = benchmarks[benchmark_name]

        print(f"\n{'─'*135}")
        print(f"  BENCHMARK: {benchmark_name}")
        print(f"{'─'*135}")
        print(f"{'Dataset':<25} {'N_train':>8} {'N_test':>7} {'N_feat':>6} {'R2':>8} {'RMSE_norm':>10} {'Time':>7} {'Method':<20} {'Status':<8}")
        print('-'*135)

        for r in sorted(benchmark_results, key=lambda x: x['dataset']):
            n_train = f"{r.get('n_train', 0)}" if r['status'] == 'success' else 'N/A'
            n_test = f"{r.get('n_test', 0)}" if r['status'] == 'success' else 'N/A'
            n_feat = f"{r.get('n_features', 0)}" if r['status'] == 'success' else 'N/A'
            r2 = f"{r.get('r2', 0):.4f}" if r['status'] == 'success' else 'N/A'
            rmse_norm = f"{r.get('rmse_norm', 0):.4f}" if r['status'] == 'success' else 'N/A'
            time_s = f"{r.get('inference_time', 0):.1f}s" if r['status'] == 'success' else 'N/A'
            method = r.get('method_used', 'standard')[:20] if r['status'] == 'success' else 'N/A'
            print(f"{r['dataset'][:25]:<25} {n_train:>8} {n_test:>7} {n_feat:>6} {r2:>8} {rmse_norm:>10} {time_s:>7} {method:<20} {r['status']:<8}")

        successful = [r for r in benchmark_results if r['status'] == 'success']
        if successful:
            avg_r2 = np.mean([r['r2'] for r in successful])
            avg_rmse_norm = np.mean([r['rmse_norm'] for r in successful])

            print('-'*135)
            print(f"{'>>> ' + benchmark_name + ' AVERAGE':<42} {avg_r2:>8.4f} {avg_rmse_norm:>10.4f}")
            print(f"    Success: {len(successful)}/{len(benchmark_results)} datasets")

            all_benchmark_summaries.append({
                'benchmark': benchmark_name,
                'r2': avg_r2,
                'rmse_norm': avg_rmse_norm,
                'success': len(successful),
                'total': len(benchmark_results)
            })

    # Overall summary
    print(f"\n{'='*135}")
    print(f"  FEAT REGRESSION SUMMARY")
    print(f"{'='*135}")
    print(f"{'Benchmark':<35} {'R2':>12} {'RMSE_norm':>12} {'Success':>15}")
    print('-'*135)

    for s in all_benchmark_summaries:
        print(f"{s['benchmark']:<35} {s['r2']:>12.4f} {s['rmse_norm']:>12.4f} {s['success']:>6}/{s['total']:<8}")

    if all_benchmark_summaries:
        overall_r2 = np.mean([s['r2'] for s in all_benchmark_summaries])
        overall_rmse_norm = np.mean([s['rmse_norm'] for s in all_benchmark_summaries])
        total_success = sum(s['success'] for s in all_benchmark_summaries)
        total_datasets = sum(s['total'] for s in all_benchmark_summaries)

        print('-'*135)
        print(f"{'OVERALL AVERAGE':<35} {overall_r2:>12.4f} {overall_rmse_norm:>12.4f} {total_success:>6}/{total_datasets:<8}")

    print(f"{'='*135}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate FEAT on regression benchmarks (DDP supported)'
    )
    parser.add_argument('--model_path', default=str(DEFAULT_MODEL_PATH))
    parser.add_argument('--eval_reg_data_path', default='data/regression')
    parser.add_argument('--max_cells', type=int, default=2_000_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_passes', type=int, default=5)
    parser.add_argument('--aggregation', default='median', choices=['median', 'mean', 'trimmed'])
    parser.add_argument('--strategy', default='shuffle', choices=['shuffle', 'subsample', 'knn'])
    parser.add_argument('--context_fraction', type=float, default=0.8)
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--benchmarks', nargs='+', default=None)
    parser.add_argument('--use_hybrid', action='store_true')
    parser.add_argument('--fallback_method', default='ridge', choices=['ridge', 'knn', 'mean'])
    parser.add_argument('--use_distillation', action='store_true')
    parser.add_argument('--max_context_samples', type=int, default=None)
    parser.add_argument(
        '--target_transform',
        default='none',
        choices=['none', 'auto', 'yeo-johnson', 'quantile', 'log', 'sqrt'],
    )
    args = parser.parse_args()

    if args.n_passes < 1:
        raise ValueError('--n_passes must be at least 1')
    if not 0 < args.context_fraction <= 1:
        raise ValueError('--context_fraction must be in (0, 1]')

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    data_path = Path(args.eval_reg_data_path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not model_path.is_file():
        raise FileNotFoundError(f"FEAT checkpoint not found: {model_path}")

    all_datasets = get_all_datasets(str(data_path))
    if args.benchmarks:
        all_datasets = [
            dataset for dataset in all_datasets
            if any(name in dataset for name in args.benchmarks)
        ]
    if not all_datasets:
        raise FileNotFoundError(f"No regression datasets found under: {data_path}")

    rank, world_size, is_distributed = setup_distributed()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = (
        torch.device(f'cuda:{local_rank}')
        if is_distributed
        else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    print_rank0(f"[FEAT] Regression datasets: {len(all_datasets)}", rank)
    print_rank0(f"[FEAT] Checkpoint: {model_path}", rank)
    model = load_model(model_path=model_path, mask_prediction=False)
    model.to(device)

    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    predictor = BoostedRegressor(model, device, args.max_cells, args.seed + rank)
    local_results = []

    for dataset_path in distribute_datasets(all_datasets, rank, world_size):
        result = evaluate_single_dataset(
            predictor,
            dataset_path,
            n_passes=args.n_passes,
            use_shuffle=not args.no_shuffle,
            aggregation=args.aggregation,
            strategy=args.strategy,
            context_fraction=args.context_fraction,
            use_hybrid=args.use_hybrid,
            fallback_method=args.fallback_method,
            use_distillation=args.use_distillation,
            max_context_samples=args.max_context_samples,
            target_transform=args.target_transform,
        )
        local_results.append(result)
        if result['status'] == 'success':
            print(
                f"[Rank {rank}] {result['benchmark']}/{result['dataset']}: "
                f"R2={result['r2']:.4f}, RMSE_norm={result['rmse_norm']:.4f}"
            )
        else:
            print(f"[Rank {rank}] {result['benchmark']}/{result['dataset']}: FAILED ({result['error']})")

    if is_distributed:
        dist.barrier()
    all_results = gather_results(local_results, rank, world_size)
    if is_main_process(rank):
        all_results.sort(key=lambda item: (item['benchmark'], item['dataset']))
        print_results_table(all_results, rank)
    cleanup_distributed()


if __name__ == '__main__':
    main()
