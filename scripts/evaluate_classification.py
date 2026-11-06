import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Fix OpenBLAS thread allocation issue
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feat.inference.data_distillation import distill_training_data
from feat.utils.loading import load_model


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
# Data Loading & Preprocessing
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
# FEAT Multi-Pass Classifier
# ============================================================================

class BoostedClassifier:
    """
    Boosted classification predictor with multi-pass averaging and context sub-sampling.
    """

    def __init__(self, model, device, max_cells=500000, seed=42):
        self.model = model
        self.device = device
        self.max_cells = max_cells
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def predict_proba(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        n_passes: int = 5,
        use_shuffle: bool = True,
        aggregation: str = 'mean',
        strategy: str = 'shuffle',
        context_fraction: float = 0.8
    ) -> np.ndarray:
        """
        Multi-pass classification with probability averaging.
        """
        y_train_int = y_train.astype(np.int64)
        num_classes = len(np.unique(y_train_int))
        
        all_probs = []

        if strategy == 'knn':
            probs = self._knn_predict(X_train, y_train_int, X_test, num_classes, n_neighbors=min(500, len(X_train)))
            if probs is not None:
                return probs
            strategy = 'shuffle'

        for i in range(n_passes):
            if strategy == 'subsample':
                n_ctx = min(len(X_train), max(10, int(len(X_train) * context_fraction)))
                indices = self.rng.choice(len(X_train), size=n_ctx, replace=False)
                X_train_pass = X_train[indices]
                y_pass = y_train_int[indices]
            elif use_shuffle and i > 0:
                perm = self.rng.permutation(len(X_train))
                X_train_pass = X_train[perm]
                y_pass = y_train_int[perm]
            else:
                X_train_pass = X_train
                y_pass = y_train_int

            probs = self._single_forward(
                X_train_pass, y_pass, X_test, num_classes
            )

            if not np.any(np.isnan(probs)):
                all_probs.append(probs)

        if len(all_probs) == 0:
            return self._single_forward(X_train, y_train_int, X_test, num_classes)

        # Average probabilities
        all_probs_np = np.array(all_probs)
        
        if aggregation == 'trimmed':
            avg_probs = self._trimmed_mean(all_probs_np, trim_fraction=0.2)
        elif aggregation == 'median':
            avg_probs = np.median(all_probs_np, axis=0)
        else:
            avg_probs = np.mean(all_probs_np, axis=0)

        # Re-normalize just in case
        avg_probs = avg_probs / (np.sum(avg_probs, axis=-1, keepdims=True) + 1e-8)

        self.pred_variance = np.var(all_probs_np, axis=0) if len(all_probs) > 1 else np.zeros_like(avg_probs)
        return avg_probs

    def _trimmed_mean(self, all_probs: np.ndarray, trim_fraction: float = 0.2) -> np.ndarray:
        """Compute trimmed mean along axis 0 (across passes)."""
        n_passes = all_probs.shape[0]
        n_trim = max(1, int(n_passes * trim_fraction))

        if n_passes <= 2 * n_trim:
            return np.median(all_probs, axis=0)

        sorted_probs = np.sort(all_probs, axis=0)
        trimmed = sorted_probs[n_trim:-n_trim, ...]
        return np.mean(trimmed, axis=0)

    def predict_with_distillation(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        n_passes: int = 5,
        use_shuffle: bool = True,
        max_context_samples: int = None
    ) -> Tuple[np.ndarray, bool]:
        """
        Distillation method for large context. We use regression distillation method v14_reg 
        as an approximation for classifying important samples.
        """
        n_train = len(X_train)
        n_features = X_train.shape[1]

        if max_context_samples is None:
            max_context_samples = max(100, int(self.max_cells / max(1, n_features)) - len(X_test))
            max_context_samples = min(max_context_samples, 5000)

        need_distillation = n_train > max_context_samples

        if need_distillation:
            X_train_t = torch.from_numpy(X_train.astype(np.float32)).to(self.device)
            y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(self.device)
            X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(self.device)

            # Using v14_reg as a surrogate for sample selection based on features
            try:
                X_distilled, y_distilled, _ = distill_training_data(
                    X_train_t, y_train_t, X_test_t,
                    max_samples=max_context_samples,
                    method='v14_reg'
                )
                X_train_used = X_distilled.cpu().numpy()
                y_train_used = y_distilled.cpu().numpy()
            except Exception:
                # fallback to random selection
                idx = np.random.choice(len(X_train), max_context_samples, replace=False)
                X_train_used = X_train[idx]
                y_train_used = y_train[idx]
        else:
            X_train_used = X_train
            y_train_used = y_train

        y_proba = self.predict_proba(
            X_train_used, y_train_used, X_test,
            n_passes=n_passes,
            use_shuffle=use_shuffle,
            aggregation='mean',
            strategy='shuffle'
        )

        return y_proba, need_distillation

    def _knn_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        num_classes: int,
        n_neighbors: int = 500
    ) -> np.ndarray:
        """
        KNN-based context selection per test sample.
        """
        from sklearn.neighbors import NearestNeighbors

        n_test = len(X_test)
        n_train = len(X_train)
        k = min(n_neighbors, n_train)

        knn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
        knn.fit(X_train)
        distances, indices = knn.kneighbors(X_test)

        all_probs = []

        batch_size = 100
        for batch_start in range(0, n_test, batch_size):
            batch_end = min(batch_start + batch_size, n_test)
            batch_indices = indices[batch_start:batch_end]
            batch_X_test = X_test[batch_start:batch_end]

            batch_preds = []
            for i, (test_x, ctx_idx) in enumerate(zip(batch_X_test, batch_indices)):
                X_ctx = X_train[ctx_idx]
                y_ctx = y_train[ctx_idx]
                
                probs = self._single_forward(
                    X_ctx, y_ctx, test_x.reshape(1, -1), num_classes
                )
                batch_preds.append(probs[0])
            all_probs.extend(batch_preds)

        return np.array(all_probs)

    def _single_forward(
        self,
        X_ctx: np.ndarray,
        y_ctx: np.ndarray,
        X_test: np.ndarray,
        num_classes: int
    ) -> np.ndarray:
        """Single forward pass for classification."""
        n_ctx = len(X_ctx)
        n_test = len(X_test)
        n_features = X_ctx.shape[1]

        X_all = np.concatenate([X_ctx, X_test], axis=0)
        x_tensor = torch.from_numpy(X_all.astype(np.float32)).to(self.device)
        y_tensor = torch.from_numpy(y_ctx.astype(np.int64)).to(self.device)

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
                out = self.model(x=x_batch, y=y_batch, eval_pos=n_ctx, task_type='cls')
                pred = out['cls_output'] if isinstance(out, dict) else out
                pred = pred.squeeze(0)
            else:
                target_ctx_len = int(seq_len_limit * 0.9)
                actual_ctx = min(n_ctx, target_ctx_len)

                x_ctx_eff = x_tensor[:n_ctx][-actual_ctx:]
                y_ctx_eff = y_tensor[-actual_ctx:]

                space_for_query = max(1, seq_len_limit - actual_ctx)
                x_test_tensor = x_tensor[n_ctx:]
                chunk_outputs = []

                for chunk_start in range(0, n_test, space_for_query):
                    chunk_end = min(n_test, chunk_start + space_for_query)
                    x_query = x_test_tensor[chunk_start:chunk_end]
                    x_batch = torch.cat([x_ctx_eff, x_query], dim=0).unsqueeze(0)
                    y_batch = y_ctx_eff.unsqueeze(0)

                    out = self.model(x=x_batch, y=y_batch, eval_pos=actual_ctx, task_type='cls')
                    out = out['cls_output'] if isinstance(out, dict) else out
                    out = out.squeeze(0)
                    
                    chunk_size = chunk_end - chunk_start
                    if len(out) < chunk_size:
                        pad = torch.zeros((chunk_size - len(out), out.shape[-1]), device=self.device)
                        out = torch.cat([out, pad], dim=0)
                    else:
                        out = out[:chunk_size]
                    chunk_outputs.append(out)

                pred = torch.cat(chunk_outputs, dim=0) if chunk_outputs else torch.zeros((n_test, num_classes), device=self.device)

        pred_np = pred.float().cpu().numpy()
        pred_logits = pred_np[:, :num_classes]
        
        # Softmax
        exp_logits = np.exp(pred_logits - np.max(pred_logits, axis=-1, keepdims=True))
        y_proba = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-8)

        return y_proba


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_classification(y_true: np.ndarray, y_proba: np.ndarray):
    """Calculate classification metrics."""
    y_pred = np.argmax(y_proba, axis=-1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    num_classes = y_proba.shape[1]
    
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
    except Exception:
        auc = 0.0

    return {'acc': acc, 'auc': auc, 'f1': f1}


def get_all_datasets(base_path: str) -> List[str]:
    """Get all classification dataset directories."""
    datasets = []
    base = Path(base_path)

    if not base.exists():
        return datasets

    for sub_benchmark in base.iterdir():
        if sub_benchmark.is_dir() and not sub_benchmark.name.startswith('.'):
            for dataset_dir in sub_benchmark.iterdir():
                if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
                    datasets.append(str(dataset_dir))

    return sorted(datasets)


def evaluate_single_dataset(
    predictor: BoostedClassifier,
    dataset_path: str,
    n_passes: int,
    use_shuffle: bool,
    aggregation: str,
    strategy: str,
    context_fraction: float,
    use_distillation: bool,
    max_context_samples: int = None,
) -> Dict[str, Any]:
    """Evaluate one dataset without using test targets for model selection."""
    dataset_name = os.path.basename(dataset_path)
    benchmark_name = os.path.basename(os.path.dirname(dataset_path))
    result = {
        'dataset': dataset_name,
        'benchmark': benchmark_name,
        'status': 'success',
        'error': None,
    }

    try:
        X_train, y_train, X_test, y_test = load_dataset(dataset_path)
        X_train_norm, X_test_norm = preprocess_features(X_train, X_test)
        y_train_np = np.asarray(y_train)
        y_test_np = np.asarray(y_test)

        classes, y_train_idx = np.unique(y_train_np, return_inverse=True)
        class_to_idx = {value: index for index, value in enumerate(classes)}
        y_test_idx = np.array([class_to_idx.get(value, -1) for value in y_test_np])
        valid_mask = y_test_idx >= 0
        y_test_idx = y_test_idx[valid_mask]
        X_test_norm = X_test_norm[valid_mask]
        if len(y_test_idx) == 0:
            raise ValueError("The test split contains no classes observed in the training split.")

        start_time = time.time()
        if use_distillation:
            y_proba, distilled = predictor.predict_with_distillation(
                X_train_norm,
                y_train_idx,
                X_test_norm,
                n_passes=n_passes,
                use_shuffle=use_shuffle,
                max_context_samples=max_context_samples,
            )
            method = 'distilled' if distilled else 'standard'
        else:
            y_proba = predictor.predict_proba(
                X_train_norm,
                y_train_idx,
                X_test_norm,
                n_passes=n_passes,
                use_shuffle=use_shuffle,
                aggregation=aggregation,
                strategy=strategy,
                context_fraction=context_fraction,
            )
            method = strategy

        result.update(evaluate_classification(y_test_idx, y_proba))
        result.update({
            'n_train': len(y_train_idx),
            'n_test': len(y_test_idx),
            'n_features': X_train_norm.shape[1],
            'inference_time': time.time() - start_time,
            'method_used': method,
        })
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = str(exc)

    return result


def print_results_table(results: List[Dict]):
    """Print dataset-level and aggregate classification metrics."""
    print(f"\n{'=' * 112}")
    print(f"{'Benchmark':<28} {'Dataset':<32} {'ACC':>9} {'AUC':>9} {'F1':>9} {'Time(s)':>10}")
    print('-' * 112)
    for result in results:
        if result['status'] == 'success':
            print(
                f"{result['benchmark']:<28} {result['dataset']:<32} "
                f"{result['acc']:>9.4f} {result['auc']:>9.4f} {result['f1']:>9.4f} "
                f"{result['inference_time']:>10.2f}"
            )
        else:
            print(f"{result['benchmark']:<28} {result['dataset']:<32} FAILED: {result['error']}")

    successful = [result for result in results if result['status'] == 'success']
    if successful:
        print('-' * 112)
        print(
            f"{'Overall mean':<61} "
            f"{np.mean([r['acc'] for r in successful]):>9.4f} "
            f"{np.mean([r['auc'] for r in successful]):>9.4f} "
            f"{np.mean([r['f1'] for r in successful]):>9.4f}"
        )
    print(f"{'=' * 112}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate FEAT on classification benchmarks (DDP supported)'
    )
    parser.add_argument('--model_path', default=str(DEFAULT_MODEL_PATH))
    parser.add_argument('--eval_cls_data_path', default='data/classification')
    parser.add_argument('--max_cells', type=int, default=500_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_passes', type=int, default=5)
    parser.add_argument('--aggregation', default='mean', choices=['mean', 'trimmed', 'median'])
    parser.add_argument('--strategy', default='shuffle', choices=['shuffle', 'subsample', 'knn'])
    parser.add_argument('--context_fraction', type=float, default=0.8)
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--benchmarks', nargs='+', default=None)
    parser.add_argument('--use_distillation', action='store_true')
    parser.add_argument('--max_context_samples', type=int, default=None)
    args = parser.parse_args()

    if args.n_passes < 1:
        raise ValueError('--n_passes must be at least 1')
    if not 0 < args.context_fraction <= 1:
        raise ValueError('--context_fraction must be in (0, 1]')

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    data_path = Path(args.eval_cls_data_path)
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
        raise FileNotFoundError(f"No classification datasets found under: {data_path}")

    rank, world_size, is_distributed = setup_distributed()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = (
        torch.device(f'cuda:{local_rank}')
        if is_distributed
        else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    print_rank0(f"[FEAT] Classification datasets: {len(all_datasets)}", rank)
    print_rank0(f"[FEAT] Checkpoint: {model_path}", rank)
    model = load_model(model_path=model_path, mask_prediction=False)
    model.to(device)

    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    predictor = BoostedClassifier(model, device, args.max_cells, args.seed + rank)
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
            use_distillation=args.use_distillation,
            max_context_samples=args.max_context_samples,
        )
        local_results.append(result)
        if result['status'] == 'success':
            print(
                f"[Rank {rank}] {result['benchmark']}/{result['dataset']}: "
                f"ACC={result['acc']:.4f}, AUC={result['auc']:.4f}, F1={result['f1']:.4f}"
            )
        else:
            print(f"[Rank {rank}] {result['benchmark']}/{result['dataset']}: FAILED ({result['error']})")

    if is_distributed:
        dist.barrier()
    all_results = gather_results(local_results, rank, world_size)
    if is_main_process(rank):
        all_results.sort(key=lambda item: (item['benchmark'], item['dataset']))
        print_results_table(all_results)
    cleanup_distributed()


if __name__ == '__main__':
    main()
