"""
Data Distillation Module for FEAT Regression
=============================================
用于处理无法全部塞入 max_cell 限制的大数据集，
通过智能采样选择最有代表性的训练样本。

核心思想：
- 基于 query (测试集) 子空间投影，选择与测试集相关的训练样本
- 结合均匀采样、覆盖度、对齐度三种概率

使用场景：
1. 数据量超过 max_cell 限制
2. R2 < 0.1 的问题数据集
"""

import torch


def compute_query_alignment(X_train: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
    """
    计算每个训练样本与测试集的对齐分数。

    原理：对于每个训练样本，计算它到所有测试样本的平均距离的倒数。
    距离越近，对齐分数越高，说明该训练样本对预测测试集更有帮助。

    Args:
        X_train: 训练特征 [N_train, F]
        X_test: 测试特征 [N_test, F]

    Returns:
        alignment: 每个训练样本的对齐分数 [N_train]，已归一化到 [0, 1]
    """
    chunk_size = min(500, X_test.shape[0])
    alignment = torch.zeros(X_train.shape[0], device=X_train.device, dtype=X_train.dtype)

    for i in range(0, X_test.shape[0], chunk_size):
        q_chunk = X_test[i:i + chunk_size]
        dists = torch.cdist(q_chunk, X_train)
        mean_dist = dists.mean(dim=0)
        alignment += 1.0 / (mean_dist + 1e-6)

    # Min-max 归一化
    alignment = (alignment - alignment.min()) / (alignment.max() - alignment.min() + 1e-8)
    return alignment


def select_weighted_random(x_train: torch.Tensor, x_test: torch.Tensor, K: int, query_weight: float = 0.3) -> torch.Tensor:
    """
    加权随机采样（fallback 方法）。

    当 SVD 失败时使用，仅基于对齐分数和均匀分布的混合。

    Args:
        x_train: 训练特征 [N, F]
        x_test: 测试特征 [M, F]
        K: 要选择的样本数
        query_weight: 对齐分数的权重 (0-1)

    Returns:
        idx: 选中的训练样本索引 [K]
    """
    N = x_train.shape[0]
    K = min(K, N)

    q_align = compute_query_alignment(x_train, x_test)
    align_prob = q_align / (q_align.sum() + 1e-8)
    base_prob = torch.ones(N, device=x_train.device, dtype=x_train.dtype) / N

    sample_prob = (1 - query_weight) * base_prob + query_weight * align_prob
    sample_prob = sample_prob / sample_prob.sum()
    idx = torch.multinomial(sample_prob, K, replacement=False)
    return idx


def select_v9_boosted_qsubspace(x_train: torch.Tensor, x_test: torch.Tensor, K: int, n_components: int = 10) -> torch.Tensor:
    """
    基于 Query 子空间的增强采样 (V9)。

    原理：
    1. 对测试集做 PCA，得到测试集的主方向 V
    2. 将训练集投影到这个子空间，计算覆盖度分数
    3. 结合均匀采样、覆盖度、对齐度进行采样

    Args:
        x_train: 训练特征 [N, F]
        x_test: 测试特征 [M, F]
        K: 要选择的样本数
        n_components: SVD 的成分数

    Returns:
        idx: 选中的训练样本索引 [K]
    """
    N, F = x_train.shape
    K = min(K, N)

    query_mean = x_test.mean(dim=0, keepdim=True)
    X_qry_centered = x_test - query_mean
    n_comp = min(n_components, min(x_test.shape[0] - 1, F - 1))
    n_comp = max(n_comp, 1)

    try:
        U, S, V = torch.svd_lowrank(X_qry_centered, q=n_comp)
        X_train_centered = x_train - query_mean
        X_proj = X_train_centered @ V
        coverage_score = X_proj.norm(dim=-1)
        coverage_prob = coverage_score / (coverage_score.sum() + 1e-8)
    except Exception:
        return select_weighted_random(x_train, x_test, K, query_weight=0.3)

    q_align = compute_query_alignment(x_train, x_test)
    align_prob = q_align / (q_align.sum() + 1e-8)
    base_prob = torch.ones(N, device=x_train.device, dtype=x_train.dtype) / N

    # 固定权重
    sample_prob = 0.65 * base_prob + 0.20 * coverage_prob + 0.15 * align_prob
    sample_prob = sample_prob / sample_prob.sum()
    idx = torch.multinomial(sample_prob, K, replacement=False)
    return idx


def select_v14_reg(x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, K: int, selection_ratio: float) -> torch.Tensor:
    """
    回归专用的数据蒸馏采样 (V14)。

    与 V9 类似，但根据 selection_ratio 调整权重：
    - selection_ratio <= 0.03: 更依赖覆盖度 (0.22)
    - selection_ratio > 0.03: 标准权重 (0.20)

    Args:
        x_train: 训练特征 [N, F]
        y_train: 训练标签 [N] (当前未使用，保留接口)
        x_test: 测试特征 [M, F]
        K: 要选择的样本数
        selection_ratio: K/N，选择比例

    Returns:
        idx: 选中的训练样本索引 [K]
    """
    N, F = x_train.shape
    K = min(K, N)

    query_mean = x_test.mean(dim=0, keepdim=True)
    X_qry_centered = x_test - query_mean
    n_comp = min(10, min(x_test.shape[0] - 1, F - 1))
    n_comp = max(n_comp, 1)

    try:
        U, S, V = torch.svd_lowrank(X_qry_centered, q=n_comp)
    except Exception:
        return select_v9_boosted_qsubspace(x_train, x_test, K)

    X_train_centered = x_train - query_mean
    X_proj = X_train_centered @ V
    coverage_score = X_proj.norm(dim=-1)
    coverage_prob = coverage_score / (coverage_score.sum() + 1e-8)

    q_align = compute_query_alignment(x_train, x_test)
    align_prob = q_align / (q_align.sum() + 1e-8)
    base_prob = torch.ones(N, device=x_train.device, dtype=x_train.dtype) / N

    if selection_ratio <= 0.03:
        sample_prob = 0.63 * base_prob + 0.22 * coverage_prob + 0.15 * align_prob
    else:
        sample_prob = 0.65 * base_prob + 0.20 * coverage_prob + 0.15 * align_prob

    sample_prob = sample_prob / sample_prob.sum()
    idx = torch.multinomial(sample_prob, K, replacement=False)
    return idx


def distill_training_data(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    max_samples: int,
    method: str = 'v14_reg'
) -> tuple:
    """
    数据蒸馏的高层接口。

    根据测试集特征，从训练集中选择最有代表性的样本。

    Args:
        X_train: 训练特征 [N, F]
        y_train: 训练标签 [N]
        X_test: 测试特征 [M, F]
        max_samples: 最大采样数
        method: 'v14_reg' 或 'v9_boosted'

    Returns:
        X_train_distilled: 蒸馏后的训练特征 [K, F]
        y_train_distilled: 蒸馏后的训练标签 [K]
        selected_idx: 选中的索引 [K]
    """
    N = X_train.shape[0]
    K = min(max_samples, N)

    if K >= N:
        # 不需要蒸馏
        return X_train, y_train, torch.arange(N, device=X_train.device)

    selection_ratio = K / N

    if method == 'v14_reg':
        idx = select_v14_reg(X_train, y_train, X_test, K, selection_ratio)
    elif method == 'v9_boosted':
        idx = select_v9_boosted_qsubspace(X_train, X_test, K)
    else:
        idx = select_v14_reg(X_train, y_train, X_test, K, selection_ratio)

    return X_train[idx], y_train[idx], idx
