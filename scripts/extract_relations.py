"""
FEAT Feature Relation Matrix Extraction Script
================================================

Extract a feature-relation matrix from FEAT's final feature-attention layer.
The extra row and column in the output correspond to the target.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feat.utils.loading import load_model


DEFAULT_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "FEAT-38M.pt"


def prepare_data(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_test: Optional[np.ndarray] = None,
    task_type: str = 'cls'
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    准备输入数据，进行标准化和格式转换。
    
    Args:
        X_train: 训练特征 numpy array [n_train, n_features]
        y_train: 训练标签 numpy array [n_train]
        X_test: 测试特征 numpy array [n_test, n_features]，如果为 None 则只用训练数据
        task_type: 'cls' 或 'reg'
    
    Returns:
        x_tensor: [1, seq_len, n_features]
        y_tensor: [1, seq_len]
        eval_pos: 训练数据的分割点
    """
    # 标准化 X
    mean_X = np.nanmean(X_train, axis=0)
    std_X = np.nanstd(X_train, axis=0) + 1e-6
    mean_X = np.nan_to_num(mean_X, nan=0.0)
    std_X = np.nan_to_num(std_X, nan=1.0)
    
    X_train_norm = np.nan_to_num((X_train - mean_X) / std_X, nan=0.0)
    
    if X_test is not None:
        X_test_norm = np.nan_to_num((X_test - mean_X) / std_X, nan=0.0)
        X_all = np.concatenate([X_train_norm, X_test_norm], axis=0)
    else:
        X_all = X_train_norm
    
    # 处理标签
    if task_type == 'cls':
        y_train_arr = np.array(y_train)
        unique_classes = np.unique(y_train_arr)
        class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        y_train_processed = np.array([class_to_idx[c] for c in y_train_arr], dtype=np.float32)
    else:
        y_train_float = np.array(y_train).astype(np.float32)
        mean_y = np.nanmean(y_train_float)
        std_y = np.nanstd(y_train_float) + 1e-6
        y_train_processed = (y_train_float - mean_y) / std_y
    
    # 转为 Tensor，增加 batch 维度
    x_tensor = torch.from_numpy(X_all).float().unsqueeze(0)  # [1, seq_len, n_features]
    y_tensor = torch.from_numpy(y_train_processed).float().unsqueeze(0)  # [1, n_train]
    
    eval_pos = len(X_train)
    
    return x_tensor, y_tensor, eval_pos


def extract_feature_attention(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor, 
    eval_pos: int,
    task_type: str = 'cls',
    device: torch.device = None
) -> np.ndarray:
    """
    从模型中提取特征注意力矩阵。
    
    通过临时设置 features_per_group=1，让每个特征独立计算注意力，
    从而得到真正的 N×N 特征级注意力矩阵。
    
    Args:
        model: FEAT model
        x: 输入特征 [batch, seq_len, n_features]
        y: 标签 [batch, n_train]
        eval_pos: 分割点
        task_type: 'cls' 或 'reg'
        device: 计算设备
    
    Returns:
        attention_matrix: [n_features, n_features] 的注意力矩阵 (numpy)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    x = x.to(device)
    y = y.to(device)
    
    # =========================================================================
    # 关键修改：临时将 features_per_group 设为 1，让每个特征独立计算注意力
    # =========================================================================
    original_features_per_group = model.features_per_group
    model.features_per_group = 1
    
    # 同步修改相关的子模块参数（如 feature_decoder）
    if hasattr(model, 'feature_decoder'):
        # feature_decoder 的输出维度可能依赖 features_per_group
        # 但在提取注意力时不需要用到 feature_decoder，所以可以忽略
        pass
    
    print(f"Temporarily set features_per_group: {original_features_per_group} -> 1")
    
    # 使用混合精度推理
    amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    
    try:
        with torch.autocast(
            device_type=device.type,
            enabled=device.type == 'cuda',
            dtype=amp_dtype,
        ), torch.inference_mode():
            # 调用 forward 并请求返回 feature attention
            output = model(
                x=x, 
                y=y, 
                eval_pos=eval_pos, 
                task_type=task_type,
                calculate_feature_attention=True,
                calculate_sample_attention=False
            )
    finally:
        # 恢复原始参数
        model.features_per_group = original_features_per_group
        print(f"Restored features_per_group: 1 -> {original_features_per_group}")
    
    # output 是一个 tuple: (encoder_out, feature_attention, sample_attention)
    if isinstance(output, tuple) and len(output) >= 2:
        encoder_out, feature_attention, sample_attention = output
    else:
        raise ValueError("Model did not return feature attention. Check calculate_feature_attention parameter.")
    
    if feature_attention is None:
        raise ValueError("Feature attention is None. The model might not support attention extraction.")
    
    # feature_attention 形状可能是 [Batch*Seq, n_features, n_features] 或类似
    print(f"Raw feature_attention shape: {feature_attention.shape}")
    
    # 转为 numpy 并处理
    attn_np = feature_attention.float().cpu().numpy()
    
    # 如果有多个维度，取平均或选择特定切片
    if attn_np.ndim == 3:
        # 通常是 [Batch*Seq, n_features, n_features]，对第一个维度取平均
        attn_matrix = attn_np.mean(axis=0)
    elif attn_np.ndim == 2:
        attn_matrix = attn_np
    else:
        # 尝试重塑
        n_groups = int(np.sqrt(attn_np.size))
        attn_matrix = attn_np.reshape(n_groups, n_groups)
    
    # 由于使用 features_per_group=1，现在直接得到 N×N 矩阵，不需要扩展
    print(f"Final attention matrix shape: {attn_matrix.shape}")
    
    return attn_matrix


def expand_group_matrix_to_features(
    group_matrix: np.ndarray,
    n_features: int,
    features_per_group: int
) -> np.ndarray:
    """
    将组级注意力矩阵扩展为原始特征级 N×N 矩阵。
    
    原理：如果特征 i 和 j 分别属于组 g_i 和 g_j，
    则 feature_matrix[i, j] = group_matrix[g_i, g_j]
    
    Args:
        group_matrix: [n_groups, n_groups] 的组级注意力矩阵
        n_features: 原始特征数量 N
        features_per_group: 每组包含的特征数
    
    Returns:
        feature_matrix: [n_features, n_features] 的特征级矩阵
    """
    n_groups = group_matrix.shape[0]
    feature_matrix = np.zeros((n_features, n_features), dtype=group_matrix.dtype)
    
    for i in range(n_features):
        g_i = i // features_per_group  # 特征 i 所属的组
        for j in range(n_features):
            g_j = j // features_per_group  # 特征 j 所属的组
            
            # 确保不越界（处理最后一组可能不满的情况）
            if g_i < n_groups and g_j < n_groups:
                feature_matrix[i, j] = group_matrix[g_i, g_j]
    
    return feature_matrix


def plot_relation_matrix(
    attention_matrix: np.ndarray,
    feature_names: Optional[list] = None,
    title: str = "Feature Relation Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = None,
    show_values: bool = True,
    normalize: bool = True,
    paper_mode: bool = True,
    show: bool = False,
):
    """
    绘制关联性矩阵热力图，支持论文级高质量矢量图输出。

    Args:
        attention_matrix: [N, N] 的注意力矩阵（N 为特征数）
        feature_names: 特征名列表，如果提供会显示在坐标轴上
        title: 图标题（论文模式下可设为None不显示标题）
        save_path: 保存路径，支持 .pdf/.svg/.png/.eps
        figsize: 图像大小 (width, height) in inches
        show_values: 是否在单元格中显示数值
        normalize: 是否归一化
        paper_mode: 是否为论文模式（优化字体、尺寸）
    """
    n_dim = attention_matrix.shape[0]

    # =========================================================================
    # 论文模式配置 - VLDB双栏，单栏宽度约3.33英寸
    # =========================================================================
    if paper_mode:
        # 使用论文友好的字体设置
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
            'font.size': 9,  # 基础字体大小，适合论文
            'axes.labelsize': 9,
            'axes.titlesize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'pdf.fonttype': 42,  # TrueType fonts in PDF (可编辑)
            'ps.fonttype': 42,
            'text.usetex': False,  # 如果有LaTeX可设为True
        })

        # VLDB单栏宽度 3.33 inch，留一点边距
        col_width = 3.33
        # 根据特征数量动态调整高度，保持方形紧凑热力图
        if figsize is None:
            fig_width = col_width
            fig_height = col_width * 0.85  # 为了节约版面纵向高度，压扁一点
            figsize = (fig_width, fig_height)

        # 根据矩阵大小调整字体
        if n_dim <= 6:
            cell_fontsize = 8
            label_fontsize = 8
        elif n_dim <= 10:
            cell_fontsize = 7
            label_fontsize = 7
        else:
            cell_fontsize = 6
            label_fontsize = 6
    else:
        if figsize is None:
            figsize = (8, 7)
        cell_fontsize = max(6, 10 - n_dim // 3)
        label_fontsize = max(7, 10 - n_dim // 4)

    # 处理矩阵
    attn_display = attention_matrix.copy()
    np.fill_diagonal(attn_display, 0)  # 移除自注意力（对角线）

    if normalize and attn_display.max() > 0:
        attn_display = attn_display / attn_display.max()

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 使用更适合论文的配色方案
    cmap = 'YlOrRd'  # 黄-橙-红，打印友好
    # cmap = 'Blues'  # 备选：蓝色系
    # cmap = 'viridis'  # 备选：原配色

    # 绘制热力图
    im = ax.imshow(attn_display, cmap=cmap, aspect='equal', interpolation='nearest',
                   vmin=0, vmax=1 if normalize else attn_display.max())

    # Colorbar设置 (紧凑化)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, shrink=0.75) # 稍微回调一点 pad 避免太拥挤
    cbar.ax.tick_params(labelsize=label_fontsize)
    if paper_mode:
        # 将 Score 放回外侧并旋转 270 度，既紧凑又绝对不会与顶部矩阵重合
        cbar.set_label('Score', fontsize=label_fontsize, rotation=270, labelpad=10)
    else:
        cbar.set_label('Score', fontsize=9)

    # 在每个单元格中标注数值
    if show_values:
        for i in range(n_dim):
            for j in range(n_dim):
                value = attn_display[i, j]
                # 根据背景色选择文字颜色（YlOrRd配色）
                text_color = 'black' if value < 0.6 else 'white'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       color=text_color, fontsize=cell_fontsize, fontweight='normal')

    # 设置坐标轴标签
    if feature_names is not None and len(feature_names) >= n_dim:
        labels = feature_names[:n_dim]
        # 截断过长的特征名
        max_label_len = 12 if paper_mode else 15
        labels = [name[:max_label_len] if len(name) > max_label_len else name for name in labels]
        ax.set_xticks(np.arange(n_dim))
        ax.set_yticks(np.arange(n_dim))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=label_fontsize)
        ax.set_yticklabels(labels, fontsize=label_fontsize)
    else:
        ax.set_xticks(np.arange(n_dim))
        ax.set_yticks(np.arange(n_dim))
        labels = [f'$F_{{{i+1}}}$' for i in range(n_dim)] if paper_mode else [f'F{i+1}' for i in range(n_dim)]
        ax.set_xticklabels(labels, fontsize=label_fontsize)
        ax.set_yticklabels(labels, fontsize=label_fontsize)

    # 添加细网格线
    ax.set_xticks(np.arange(-0.5, n_dim, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_dim, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 坐标轴标签 - 考虑到版面，删除 Target Feature 这种占位描述
    if paper_mode:
        # 删除占据很大空间的横向纵向轴名称，交由 caption 解释
        pass
    else:
        ax.set_xlabel('Target (j)', fontsize=11)
        ax.set_ylabel('Source (i)', fontsize=11)

    # 标题（论文中通常在caption里，可选择不显示）
    if title and not paper_mode:
        ax.set_title(title, fontsize=12, pad=10)

    # 调整布局，增加 pad 减少冗余留白
    plt.tight_layout(pad=0.2)

    # 保存文件
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        # 根据扩展名选择格式
        ext = os.path.splitext(save_path)[1].lower()

        save_kwargs = {
            'bbox_inches': 'tight',
            'pad_inches': 0.02,
        }

        if ext == '.pdf':
            save_kwargs['format'] = 'pdf'
            save_kwargs['backend'] = 'pdf'
        elif ext == '.svg':
            save_kwargs['format'] = 'svg'
        elif ext == '.eps':
            save_kwargs['format'] = 'eps'
        else:
            # PNG等栅格格式，使用高DPI
            save_kwargs['dpi'] = 600

        plt.savefig(save_path, **save_kwargs)
        print(f"Figure saved to: {save_path}")

        # 如果保存的是PDF，同时生成PNG预览
        if ext == '.pdf':
            png_path = save_path.replace('.pdf', '_preview.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
            print(f"PNG preview saved to: {png_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # 恢复默认设置
    if paper_mode:
        plt.rcParams.update(plt.rcParamsDefault)


def compute_correlation_matrix(
    model: torch.nn.Module,
    df: pd.DataFrame,
    target_col: str,
    task_type: str = 'cls',
    device: torch.device = None
) -> Dict:
    """
    高层 API：从 DataFrame 直接计算关联矩阵。
    
    Args:
        model: FEAT model
        df: 包含特征和标签的 DataFrame
        target_col: 目标列名
        task_type: 'cls' 或 'reg'
        device: 计算设备
    
    Returns:
        dict: 包含 attention_matrix, feature_names, n_groups 等信息
    """
    # 分离特征和标签
    feature_cols = [c for c in df.columns if c != target_col]
    y = df[target_col].values
    
    # 先对分类特征进行编码，再转换为 float
    X_encoded = np.zeros((len(df), len(feature_cols)), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            # 字符串/分类列：使用 factorize 编码
            codes, _ = pd.factorize(df[col].fillna('__MISSING__'))
            X_encoded[:, i] = codes.astype(np.float32)
        else:
            # 数值列：直接转换
            X_encoded[:, i] = pd.to_numeric(df[col], errors='coerce').values.astype(np.float32)
    
    # 准备数据
    x_tensor, y_tensor, eval_pos = prepare_data(X_encoded, y, task_type=task_type)
    
    n_features = len(feature_cols)
    
    # 提取注意力矩阵（临时使用 features_per_group=1 得到真正的 N×N 矩阵）
    attention_matrix = extract_feature_attention(
        model=model,
        x=x_tensor,
        y=y_tensor,
        eval_pos=eval_pos,
        task_type=task_type,
        device=device
    )
    
    # 获取模型的原始 features_per_group（仅作记录）
    features_per_group = getattr(model, 'features_per_group', 2)
    n_groups = (n_features + features_per_group - 1) // features_per_group
    
    # 注意：模型 forward 会将 y 作为最后一列拼接，所以 attention_matrix 维度是 (n_features+1) x (n_features+1)
    # 将 target_col 加入 feature_names 末尾以匹配矩阵维度
    all_feature_names = feature_cols + [target_col]
    
    return {
        'attention_matrix': attention_matrix,
        'feature_names': all_feature_names,
        'n_groups': n_groups,
        'features_per_group': features_per_group,
        'n_features': len(all_feature_names)  # 包含 target
    }


def main():
    """
    主函数：演示如何提取和可视化关联矩阵。
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract Feature Relation Matrix from FEAT')
    parser.add_argument('--model_path', type=str, default=str(DEFAULT_MODEL_PATH), help='Path to the FEAT-38M checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--target_col', type=str, required=True, help='Name of target column')
    parser.add_argument('--task_type', type=str, default='cls', choices=['cls', 'reg'], help='Task type')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the figure (.pdf/.svg/.png)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use: auto, cpu, cuda, or cuda:N')
    parser.add_argument('--paper_mode', action='store_true', default=True, help='Generate paper-quality figure (default: True)')
    parser.add_argument('--no_paper_mode', action='store_false', dest='paper_mode', help='Disable paper mode')
    parser.add_argument('--no_title', action='store_true', help='Do not show title (for paper figures)')
    parser.add_argument('--show_values', action='store_true', default=True, help='Show values in cells')
    parser.add_argument('--no_values', action='store_false', dest='show_values', help='Do not show values in cells')
    parser.add_argument('--show', action='store_true', help='Display the figure interactively')

    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    data_path = Path(args.data_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"FEAT checkpoint not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {data_path}")

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    if args.target_col not in df.columns:
        raise ValueError(f"Target column not found: {args.target_col}")
    if len(df.columns) < 2:
        raise ValueError("The input must contain at least one feature and one target column.")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print(f"Loading model from: {model_path}")
    model = load_model(model_path, mask_prediction=False)
    print(f"Model features_per_group: {model.features_per_group}")
    print(f"Model embed_dim: {model.embed_dim}")
    print(f"Model nlayers: {model.nlayers}")
    
    # 计算关联矩阵
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)
    result = compute_correlation_matrix(
        model=model,
        df=df,
        target_col=args.target_col,
        task_type=args.task_type,
        device=device
    )
    
    print(f"\n{'='*60}")
    print(f"Relation Matrix Shape: {result['attention_matrix'].shape}")
    print(f"Number of features: {result['n_features']}")
    print(f"Features: {result['feature_names']}")
    print(f"{'='*60}")
    
    # 终端输出矩阵
    print("\n[Attention Matrix (normalized, diagonal=0)]")
    attn = result['attention_matrix'].copy()
    np.fill_diagonal(attn, 0)
    if attn.max() > 0:
        attn = attn / attn.max()
    
    # 使用 pandas DataFrame 格式化输出
    matrix_df = pd.DataFrame(
        attn,
        index=result['feature_names'],
        columns=result['feature_names']
    )
    print(matrix_df.round(3).to_string())
    print()
    
    # 可视化
    fig_title = None if args.no_title else f"Feature Relation Matrix"
    plot_relation_matrix(
        attention_matrix=result['attention_matrix'],
        feature_names=result['feature_names'],
        title=fig_title,
        save_path=args.save_path,
        show_values=args.show_values,
        paper_mode=args.paper_mode,
        show=args.show,
    )
    
    # 保存原始矩阵数据
    if args.save_path:
        figure_path = Path(args.save_path)
        matrix_save_path = str(figure_path.with_suffix('.npy'))
        np.save(matrix_save_path, result['attention_matrix'])
        print(f"Matrix data saved to: {matrix_save_path}")


# =========================================================================
if __name__ == "__main__":
    main()
