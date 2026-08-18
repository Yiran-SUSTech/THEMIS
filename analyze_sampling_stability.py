"""
简化版：分析不同每类采样数量下3个指标的平均值和ICC稳定性
- 折线图：各指标平均值随采样密度的变化
- ICC图：系统3次运行间的ICC(2,1)随采样密度的变化
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr


def compute_icc_2_1_k(data_matrix):
    """
    计算k个评分者的ICC(2,1): Two-way random, absolute agreement, single measures.
    data_matrix: shape (n_subjects, k_raters)
    公式与 extract_and_analyze_scores.py 中的 compute_icc_k(..., "2,1") 一致:
      ICC(2,1) = (MSB - MSE) / (MSB + (k-1)*MSE + k*(MSR - MSE)/n)
    其中 MSB=between subjects, MSR=between raters, MSE=error.
    """
    data_matrix = np.asarray(data_matrix, dtype=float)
    n, k = data_matrix.shape
    grand_mean = np.mean(data_matrix)
    subject_means = np.mean(data_matrix, axis=1)
    rater_means = np.mean(data_matrix, axis=0)

    ss_subjects = k * np.sum((subject_means - grand_mean) ** 2)
    ss_raters = n * np.sum((rater_means - grand_mean) ** 2)
    ss_error = np.sum(
        (data_matrix
         - subject_means[:, np.newaxis]
         - rater_means[np.newaxis, :]
         + grand_mean) ** 2
    )

    msb = ss_subjects / (n - 1)
    msr = ss_raters / (k - 1)
    mse = ss_error / ((n - 1) * (k - 1))

    denom = msb + (k - 1) * mse + k * (msr - mse) / n
    if denom <= 0:
        return np.nan
    return (msb - mse) / denom

# 数据配置
configs = [
    (200, 2, "analysis_output_Sys_VAR_ref_200_Human_VAR_200"),
    (400, 4, "analysis_output_Sys_VAR_ref_400_Human_VAR_400"),
    (500, 5, "analysis_output_Sys_VAR_ref_500_Human_VAR_500"),
    (600, 6, "analysis_output_Sys_VAR_ref_600_Human_VAR_600"),
    (800, 8, "analysis_output_Sys_VAR_ref_800_Human_VAR_800"),
    (1000, 10, "analysis_output_Sys_VAR_ref_1000_Human_VAR_1000"),
]

metrics = ['alignment_norm', 'authenticity_norm', 'composite_product']

# 提取数据
results = []
for total_imgs, per_class, folder in configs:
    csv_path = Path(folder) / "cross_class_macro_avg.csv"
    df_macro = pd.read_csv(csv_path)
    
    # 提取3次系统运行的数据
    sys_rows = df_macro[df_macro['source'].str.startswith('Sys_VAR_ref_') & ~df_macro['source'].str.contains('AVG')]
    # 提取3个人类标注员的数据
    user_rows = df_macro[df_macro['source'].str.startswith('User_VAR_') & ~df_macro['source'].str.contains('AVG')]
    
    row_data = {'total_images': total_imgs, 'per_class': per_class}
    
    for metric in metrics:
        # 系统数据 (mean/std 仍从 cross_class_macro_avg.csv 读取)
        sys_values = sys_rows[metric].values
        row_data[f'{metric}_sys_mean'] = np.mean(sys_values)
        row_data[f'{metric}_sys_std'] = np.std(sys_values, ddof=1)

        # 人类数据
        user_values = user_rows[metric].values
        row_data[f'{metric}_user_mean'] = np.mean(user_values)
        row_data[f'{metric}_user_std'] = np.std(user_values, ddof=1)

        # 计算ICC(2,1) - 使用 per-image 数据 (与 stability.csv / correlations.csv 一致)
        per_image_path = Path(folder) / "per_image_normalized_scores.csv"
        df_per_image = pd.read_csv(per_image_path)

        # per_image_normalized_scores.csv 中 composite_product 的列名不同
        img_col = 'composite_product_perimage' if metric == 'composite_product' else metric

        # 系统 ICC: 3 次 run 的 per-image 分数组成 (n_images, 3) 矩阵
        sys_sources = sorted(
            [s for s in df_per_image['source'].unique()
             if s.startswith('Sys_VAR_ref_') and 'AVG' not in s]
        )
        pivot_sys = df_per_image[df_per_image['source'].isin(sys_sources)] \
            .pivot_table(index='image_id', columns='source', values=img_col)
        pivot_sys = pivot_sys.dropna()  # 只保留3次run都有的image
        row_data[f'{metric}_sys_icc'] = compute_icc_2_1_k(pivot_sys.values)

        # 人类 ICC: 3 个标注员的 per-image 分数组成 (n_images, 3) 矩阵
        user_sources = sorted(
            [s for s in df_per_image['source'].unique()
             if s.startswith('User_VAR_') and 'AVG' not in s]
        )
        pivot_user = df_per_image[df_per_image['source'].isin(user_sources)] \
            .pivot_table(index='image_id', columns='source', values=img_col)
        pivot_user = pivot_user.dropna()
        row_data[f'{metric}_user_icc'] = compute_icc_2_1_k(pivot_user.values)
    
    results.append(row_data)

df_results = pd.DataFrame(results)

# 打印汇总表
print("=" * 100)
print("Sampling Stability Analysis (Simplified)")
print("=" * 100)
print(f"{'Images':>8} {'Per Class':>10} | {'Align Sys':>10} {'Align Sys ICC':>14} {'Align User':>11} {'Align User ICC':>15} | {'Auth Sys':>10} {'Auth Sys ICC':>13} {'Auth User':>10} {'Auth User ICC':>14} | {'Comp Sys':>10} {'Comp Sys ICC':>13} {'Comp User':>10} {'Comp User ICC':>14}")
for _, row in df_results.iterrows():
    print(f"{row['total_images']:>8.0f} {row['per_class']:>10.0f} | "
          f"{row['alignment_norm_sys_mean']:>10.4f} {row['alignment_norm_sys_icc']:>14.4f} {row['alignment_norm_user_mean']:>11.4f} {row['alignment_norm_user_icc']:>15.4f} | "
          f"{row['authenticity_norm_sys_mean']:>10.4f} {row['authenticity_norm_sys_icc']:>13.4f} {row['authenticity_norm_user_mean']:>10.4f} {row['authenticity_norm_user_icc']:>14.4f} | "
          f"{row['composite_product_sys_mean']:>10.4f} {row['composite_product_sys_icc']:>13.4f} {row['composite_product_user_mean']:>10.4f} {row['composite_product_user_icc']:>14.4f}")

# 保存汇总数据
df_results.to_csv("sampling_stability_simplified.csv", index=False)
print("\n\nData saved to sampling_stability_simplified.csv")

# 生成可视化图表
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 第一行：指标平均值折线图
for i, metric in enumerate(metrics):
    ax = axes[0, i]
    metric_name = metric.replace('_', ' ').title()
    
    ax.plot(df_results['per_class'], df_results[f'{metric}_sys_mean'], 'o-', linewidth=2.5, markersize=10, color='blue', label='System Mean')
    ax.fill_between(
        df_results['per_class'],
        df_results[f'{metric}_sys_mean'] - df_results[f'{metric}_sys_std'],
        df_results[f'{metric}_sys_mean'] + df_results[f'{metric}_sys_std'],
        alpha=0.2, color='blue', label='±1 Std'
    )
    
    ax.set_xlabel('Images per class', fontsize=13)
    ax.set_ylabel(metric_name, fontsize=13)
    ax.set_title(f'({chr(97+i)}) {metric_name} - Mean Score', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df_results['per_class'])
    
    # 添加数值标签
    for j, (x, y) in enumerate(zip(df_results['per_class'], df_results[f'{metric}_sys_mean'])):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

# 第二行：ICC折线图（系统和人类对比）
for i, metric in enumerate(metrics):
    ax = axes[1, i]
    metric_name = metric.replace('_', ' ').title()
    
    ax.plot(df_results['per_class'], df_results[f'{metric}_sys_icc'], 'o-', linewidth=2.5, markersize=10, color='blue', label='System ICC')
    ax.plot(df_results['per_class'], df_results[f'{metric}_user_icc'], 's-', linewidth=2.5, markersize=10, color='green', label='Human ICC')
    ax.axhline(y=0.75, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Good (0.75)')
    ax.axhline(y=0.90, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Excellent (0.90)')
    
    ax.set_xlabel('Images per class', fontsize=13)
    ax.set_ylabel('ICC(2,1)', fontsize=13)
    ax.set_title(f'({chr(100+i)}) {metric_name} - ICC Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df_results['per_class'])
    ax.set_ylim(0.5, 1.0)
    
    # 添加数值标签
    for j, (x, y) in enumerate(zip(df_results['per_class'], df_results[f'{metric}_sys_icc'])):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    for j, (x, y) in enumerate(zip(df_results['per_class'], df_results[f'{metric}_user_icc'])):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("sampling_stability_simplified.png", dpi=300, bbox_inches='tight')
plt.savefig("sampling_stability_simplified.pdf", bbox_inches='tight')
print("\nCharts saved to sampling_stability_simplified.png/pdf")

# 稳定性分析
print("\n" + "=" * 80)
print("Stability Analysis")
print("=" * 80)

print("\nSystem ICC when n>=5:")
for metric in metrics:
    metric_df = df_results[df_results['per_class'] >= 5]
    avg_icc = metric_df[f'{metric}_sys_icc'].mean()
    min_icc = metric_df[f'{metric}_sys_icc'].min()
    print(f"  {metric}: Avg ICC = {avg_icc:.4f}, Min ICC = {min_icc:.4f}")

print("\nHuman ICC when n>=5:")
for metric in metrics:
    metric_df = df_results[df_results['per_class'] >= 5]
    avg_icc = metric_df[f'{metric}_user_icc'].mean()
    min_icc = metric_df[f'{metric}_user_icc'].min()
    print(f"  {metric}: Avg ICC = {avg_icc:.4f}, Min ICC = {min_icc:.4f}")

print("\nMean score stability when n>=5:")
for metric in metrics:
    metric_df = df_results[df_results['per_class'] >= 5]
    avg_std = metric_df[f'{metric}_sys_std'].mean()
    max_std = metric_df[f'{metric}_sys_std'].max()
    print(f"  {metric}: Avg Std = {avg_std:.4f}, Max Std = {max_std:.4f}")
