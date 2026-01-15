#!/usr/bin/env python3
"""
マルチレイヤーネットワーク分析：統合スクリプト

このスクリプト1つで以下を実行：
1. フォルダ構造作成
2. ファイル整理
3. 記述統計
4. 可視化

使用方法：
  python run_analysis.py
"""

import os
import sys
import shutil
from pathlib import Path
import datetime

print("="*70)
print(" マルチレイヤーネットワーク分析：統合スクリプト ")
print("="*70)

# =====================================================================
# パート1: フォルダ構造作成
# =====================================================================
print("\n" + "="*70)
print(" パート1: フォルダ構造作成 ")
print("="*70)

FOLDER_STRUCTURE = {
    'data': {
        'raw': 'Raw data files',
        'processed': 'Processed data files',
        'quality_reports': 'Data quality reports'
    },
    'outputs': {
        'figures': 'Graphs and visualizations',
        'tables': 'Statistical tables (CSV)',
        'reports': 'Analysis reports (TXT/PDF)'
    },
    'scripts': 'Python scripts',
    'notebooks': 'Jupyter notebooks',
    'docs': 'Documentation'
}

print("\n[1-1] フォルダ作成中...")
created_folders = []

for main_folder, content in FOLDER_STRUCTURE.items():
    if isinstance(content, dict):
        for sub_folder, description in content.items():
            folder_path = Path(main_folder) / sub_folder
            if not folder_path.exists():
                folder_path.mkdir(parents=True, exist_ok=True)
                created_folders.append(str(folder_path))
                print(f"  ✓ 作成: {folder_path}")
            else:
                print(f"  - 既存: {folder_path}")
    else:
        folder_path = Path(main_folder)
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
            created_folders.append(str(folder_path))
            print(f"  ✓ 作成: {folder_path}")
        else:
            print(f"  - 既存: {folder_path}")

# =====================================================================
# パート2: データファイル確認
# =====================================================================
print("\n" + "="*70)
print(" パート2: データファイル確認 ")
print("="*70)

print("\n[2-1] multilayer_network.csv を探しています...")

# 探索パス
search_paths = [
    Path('multilayer_network.csv'),
    Path('data/multilayer_network.csv'),
    Path('data/raw/multilayer_network.csv'),
]

data_file = None
for path in search_paths:
    if path.exists():
        data_file = path
        print(f"  ✓ 発見: {path}")
        break

if data_file is None:
    print("  ✗ エラー: multilayer_network.csv が見つかりません")
    print("\n  以下のいずれかの場所にファイルを配置してください:")
    for path in search_paths:
        print(f"    - {path}")
    print("\n  または、カレントディレクトリに配置してください。")
    sys.exit(1)

# data/raw/ にコピー
target_path = Path('data/raw/multilayer_network.csv')
if data_file != target_path:
    print(f"\n[2-2] データファイルをコピー中...")
    shutil.copy2(data_file, target_path)
    print(f"  ✓ コピー: {data_file} → {target_path}")
    data_file = target_path

# =====================================================================
# パート3: README作成
# =====================================================================
print("\n" + "="*70)
print(" パート3: README作成 ")
print("="*70)

readme_content = f"""# Multilayer Network Analysis Project

## 📁 Project Structure

```
graduation_report/
├── data/
│   ├── raw/                    # 元データ
│   ├── processed/              # 処理済みデータ
│   └── quality_reports/        # データ品質レポート
│
├── outputs/
│   ├── figures/                # グラフ・可視化
│   ├── tables/                 # 統計表（CSV）
│   └── reports/                # 分析レポート
│
├── scripts/                    # Pythonスクリプト
├── notebooks/                  # Jupyter notebooks
└── docs/                       # ドキュメント
```

## 🚀 Quick Start

### すべてを一度に実行
```bash
python run_analysis.py
```

## 📊 Output Files

### Figures (`outputs/figures/`)
- `fig1_layer_distributions.png` - レイヤー別分布
- `fig2_yearly_coverage.png` - 年別カバレッジ
- `fig3_time_series.png` - 時系列トレンド
- `fig4_correlation_heatmaps.png` - 相関ヒートマップ
- `fig5_scatterplot_matrix_YYYY.png` - 散布図行列
- `fig6_boxplots.png` - Box Plot

### Tables (`outputs/tables/`)
- `layer_statistics.csv` - レイヤー別統計
- `yearly_statistics.csv` - 年別統計
- `correlation_analysis.csv` - 相関分析

### Reports (`outputs/reports/`)
- `descriptive_analysis_report.txt` - 記述統計レポート

## 📚 Data Sources

1. **Diplomatic Relations**: UN Diplomatic Network
2. **Aviation Routes**: International Flight Data
3. **Migration Stock**: UN Migration Database

## 📖 References

- Bonaccorsi et al. (2019). "Country centrality in the international multiplex network"
- Applied Network Science, 4:126

## 📝 Notes

Created: {datetime.datetime.now().strftime('%Y-%m-%d')}
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)
print("  ✓ README.md作成完了")

# .gitignore
gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/

# Jupyter
.ipynb_checkpoints/

# Data
data/raw/*.csv
!data/raw/.gitkeep

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
"""

with open('.gitignore', 'w', encoding='utf-8') as f:
    f.write(gitignore_content)
print("  ✓ .gitignore作成完了")

# =====================================================================
# パート4: 記述統計・可視化
# =====================================================================
print("\n" + "="*70)
print(" パート4: 記述統計・可視化 ")
print("="*70)

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    import warnings
    warnings.filterwarnings('ignore')
    
    print("\n  ✓ 必要なライブラリ読み込み成功")
    
except ImportError as e:
    print(f"\n  ✗ エラー: {e}")
    print("\n  以下のコマンドでライブラリをインストールしてください:")
    print("    pip install pandas numpy matplotlib seaborn scipy")
    sys.exit(1)

# グラフスタイル設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# データ読み込み
print("\n[4-1] データ読み込み中...")
df = pd.read_csv(data_file)

print(f"  総行数: {len(df):,}")
print(f"  列: {list(df.columns)}")
print(f"  年: {sorted(df['year'].unique())}")

layers = ['diplomatic_relation', 'aviation_routes', 'migrant_stock']

# =====================================================================
# 基本統計量
# =====================================================================
print("\n[4-2] 基本統計量計算中...")

stats_summary = []

for layer in layers:
    data = df[layer].dropna()
    
    stats_dict = {
        'レイヤー': layer,
        'データ数': len(data),
        '平均': data.mean(),
        '中央値': data.median(),
        '標準偏差': data.std(),
        '最小値': data.min(),
        '最大値': data.max(),
        '25%点': data.quantile(0.25),
        '75%点': data.quantile(0.75)
    }
    
    stats_summary.append(stats_dict)
    
    print(f"\n  ■ {layer}")
    print(f"    データ数: {len(data):,}")
    print(f"    平均: {data.mean():.2f}")
    print(f"    中央値: {data.median():.2f}")
    print(f"    範囲: [{data.min():.2f}, {data.max():.2f}]")

stats_df = pd.DataFrame(stats_summary)
stats_path = Path('outputs/tables/layer_statistics.csv')
stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
print(f"\n  ✓ 保存: {stats_path}")

# =====================================================================
# 年別統計
# =====================================================================
print("\n[4-3] 年別統計計算中...")

yearly_stats = []

for year in sorted(df['year'].unique()):
    df_year = df[df['year'] == year]
    
    stats_dict = {
        '年': year,
        '総行数': len(df_year),
        '外交データ数': df_year['diplomatic_relation'].notna().sum(),
        '航空データ数': df_year['aviation_routes'].notna().sum(),
        '移民データ数': df_year['migrant_stock'].notna().sum(),
        '完全データ数': ((df_year['diplomatic_relation'].notna()) & 
                      (df_year['aviation_routes'].notna()) & 
                      (df_year['migrant_stock'].notna())).sum()
    }
    
    yearly_stats.append(stats_dict)
    
    print(f"\n  ■ {year}年")
    print(f"    総行数: {stats_dict['総行数']:,}")
    print(f"    完全データ: {stats_dict['完全データ数']:,} "
          f"({stats_dict['完全データ数']/stats_dict['総行数']*100:.1f}%)")

yearly_df = pd.DataFrame(yearly_stats)
yearly_path = Path('outputs/tables/yearly_statistics.csv')
yearly_df.to_csv(yearly_path, index=False, encoding='utf-8-sig')
print(f"\n  ✓ 保存: {yearly_path}")

# =====================================================================
# 相関分析
# =====================================================================
print("\n[4-4] 相関分析中...")

complete = df.dropna()
print(f"\n  完全データ: {len(complete):,}行")

correlation_results = []

for year in sorted(complete['year'].unique()):
    data = complete[complete['year'] == year]
    
    if len(data) < 3:
        continue
    
    corr_diplo_avia, p1 = stats.spearmanr(
        data['diplomatic_relation'], 
        data['aviation_routes']
    )
    
    corr_diplo_migr, p2 = stats.spearmanr(
        data['diplomatic_relation'], 
        data['migrant_stock']
    )
    
    corr_avia_migr, p3 = stats.spearmanr(
        data['aviation_routes'], 
        data['migrant_stock']
    )
    
    correlation_results.append({
        '年': year,
        '外交×航空': corr_diplo_avia,
        '外交×移民': corr_diplo_migr,
        '航空×移民': corr_avia_migr,
        'p値_外交×航空': p1,
        'p値_外交×移民': p2,
        'p値_航空×移民': p3
    })
    
    print(f"\n  ■ {year}年")
    print(f"    外交 × 航空: {corr_diplo_avia:.3f}")
    print(f"    外交 × 移民: {corr_diplo_migr:.3f}")
    print(f"    航空 × 移民: {corr_avia_migr:.3f}")

corr_df = pd.DataFrame(correlation_results)
corr_path = Path('outputs/tables/correlation_analysis.csv')
corr_df.to_csv(corr_path, index=False, encoding='utf-8-sig')
print(f"\n  ✓ 保存: {corr_path}")

# =====================================================================
# 可視化
# =====================================================================
print("\n[4-5] 可視化生成中...")

years = sorted(df['year'].unique())

# 図1: レイヤー別分布
print("  [図1] レイヤー別分布...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, layer in enumerate(layers):
    data = df[layer].dropna()
    
    axes[idx].hist(data, bins=50, alpha=0.7, edgecolor='black')
    axes[idx].set_xlabel(layer, fontsize=11)
    axes[idx].set_ylabel('Frequency', fontsize=11)
    axes[idx].set_title(f'Distribution: {layer}', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    
    axes[idx].axvline(data.mean(), color='red', linestyle='--', 
                     linewidth=2, label=f'Mean: {data.mean():.2f}')
    axes[idx].axvline(data.median(), color='blue', linestyle='--', 
                     linewidth=2, label=f'Median: {data.median():.2f}')
    axes[idx].legend(fontsize=9)

plt.tight_layout()
plt.savefig('outputs/figures/fig1_layer_distributions.png', dpi=300, bbox_inches='tight')
print("    ✓ outputs/figures/fig1_layer_distributions.png")
plt.close()

# 図2: 年別データカバレッジ
print("  [図2] 年別データカバレッジ...")
fig, ax = plt.subplots(figsize=(10, 6))

diplomatic_counts = [df[df['year']==y]['diplomatic_relation'].notna().sum() for y in years]
aviation_counts = [df[df['year']==y]['aviation_routes'].notna().sum() for y in years]
migration_counts = [df[df['year']==y]['migrant_stock'].notna().sum() for y in years]

x = np.arange(len(years))
width = 0.25

ax.bar(x - width, diplomatic_counts, width, label='Diplomatic', alpha=0.8, edgecolor='black')
ax.bar(x, aviation_counts, width, label='Aviation', alpha=0.8, edgecolor='black')
ax.bar(x + width, migration_counts, width, label='Migration', alpha=0.8, edgecolor='black')

ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Records', fontsize=12, fontweight='bold')
ax.set_title('Data Coverage by Layer and Year', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/figures/fig2_yearly_coverage.png', dpi=300, bbox_inches='tight')
print("    ✓ outputs/figures/fig2_yearly_coverage.png")
plt.close()

# 図3: 時系列トレンド
print("  [図3] 時系列トレンド...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, layer in enumerate(layers):
    yearly_means = []
    for year in years:
        data = df[df['year'] == year][layer].dropna()
        yearly_means.append(data.mean())
    
    axes[idx].plot(years, yearly_means, marker='o', linewidth=2, 
                   markersize=8, color='steelblue')
    axes[idx].set_xlabel('Year', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Mean Value', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'Time Series: {layer}', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xticks(years)

plt.tight_layout()
plt.savefig('outputs/figures/fig3_time_series.png', dpi=300, bbox_inches='tight')
print("    ✓ outputs/figures/fig3_time_series.png")
plt.close()

# 図4: 相関ヒートマップ
print("  [図4] 相関ヒートマップ...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, year in enumerate(years):
    data = complete[complete['year'] == year][layers]
    
    if len(data) < 3:
        axes[idx].text(0.5, 0.5, 'Insufficient Data', 
                      ha='center', va='center', fontsize=14)
        axes[idx].set_title(f'{year}', fontsize=12, fontweight='bold')
        continue
    
    corr_matrix = data.corr(method='spearman')
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, square=True, 
                cbar_kws={'shrink': 0.8}, ax=axes[idx],
                xticklabels=['Diplo', 'Avia', 'Migr'],
                yticklabels=['Diplo', 'Avia', 'Migr'])
    axes[idx].set_title(f'{year}', fontsize=12, fontweight='bold')

for idx in range(len(years), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('outputs/figures/fig4_correlation_heatmaps.png', dpi=300, bbox_inches='tight')
print("    ✓ outputs/figures/fig4_correlation_heatmaps.png")
plt.close()

# 図5: Box Plot
print("  [図5] Box Plot...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, layer in enumerate(layers):
    data_list = []
    labels = []
    
    for year in years:
        data = df[df['year'] == year][layer].dropna()
        if len(data) > 0:
            data_list.append(data)
            labels.append(str(year))
    
    bp = axes[idx].boxplot(data_list, labels=labels, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2),
                          whiskerprops=dict(linewidth=1.5),
                          capprops=dict(linewidth=1.5))
    
    axes[idx].set_xlabel('Year', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Value', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'Box Plot: {layer}', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/figures/fig6_boxplots.png', dpi=300, bbox_inches='tight')
print("    ✓ outputs/figures/fig6_boxplots.png")
plt.close()

# =====================================================================
# サマリーレポート
# =====================================================================
print("\n[4-6] サマリーレポート生成中...")

report_path = Path('outputs/reports/descriptive_analysis_report.txt')

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write(" MULTILAYER NETWORK: DESCRIPTIVE ANALYSIS REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write("1. DATASET OVERVIEW\n")
    f.write("-"*70 + "\n")
    f.write(f"Total Records: {len(df):,}\n")
    f.write(f"Years: {sorted(df['year'].unique())}\n")
    f.write(f"Layers: {len(layers)}\n\n")
    
    f.write("2. LAYER STATISTICS\n")
    f.write("-"*70 + "\n")
    f.write(stats_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("3. YEARLY COVERAGE\n")
    f.write("-"*70 + "\n")
    f.write(yearly_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("4. CORRELATION ANALYSIS\n")
    f.write("-"*70 + "\n")
    f.write(corr_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("="*70 + "\n")

print(f"  ✓ {report_path}")

# =====================================================================
# 完了
# =====================================================================
print("\n" + "="*70)
print(" ✓ すべての処理が完了しました！ ")
print("="*70)

print("\n【生成されたファイル】")
print("\n📁 フォルダ構造:")
print("   ├── data/raw/")
print("   ├── outputs/figures/     ← 📊 グラフ6枚")
print("   ├── outputs/tables/      ← 📋 統計表3枚")
print("   └── outputs/reports/     ← 📄 レポート1枚")

print("\n📊 統計表 (outputs/tables/):")
print("   - layer_statistics.csv")
print("   - yearly_statistics.csv")
print("   - correlation_analysis.csv")

print("\n📈 グラフ (outputs/figures/):")
print("   - fig1_layer_distributions.png")
print("   - fig2_yearly_coverage.png")
print("   - fig3_time_series.png")
print("   - fig4_correlation_heatmaps.png")
print("   - fig6_boxplots.png")

print("\n📄 レポート (outputs/reports/):")
print("   - descriptive_analysis_report.txt")

print("\n次のステップ:")
print("  1. outputs/figures/ のグラフを確認")
print("  2. outputs/reports/descriptive_analysis_report.txt を読む")
print("  3. ネットワーク分析に進む")

print("\n" + "="*70)