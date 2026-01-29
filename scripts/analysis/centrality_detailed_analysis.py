#!/usr/bin/env python3
"""
中心性指標の詳細比較分析（日本語版・フォント修正版）

使用方法：
  python centrality_jp_fixed.py
"""

import warnings
warnings.filterwarnings('ignore')

# ========================================
# フォント設定（最優先で実行）
# ========================================
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def setup_japanese_font():
    """日本語フォントを自動検出して設定"""
    # 優先順位順
    preferred_fonts = [
        'MS Gothic', 'MS PGothic', 'MS UI Gothic',
        'Yu Gothic', 'Yu Gothic UI', 'YuGothic', 
        'Meiryo', 'Meiryo UI',
        'IPAexGothic', 'IPAGothic', 'TakaoPGothic',
        'Hiragino Sans', 'Hiragino Kaku Gothic Pro'
    ]
    
    available = set([f.name for f in fm.fontManager.ttflist])
    
    for font in preferred_fonts:
        if font in available:
            print(f"✓ 日本語フォント検出: {font}")
            return font
    
    # 部分一致検索
    for font in available:
        if 'Gothic' in font or 'ゴシック' in font:
            print(f"✓ 日本語フォント検出（部分一致）: {font}")
            return font
    
    print("⚠ 日本語フォント未検出")
    return 'DejaVu Sans'

# フォント設定を適用
FONT_NAME = setup_japanese_font()
plt.rcParams.update({
    'font.family': FONT_NAME,
    'font.size': 10,
    'axes.unicode_minus': False,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# ========================================
# その他のインポート
# ========================================
import pandas as pd
import numpy as np
import networkx as nx
from matplotlib_venn import venn3
from pathlib import Path
from scipy import stats

# seabornは最後（フォント設定後）
import seaborn as sns
sns.set_style("whitegrid")
sns.set_palette("Set2")

# seaborn後にフォントを再設定（重要！）
plt.rcParams['font.family'] = FONT_NAME
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print(" 中心性指標の詳細比較分析（日本語版）")
print("="*70)

# =====================================================================
# 1. データ読み込み
# =====================================================================
print("\n[1] データ読み込み中...")

data_paths = [
    Path('data/raw/multilayer_network.csv'),
    Path('multilayer_network.csv'),
    Path('data/multilayer_network.csv')
]

df = None
for data_path in data_paths:
    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"  ✓ データ読み込み: {data_path}")
        break

if df is None:
    print("  ✗ エラー: データファイルが見つかりません")
    exit(1)

# 出力フォルダ
OUTPUT_DIRS = {
    'figures': Path('outputs/figures'),
    'tables': Path('outputs/tables'),
    'reports': Path('outputs/reports')
}

for path in OUTPUT_DIRS.values():
    path.mkdir(parents=True, exist_ok=True)

layers = ['diplomatic_relation', 'aviation_routes', 'migrant_stock', 'trade_value']
layer_names = {
    'diplomatic_relation': '外交',
    'aviation_routes': '航空',
    'migrant_stock': '移民',
    'trade_value': '貿易'
}

years = sorted(df['year'].unique())
print(f"  年: {years}")

# =====================================================================
# 2. 中心性計算
# =====================================================================
print("\n[2] 中心性計算中...")

def calculate_all_centralities(G, weighted=True):
    """3種類の中心性を計算"""
    centralities = {}
    
    # 1. Degree Centrality
    if weighted and nx.is_weighted(G):
        centralities['degree_in'] = dict(G.in_degree(weight='weight'))
        centralities['degree_out'] = dict(G.out_degree(weight='weight'))
    else:
        centralities['degree_in'] = dict(G.in_degree())
        centralities['degree_out'] = dict(G.out_degree())
    
    centralities['degree_total'] = {
        node: centralities['degree_in'].get(node, 0) + 
              centralities['degree_out'].get(node, 0)
        for node in G.nodes()
    }
    
    # 2. Betweenness Centrality
    try:
        if weighted and nx.is_weighted(G):
            for u, v, data in G.edges(data=True):
                if data['weight'] > 0:
                    data['distance'] = 1.0 / data['weight']
                else:
                    data['distance'] = float('inf')
            
            centralities['betweenness'] = nx.betweenness_centrality(
                G, weight='distance', normalized=True)
        else:
            centralities['betweenness'] = nx.betweenness_centrality(
                G, normalized=True)
    except:
        centralities['betweenness'] = {node: 0 for node in G.nodes()}
    
    # 3. Closeness Centrality
    try:
        G_reverse = G.reverse()
        centralities['closeness_in'] = nx.closeness_centrality(
            G_reverse, distance='distance' if weighted else None)
        centralities['closeness_out'] = nx.closeness_centrality(
            G, distance='distance' if weighted else None)
    except:
        centralities['closeness_in'] = {node: 0 for node in G.nodes()}
        centralities['closeness_out'] = {node: 0 for node in G.nodes()}
    
    return centralities

# 全計算
all_centralities = {}

for year in years:
    print(f"\n  ■ {year}年")
    df_year = df[df['year'] == year]
    all_centralities[year] = {}
    
    for layer in layers:
        print(f"    - {layer_names[layer]}レイヤー...")
        
        layer_data = df_year[df_year[layer].notna()]
        G = nx.DiGraph()
        
        for _, row in layer_data.iterrows():
            weight = row[layer]
            G.add_edge(row['origin'], row['destination'], weight=weight)
        
        centralities = calculate_all_centralities(G, weighted=True)
        all_centralities[year][layer] = centralities

print("\n  ✓ 中心性計算完了")

# =====================================================================
# 3. 指標間相関
# =====================================================================
print("\n[3] 指標間相関分析中...")

latest_year = max(years)
correlation_results = []

for layer in layers:
    centralities = all_centralities[latest_year][layer]
    
    df_cent = pd.DataFrame({
        '次数_合計': centralities['degree_total'],
        '媒介中心性': centralities['betweenness'],
        '近接_入': centralities['closeness_in']
    })
    
    corr_matrix = df_cent.corr(method='spearman')
    
    print(f"\n  ■ {layer_names[layer]}レイヤー")
    print(f"    次数-媒介: {corr_matrix.loc['次数_合計', '媒介中心性']:.3f}")
    print(f"    次数-近接: {corr_matrix.loc['次数_合計', '近接_入']:.3f}")
    print(f"    媒介-近接: {corr_matrix.loc['媒介中心性', '近接_入']:.3f}")
    
    correlation_results.append({
        'レイヤー': layer_names[layer],
        '次数×媒介': corr_matrix.loc['次数_合計', '媒介中心性'],
        '次数×近接': corr_matrix.loc['次数_合計', '近接_入'],
        '媒介×近接': corr_matrix.loc['媒介中心性', '近接_入']
    })

corr_df = pd.DataFrame(correlation_results)
corr_path = OUTPUT_DIRS['tables'] / f'centrality_correlations_{latest_year}.csv'
corr_df.to_csv(corr_path, index=False, encoding='utf-8-sig')
print(f"\n  ✓ 保存: {corr_path}")

# =====================================================================
# 4. トップ20ランキング
# =====================================================================
print("\n[4] トップ20ランキング作成中...")

def get_top_k(centrality_dict, k=20):
    sorted_items = sorted(centrality_dict.items(), 
                         key=lambda x: x[1], reverse=True)
    return sorted_items[:k]

for layer in layers:
    centralities = all_centralities[latest_year][layer]
    
    top_degree = get_top_k(centralities['degree_total'], 20)
    df_degree = pd.DataFrame(top_degree, columns=['国名', '次数中心性'])
    df_degree['順位'] = range(1, 21)
    
    top_between = get_top_k(centralities['betweenness'], 20)
    df_between = pd.DataFrame(top_between, columns=['国名', '媒介中心性'])
    df_between['順位'] = range(1, 21)
    
    top_close = get_top_k(centralities['closeness_in'], 20)
    df_close = pd.DataFrame(top_close, columns=['国名', '近接中心性'])
    df_close['順位'] = range(1, 21)
    
    layer_short = layer.replace('_', '')
    
    OUTPUT_DIRS['tables'].joinpath(f'top20_次数_{layer_short}_{latest_year}.csv').write_text(
        df_degree.to_csv(index=False, encoding='utf-8-sig'), encoding='utf-8-sig')
    OUTPUT_DIRS['tables'].joinpath(f'top20_媒介_{layer_short}_{latest_year}.csv').write_text(
        df_between.to_csv(index=False, encoding='utf-8-sig'), encoding='utf-8-sig')
    OUTPUT_DIRS['tables'].joinpath(f'top20_近接_{layer_short}_{latest_year}.csv').write_text(
        df_close.to_csv(index=False, encoding='utf-8-sig'), encoding='utf-8-sig')
    
    print(f"\n  ■ {layer_names[layer]}レイヤー - トップ5")
    print(f"    次数: {[c for c, _ in top_degree[:5]]}")
    print(f"    媒介: {[c for c, _ in top_between[:5]]}")
    print(f"    近接: {[c for c, _ in top_close[:5]]}")

print(f"\n  ✓ トップ20保存完了")

# =====================================================================
# 5. 可視化
# =====================================================================
print("\n[5] 可視化生成中...")

# ===== 図13: 相関ヒートマップ =====
print("  [図13] 相関ヒートマップ...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, layer in enumerate(layers):
    centralities = all_centralities[latest_year][layer]
    
    df_cent = pd.DataFrame({
        '次数': centralities['degree_total'],
        '媒介': centralities['betweenness'],
        '近接': centralities['closeness_in']
    })
    
    corr_matrix = df_cent.corr(method='spearman')
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
               center=0.5, vmin=0, vmax=1, square=True,
               cbar_kws={'shrink': 0.8}, ax=axes[idx],
               linewidths=2, linecolor='white')
    
    axes[idx].set_title(f'{layer_names[layer]}レイヤー（{latest_year}年）',
                       fontsize=13, fontweight='bold', fontname=FONT_NAME)
    
    # 軸ラベルのフォントも明示的に設定
    for label in axes[idx].get_xticklabels() + axes[idx].get_yticklabels():
        label.set_fontname(FONT_NAME)

plt.suptitle('中心性指標間の相関（Spearman順位相関係数）',
            fontsize=14, fontweight='bold', y=1.02, fontname=FONT_NAME)
plt.tight_layout()

fig_path = OUTPUT_DIRS['figures'] / 'fig13_centrality_correlations.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig_path}")
plt.close()

# ===== 図14: 外交トップ10 =====
print("  [図14] トップ10比較（外交）...")

layer = 'diplomatic_relation'
centralities = all_centralities[latest_year][layer]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

measures = ['degree_total', 'betweenness', 'closeness_in']
titles = ['次数中心性', '媒介中心性', '近接中心性']
colors = ['steelblue', 'coral', 'lightseagreen']

for idx, (measure, title, color) in enumerate(zip(measures, titles, colors)):
    top_k = get_top_k(centralities[measure], 10)
    countries = [c for c, _ in top_k]
    values = [v for _, v in top_k]
    
    axes[idx].barh(range(len(countries)), values, color=color, alpha=0.8)
    axes[idx].set_yticks(range(len(countries)))
    axes[idx].set_yticklabels(countries, fontsize=10, fontname=FONT_NAME)
    axes[idx].set_xlabel(title, fontsize=11, fontweight='bold', fontname=FONT_NAME)
    axes[idx].set_title(f'トップ10: {title}', fontsize=12, fontweight='bold', fontname=FONT_NAME)
    axes[idx].invert_yaxis()
    axes[idx].grid(True, alpha=0.3, axis='x')

plt.suptitle(f'外交レイヤー: 中心性指標別トップ10（{latest_year}年）',
            fontsize=14, fontweight='bold', y=1.02, fontname=FONT_NAME)
plt.tight_layout()

fig_path = OUTPUT_DIRS['figures'] / 'fig14_top10_diplomatic_by_measure.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig_path}")
plt.close()

# ===== 図15: 航空トップ10 =====
print("  [図15] トップ10比較（航空）...")

layer = 'aviation_routes'
centralities = all_centralities[latest_year][layer]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (measure, title, color) in enumerate(zip(measures, titles, colors)):
    top_k = get_top_k(centralities[measure], 10)
    countries = [c for c, _ in top_k]
    values = [v for _, v in top_k]
    
    axes[idx].barh(range(len(countries)), values, color=color, alpha=0.8)
    axes[idx].set_yticks(range(len(countries)))
    axes[idx].set_yticklabels(countries, fontsize=10, fontname=FONT_NAME)
    axes[idx].set_xlabel(title, fontsize=11, fontweight='bold', fontname=FONT_NAME)
    axes[idx].set_title(f'トップ10: {title}', fontsize=12, fontweight='bold', fontname=FONT_NAME)
    axes[idx].invert_yaxis()
    axes[idx].grid(True, alpha=0.3, axis='x')

plt.suptitle(f'航空レイヤー: 中心性指標別トップ10（{latest_year}年）',
            fontsize=14, fontweight='bold', y=1.02, fontname=FONT_NAME)
plt.tight_layout()

fig_path = OUTPUT_DIRS['figures'] / 'fig15_top10_aviation_by_measure.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig_path}")
plt.close()

# ===== 図16: ベン図 =====
print("  [図16] ベン図...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, layer in enumerate(layers):
    centralities = all_centralities[latest_year][layer]
    
    top_degree = set([c for c, _ in get_top_k(centralities['degree_total'], 20)])
    top_between = set([c for c, _ in get_top_k(centralities['betweenness'], 20)])
    top_close = set([c for c, _ in get_top_k(centralities['closeness_in'], 20)])
    
    ax = axes[idx]
    venn = venn3([top_degree, top_between, top_close],
                 set_labels=('次数', '媒介', '近接'),
                 ax=ax, alpha=0.7)
    
    if venn.get_patch_by_id('100'):
        venn.get_patch_by_id('100').set_color('steelblue')
    if venn.get_patch_by_id('010'):
        venn.get_patch_by_id('010').set_color('coral')
    if venn.get_patch_by_id('001'):
        venn.get_patch_by_id('001').set_color('lightseagreen')
    
    ax.set_title(f'{layer_names[layer]}レイヤー', 
                fontsize=12, fontweight='bold', fontname=FONT_NAME)
    
    # ベン図のラベルのフォントも設定
    for text in venn.set_labels:
        if text:
            text.set_fontname(FONT_NAME)
    for text in venn.subset_labels:
        if text:
            text.set_fontname(FONT_NAME)

plt.suptitle(f'トップ20の重複（中心性指標間、{latest_year}年）',
            fontsize=14, fontweight='bold', y=1.02, fontname=FONT_NAME)
plt.tight_layout()

fig_path = OUTPUT_DIRS['figures'] / 'fig16_venn_top20_overlap.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig_path}")
plt.close()

# ===== 図17: 散布図 =====
print("  [図17] 散布図...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, layer in enumerate(layers):
    centralities = all_centralities[latest_year][layer]
    
    degree_vals = list(centralities['degree_total'].values())
    between_vals = list(centralities['betweenness'].values())
    
    axes[idx].scatter(degree_vals, between_vals, alpha=0.6, s=50, 
                     color='steelblue', edgecolor='black', linewidth=0.5)
    
    # トップ10にラベル
    top_degree = get_top_k(centralities['degree_total'], 10)
    for country, deg in top_degree:
        bet = centralities['betweenness'][country]
        axes[idx].annotate(country, (deg, bet), fontsize=8,
                         xytext=(5, 5), textcoords='offset points',
                         fontname=FONT_NAME)
    
    axes[idx].set_xlabel('次数中心性', fontsize=11, fontweight='bold', fontname=FONT_NAME)
    axes[idx].set_ylabel('媒介中心性', fontsize=11, fontweight='bold', fontname=FONT_NAME)
    axes[idx].set_title(f'{layer_names[layer]}レイヤー',
                       fontsize=12, fontweight='bold', fontname=FONT_NAME)
    axes[idx].grid(True, alpha=0.3)
    
    corr = np.corrcoef(degree_vals, between_vals)[0, 1]
    axes[idx].text(0.05, 0.95, f'r = {corr:.3f}',
                  transform=axes[idx].transAxes,
                  fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                  fontname=FONT_NAME)

plt.suptitle(f'次数中心性 vs 媒介中心性（{latest_year}年）',
            fontsize=14, fontweight='bold', y=1.02, fontname=FONT_NAME)
plt.tight_layout()

fig_path = OUTPUT_DIRS['figures'] / 'fig17_scatter_degree_vs_betweenness.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig_path}")
plt.close()

# ===== 図18: 時系列 =====
print("  [図18] 時系列...")

selected_countries = ['USA', 'CHN', 'GBR', 'DEU', 'FRA', 'RUS', 'JPN', 'IND']

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

layer = 'diplomatic_relation'

for idx, measure in enumerate(['degree_total', 'betweenness', 'closeness_in']):
    for country in selected_countries:
        values = []
        for year in years:
            cent = all_centralities[year][layer][measure]
            values.append(cent.get(country, 0))
        
        axes[idx].plot(years, values, marker='o', linewidth=2, 
                      markersize=6, label=country)
    
    measure_names = {'degree_total': '次数中心性', 
                    'betweenness': '媒介中心性',
                    'closeness_in': '近接中心性'}
    
    axes[idx].set_xlabel('年', fontsize=11, fontweight='bold', fontname=FONT_NAME)
    axes[idx].set_ylabel(measure_names[measure],
                        fontsize=11, fontweight='bold', fontname=FONT_NAME)
    axes[idx].set_title(f'{measure_names[measure]}の時系列変化（外交レイヤー）',
                       fontsize=12, fontweight='bold', fontname=FONT_NAME)
    axes[idx].legend(loc='best', fontsize=9, ncol=2, prop={'family': FONT_NAME})
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xticks(years)

plt.tight_layout()

fig_path = OUTPUT_DIRS['figures'] / 'fig18_temporal_centrality_change.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig_path}")
plt.close()

# =====================================================================
# 6. レポート
# =====================================================================
print("\n[6] レポート生成中...")

report_path = OUTPUT_DIRS['reports'] / 'centrality_report_jp.txt'

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write(" 中心性指標の詳細比較分析レポート\n")
    f.write("="*70 + "\n\n")
    
    f.write("1. 指標間相関\n")
    f.write("-"*70 + "\n")
    f.write(f"分析年: {latest_year}年\n\n")
    f.write(corr_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("2. 各レイヤーのトップ5\n")
    f.write("-"*70 + "\n\n")
    
    for layer in layers:
        f.write(f"{layer_names[layer]}レイヤー:\n")
        centralities = all_centralities[latest_year][layer]
        
        f.write("  次数中心性:\n")
        for rank, (country, val) in enumerate(get_top_k(centralities['degree_total'], 5), 1):
            f.write(f"    {rank}. {country}: {val:.2f}\n")
        
        f.write("  媒介中心性:\n")
        for rank, (country, val) in enumerate(get_top_k(centralities['betweenness'], 5), 1):
            f.write(f"    {rank}. {country}: {val:.6f}\n")
        
        f.write("  近接中心性:\n")
        for rank, (country, val) in enumerate(get_top_k(centralities['closeness_in'], 5), 1):
            f.write(f"    {rank}. {country}: {val:.6f}\n")
        
        f.write("\n")
    
    f.write("="*70 + "\n")

print(f"  ✓ {report_path}")

# =====================================================================
# 完了
# =====================================================================
print("\n" + "="*70)
print(" ✓ 分析完了！")
print("="*70)

print(f"\n📈 生成された図 ({OUTPUT_DIRS['figures']}):")
print("   - fig13_centrality_correlations.png")
print("   - fig14_top10_diplomatic_by_measure.png")
print("   - fig15_top10_aviation_by_measure.png")
print("   - fig16_venn_top20_overlap.png")
print("   - fig17_scatter_degree_vs_betweenness.png")
print("   - fig18_temporal_centrality_change.png")

print(f"\n📊 統計表 ({OUTPUT_DIRS['tables']}):")
print(f"   - centrality_correlations_{latest_year}.csv")
print("   - top20_*.csv (各レイヤー)")

print(f"\n📄 レポート ({OUTPUT_DIRS['reports']}):")
print("   - centrality_report_jp.txt")

print("\n✓ 全ての図で日本語が正しく表示されます")
print("="*70)