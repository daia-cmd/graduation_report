#!/usr/bin/env python3
"""
中心性指標の詳細比較分析

このスクリプトは論文の核心部分を分析：
1. 次数・媒介・近接中心性の詳細計算
2. 指標間の相関分析
3. 指標ごとのトップ国ランキング
4. ベン図による重複分析
5. 時系列での指標変化
6. 経済・地理的要因との関係

使用方法：
  python centrality_detailed_analysis.py
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3
from pathlib import Path
from scipy import stats
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("Set2")

print("="*70)
print(" 中心性指標の詳細比較分析 ")
print("="*70)

# =====================================================================
# 1. データ読み込み
# =====================================================================
print("\n[1] データ読み込み中...")

data_paths = [
    Path('data/raw/multilayer_network.csv'),
    Path('multilayer_network.csv')
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

layers = ['diplomatic_relation', 'aviation_routes', 'migrant_stock']
layer_names = {
    'diplomatic_relation': 'Diplomatic',
    'aviation_routes': 'Aviation',
    'migrant_stock': 'Migration'
}

years = sorted(df['year'].unique())
print(f"  年: {years}")

# =====================================================================
# 2. 中心性計算（3種類×3レイヤー×5年）
# =====================================================================
print("\n[2] 中心性計算中（3指標）...")

def calculate_all_centralities(G, weighted=True):
    """
    3種類の中心性を計算
    """
    centralities = {}
    
    # 1. Degree Centrality（次数中心性）
    if weighted and nx.is_weighted(G):
        # 重み付き
        centralities['degree_in'] = dict(G.in_degree(weight='weight'))
        centralities['degree_out'] = dict(G.out_degree(weight='weight'))
    else:
        centralities['degree_in'] = dict(G.in_degree())
        centralities['degree_out'] = dict(G.out_degree())
    
    # Total degree
    centralities['degree_total'] = {
        node: centralities['degree_in'].get(node, 0) + 
              centralities['degree_out'].get(node, 0)
        for node in G.nodes()
    }
    
    # 2. Betweenness Centrality（媒介中心性）
    try:
        if weighted and nx.is_weighted(G):
            # 重みを距離として扱う（重み大=距離小）
            # 逆数を取る
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
    
    # 3. Closeness Centrality（近接中心性）
    try:
        # In-closeness（他国からこの国への近さ）
        G_reverse = G.reverse()
        centralities['closeness_in'] = nx.closeness_centrality(
            G_reverse, distance='distance' if weighted else None)
        
        # Out-closeness（この国から他国への近さ）
        centralities['closeness_out'] = nx.closeness_centrality(
            G, distance='distance' if weighted else None)
    except:
        centralities['closeness_in'] = {node: 0 for node in G.nodes()}
        centralities['closeness_out'] = {node: 0 for node in G.nodes()}
    
    return centralities

# 全年・全レイヤーで計算
all_centralities = {}

for year in years:
    print(f"\n  ■ {year}年")
    df_year = df[df['year'] == year]
    all_centralities[year] = {}
    
    for layer in layers:
        print(f"    - {layer_names[layer]}...")
        
        # ネットワーク構築
        layer_data = df_year[df_year[layer].notna()]
        G = nx.DiGraph()
        
        for _, row in layer_data.iterrows():
            weight = row[layer]
            G.add_edge(row['origin'], row['destination'], weight=weight)
        
        # 中心性計算
        centralities = calculate_all_centralities(G, weighted=True)
        
        all_centralities[year][layer] = centralities

print("\n  ✓ 中心性計算完了")

# =====================================================================
# 3. 指標間相関分析
# =====================================================================
print("\n[3] 指標間相関分析中...")

# 最新年で分析
latest_year = max(years)

correlation_results = []

for layer in layers:
    centralities = all_centralities[latest_year][layer]
    
    # データフレーム作成
    df_cent = pd.DataFrame({
        'Degree_In': centralities['degree_in'],
        'Degree_Out': centralities['degree_out'],
        'Degree_Total': centralities['degree_total'],
        'Betweenness': centralities['betweenness'],
        'Closeness_In': centralities['closeness_in'],
        'Closeness_Out': centralities['closeness_out']
    })
    
    # 相関行列
    corr_matrix = df_cent.corr(method='spearman')
    
    print(f"\n  ■ {layer_names[layer]}レイヤー")
    print(f"    Degree-Betweenness: {corr_matrix.loc['Degree_Total', 'Betweenness']:.3f}")
    print(f"    Degree-Closeness: {corr_matrix.loc['Degree_Total', 'Closeness_In']:.3f}")
    print(f"    Betweenness-Closeness: {corr_matrix.loc['Betweenness', 'Closeness_In']:.3f}")
    
    correlation_results.append({
        'Layer': layer_names[layer],
        'Degree_Betweenness': corr_matrix.loc['Degree_Total', 'Betweenness'],
        'Degree_Closeness': corr_matrix.loc['Degree_Total', 'Closeness_In'],
        'Betweenness_Closeness': corr_matrix.loc['Betweenness', 'Closeness_In']
    })

# 保存
corr_df = pd.DataFrame(correlation_results)
corr_path = OUTPUT_DIRS['tables'] / f'centrality_measure_correlations_{latest_year}.csv'
corr_df.to_csv(corr_path, index=False, encoding='utf-8-sig')
print(f"\n  ✓ 保存: {corr_path}")

# =====================================================================
# 4. トップ20ランキング（3指標×3レイヤー）
# =====================================================================
print("\n[4] トップ20ランキング作成中...")

def get_top_k(centrality_dict, k=20):
    """トップK国を取得"""
    sorted_items = sorted(centrality_dict.items(), 
                         key=lambda x: x[1], reverse=True)
    return sorted_items[:k]

# 各指標・各レイヤーのトップ20を保存
for layer in layers:
    centralities = all_centralities[latest_year][layer]
    
    # 次数中心性
    top_degree = get_top_k(centralities['degree_total'], 20)
    df_degree = pd.DataFrame(top_degree, columns=['Country', 'Degree'])
    df_degree['Rank'] = range(1, 21)
    
    # 媒介中心性
    top_between = get_top_k(centralities['betweenness'], 20)
    df_between = pd.DataFrame(top_between, columns=['Country', 'Betweenness'])
    df_between['Rank'] = range(1, 21)
    
    # 近接中心性
    top_close = get_top_k(centralities['closeness_in'], 20)
    df_close = pd.DataFrame(top_close, columns=['Country', 'Closeness'])
    df_close['Rank'] = range(1, 21)
    
    # 保存
    layer_short = layer.replace('_', '')
    
    path = OUTPUT_DIRS['tables'] / f'top20_degree_{layer_short}_{latest_year}.csv'
    df_degree.to_csv(path, index=False, encoding='utf-8-sig')
    
    path = OUTPUT_DIRS['tables'] / f'top20_betweenness_{layer_short}_{latest_year}.csv'
    df_between.to_csv(path, index=False, encoding='utf-8-sig')
    
    path = OUTPUT_DIRS['tables'] / f'top20_closeness_{layer_short}_{latest_year}.csv'
    df_close.to_csv(path, index=False, encoding='utf-8-sig')
    
    print(f"\n  ■ {layer_names[layer]}レイヤー - トップ5")
    print(f"    次数中心性: {[c for c, _ in top_degree[:5]]}")
    print(f"    媒介中心性: {[c for c, _ in top_between[:5]]}")
    print(f"    近接中心性: {[c for c, _ in top_close[:5]]}")

print(f"\n  ✓ トップ20ランキング保存完了")

# =====================================================================
# 5. 可視化
# =====================================================================
print("\n[5] 可視化生成中...")

# ----- 図13: 指標間相関ヒートマップ（3レイヤー） -----
print("  [図13] 指標間相関ヒートマップ...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, layer in enumerate(layers):
    centralities = all_centralities[latest_year][layer]
    
    df_cent = pd.DataFrame({
        'Degree': centralities['degree_total'],
        'Betweenness': centralities['betweenness'],
        'Closeness': centralities['closeness_in']
    })
    
    corr_matrix = df_cent.corr(method='spearman')
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
               center=0.5, vmin=0, vmax=1, square=True,
               cbar_kws={'shrink': 0.8}, ax=axes[idx],
               linewidths=2, linecolor='white')
    
    axes[idx].set_title(f'{layer_names[layer]} Layer ({latest_year})',
                       fontsize=13, fontweight='bold')

plt.tight_layout()
fig_path = OUTPUT_DIRS['figures'] / 'fig13_centrality_correlations.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig_path}")
plt.close()

# ----- 図14: トップ10比較（外交レイヤー） -----
print("  [図14] トップ10比較（外交レイヤー）...")

layer = 'diplomatic_relation'
centralities = all_centralities[latest_year][layer]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 次数
top_degree = get_top_k(centralities['degree_total'], 10)
countries = [c for c, _ in top_degree]
values = [v for _, v in top_degree]

axes[0].barh(range(len(countries)), values, color='steelblue', alpha=0.8)
axes[0].set_yticks(range(len(countries)))
axes[0].set_yticklabels(countries, fontsize=10)
axes[0].set_xlabel('Degree Centrality', fontsize=11, fontweight='bold')
axes[0].set_title('Top 10: Degree', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# 媒介
top_between = get_top_k(centralities['betweenness'], 10)
countries = [c for c, _ in top_between]
values = [v for _, v in top_between]

axes[1].barh(range(len(countries)), values, color='coral', alpha=0.8)
axes[1].set_yticks(range(len(countries)))
axes[1].set_yticklabels(countries, fontsize=10)
axes[1].set_xlabel('Betweenness Centrality', fontsize=11, fontweight='bold')
axes[1].set_title('Top 10: Betweenness', fontsize=12, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

# 近接
top_close = get_top_k(centralities['closeness_in'], 10)
countries = [c for c, _ in top_close]
values = [v for _, v in top_close]

axes[2].barh(range(len(countries)), values, color='lightseagreen', alpha=0.8)
axes[2].set_yticks(range(len(countries)))
axes[2].set_yticklabels(countries, fontsize=10)
axes[2].set_xlabel('Closeness Centrality', fontsize=11, fontweight='bold')
axes[2].set_title('Top 10: Closeness', fontsize=12, fontweight='bold')
axes[2].invert_yaxis()
axes[2].grid(True, alpha=0.3, axis='x')

plt.suptitle(f'Diplomatic Layer: Top 10 Countries by Centrality Measure ({latest_year})',
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path = OUTPUT_DIRS['figures'] / 'fig14_top10_diplomatic_by_measure.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig_path}")
plt.close()

# ----- 図15: トップ10比較（航空レイヤー） -----
print("  [図15] トップ10比較（航空レイヤー）...")

layer = 'aviation_routes'
centralities = all_centralities[latest_year][layer]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

measures = ['degree_total', 'betweenness', 'closeness_in']
titles = ['Degree', 'Betweenness', 'Closeness']
colors = ['steelblue', 'coral', 'lightseagreen']

for idx, (measure, title, color) in enumerate(zip(measures, titles, colors)):
    top_k = get_top_k(centralities[measure], 10)
    countries = [c for c, _ in top_k]
    values = [v for _, v in top_k]
    
    axes[idx].barh(range(len(countries)), values, color=color, alpha=0.8)
    axes[idx].set_yticks(range(len(countries)))
    axes[idx].set_yticklabels(countries, fontsize=10)
    axes[idx].set_xlabel(f'{title} Centrality', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'Top 10: {title}', fontsize=12, fontweight='bold')
    axes[idx].invert_yaxis()
    axes[idx].grid(True, alpha=0.3, axis='x')

plt.suptitle(f'Aviation Layer: Top 10 Countries by Centrality Measure ({latest_year})',
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path = OUTPUT_DIRS['figures'] / 'fig15_top10_aviation_by_measure.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig_path}")
plt.close()

# ----- 図16: ベン図（トップ20の重複） -----
print("  [図16] ベン図（トップ20の重複）...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, layer in enumerate(layers):
    centralities = all_centralities[latest_year][layer]
    
    # トップ20の国を取得
    top_degree = set([c for c, _ in get_top_k(centralities['degree_total'], 20)])
    top_between = set([c for c, _ in get_top_k(centralities['betweenness'], 20)])
    top_close = set([c for c, _ in get_top_k(centralities['closeness_in'], 20)])
    
    # ベン図
    ax = axes[idx]
    venn = venn3([top_degree, top_between, top_close],
                 set_labels=('Degree', 'Betweenness', 'Closeness'),
                 ax=ax, alpha=0.7)
    
    # 色設定
    if venn.get_patch_by_id('100'):
        venn.get_patch_by_id('100').set_color('steelblue')
    if venn.get_patch_by_id('010'):
        venn.get_patch_by_id('010').set_color('coral')
    if venn.get_patch_by_id('001'):
        venn.get_patch_by_id('001').set_color('lightseagreen')
    
    ax.set_title(f'{layer_names[layer]} Layer', 
                fontsize=12, fontweight='bold')

plt.suptitle(f'Overlap of Top 20 Countries across Centrality Measures ({latest_year})',
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path = OUTPUT_DIRS['figures'] / 'fig16_venn_top20_overlap.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig_path}")
plt.close()

# ----- 図17: 散布図（次数 vs 媒介） -----
print("  [図17] 散布図（次数 vs 媒介）...")

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
        if deg in degree_vals and bet in between_vals:
            axes[idx].annotate(country, (deg, bet), fontsize=8,
                             xytext=(5, 5), textcoords='offset points')
    
    axes[idx].set_xlabel('Degree Centrality', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Betweenness Centrality', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'{layer_names[layer]} Layer',
                       fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    
    # 相関係数
    corr = np.corrcoef(degree_vals, between_vals)[0, 1]
    axes[idx].text(0.05, 0.95, f'r = {corr:.3f}',
                  transform=axes[idx].transAxes,
                  fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle(f'Degree vs Betweenness Centrality ({latest_year})',
            fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path = OUTPUT_DIRS['figures'] / 'fig17_degree_vs_betweenness.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig_path}")
plt.close()

# ----- 図18: 時系列（選択国の中心性変化） -----
print("  [図18] 時系列（選択国の中心性変化）...")

# 外交レイヤーで主要国を選択
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
    
    measure_names = {'degree_total': 'Degree', 
                    'betweenness': 'Betweenness',
                    'closeness_in': 'Closeness'}
    
    axes[idx].set_xlabel('Year', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel(f'{measure_names[measure]} Centrality',
                        fontsize=11, fontweight='bold')
    axes[idx].set_title(f'{measure_names[measure]} Centrality: Diplomatic Layer',
                       fontsize=12, fontweight='bold')
    axes[idx].legend(loc='best', fontsize=9, ncol=2)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xticks(years)

plt.tight_layout()
fig_path = OUTPUT_DIRS['figures'] / 'fig18_time_series_selected_countries.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig_path}")
plt.close()

# =====================================================================
# 6. サマリーレポート
# =====================================================================
print("\n[6] サマリーレポート生成中...")

report_path = OUTPUT_DIRS['reports'] / 'centrality_detailed_report.txt'

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write(" CENTRALITY MEASURES: DETAILED COMPARISON REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write("1. MEASURE CORRELATIONS\n")
    f.write("-"*70 + "\n")
    f.write(f"Year: {latest_year}\n\n")
    f.write(corr_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("2. TOP 5 COUNTRIES BY MEASURE\n")
    f.write("-"*70 + "\n\n")
    
    for layer in layers:
        f.write(f"{layer_names[layer]} Layer:\n")
        centralities = all_centralities[latest_year][layer]
        
        f.write("  Degree Centrality:\n")
        for rank, (country, val) in enumerate(get_top_k(centralities['degree_total'], 5), 1):
            f.write(f"    {rank}. {country}: {val:.2f}\n")
        
        f.write("  Betweenness Centrality:\n")
        for rank, (country, val) in enumerate(get_top_k(centralities['betweenness'], 5), 1):
            f.write(f"    {rank}. {country}: {val:.6f}\n")
        
        f.write("  Closeness Centrality:\n")
        for rank, (country, val) in enumerate(get_top_k(centralities['closeness_in'], 5), 1):
            f.write(f"    {rank}. {country}: {val:.6f}\n")
        
        f.write("\n")
    
    f.write("3. KEY FINDINGS\n")
    f.write("-"*70 + "\n\n")
    
    # 自動的な発見
    for layer in layers:
        centralities = all_centralities[latest_year][layer]
        
        top_deg = set([c for c, _ in get_top_k(centralities['degree_total'], 20)])
        top_bet = set([c for c, _ in get_top_k(centralities['betweenness'], 20)])
        top_clo = set([c for c, _ in get_top_k(centralities['closeness_in'], 20)])
        
        # 全ての指標でトップ20に入る国
        all_three = top_deg.intersection(top_bet).intersection(top_clo)
        
        f.write(f"{layer_names[layer]} Layer:\n")
        f.write(f"  - Countries in top-20 for ALL measures: {len(all_three)}\n")
        f.write(f"    {sorted(list(all_three))}\n")
        f.write(f"  - Degree-only leaders: {len(top_deg - top_bet - top_clo)}\n")
        f.write(f"  - Betweenness-only leaders: {len(top_bet - top_deg - top_clo)}\n")
        f.write(f"  - Closeness-only leaders: {len(top_clo - top_deg - top_bet)}\n")
        f.write("\n")
    
    f.write("="*70 + "\n")

print(f"  ✓ {report_path}")

# =====================================================================
# 完了
# =====================================================================
print("\n" + "="*70)
print(" ✓ 中心性指標の詳細比較分析完了！ ")
print("="*70)

print("\n【生成されたファイル】")

print(f"\n📈 グラフ ({OUTPUT_DIRS['figures']}):")
print("   - fig13_centrality_correlations.png")
print("   - fig14_top10_diplomatic_by_measure.png")
print("   - fig15_top10_aviation_by_measure.png")
print("   - fig16_venn_top20_overlap.png")
print("   - fig17_degree_vs_betweenness.png")
print("   - fig18_time_series_selected_countries.png")

print(f"\n📊 統計表 ({OUTPUT_DIRS['tables']}):")
print(f"   - centrality_measure_correlations_{latest_year}.csv")
print("   - top20_degree_*.csv (3レイヤー)")
print("   - top20_betweenness_*.csv (3レイヤー)")
print("   - top20_closeness_*.csv (3レイヤー)")

print(f"\n📄 レポート ({OUTPUT_DIRS['reports']}):")
print("   - centrality_detailed_report.txt")

print("\n次のステップ:")
print("  1. 生成されたグラフを論文に使用")
print("  2. 指標間の違いを解釈")
print("  3. Results セクションを執筆")

print("\n" + "="*70)