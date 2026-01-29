#!/usr/bin/env python3
"""
マルチレイヤーネットワーク中心性分析

このスクリプトは以下を実行：
1. 各レイヤーのネットワーク構築
2. 中心性指標の計算（PageRank, Degree, Betweenness）
3. マルチレイヤー統合中心性
4. トップ国のランキング
5. 時系列での変化分析
6. 可視化

使用方法：
  python network_centrality_analysis.py
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# グラフスタイル
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

print("="*70)
print(" マルチレイヤーネットワーク中心性分析 ")
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

print(f"  総行数: {len(df):,}")
print(f"  年: {sorted(df['year'].unique())}")

layers = ['diplomatic_relation', 'aviation_routes', 'migrant_stock']
layer_names = {
    'diplomatic_relation': 'Diplomatic',
    'aviation_routes': 'Aviation',
    'migrant_stock': 'Migration'
}

# 出力フォルダ確認
OUTPUT_DIRS = {
    'figures': Path('outputs/figures'),
    'tables': Path('outputs/tables'),
    'reports': Path('outputs/reports')
}

for path in OUTPUT_DIRS.values():
    path.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 2. ネットワーク構築関数
# =====================================================================
print("\n[2] ネットワーク構築関数定義...")

def build_network(data, layer, weighted=True):
    """
    指定されたレイヤーのネットワークを構築
    
    Parameters:
    -----------
    data : DataFrame
        ネットワークデータ
    layer : str
        レイヤー名
    weighted : bool
        重み付きグラフかどうか
    
    Returns:
    --------
    G : nx.DiGraph
        構築されたネットワーク
    """
    # データフィルタ
    layer_data = data[data[layer].notna()].copy()
    
    # ネットワーク構築
    G = nx.DiGraph()
    
    for _, row in layer_data.iterrows():
        origin = row['origin']
        destination = row['destination']
        weight = row[layer]
        
        if weighted:
            if G.has_edge(origin, destination):
                # 既存のエッジの重みを加算
                G[origin][destination]['weight'] += weight
            else:
                G.add_edge(origin, destination, weight=weight)
        else:
            G.add_edge(origin, destination)
    
    return G

def calculate_centralities(G, weighted=True):
    """
    ネットワークの中心性指標を計算
    
    Parameters:
    -----------
    G : nx.DiGraph
        ネットワーク
    weighted : bool
        重み付き中心性を計算するか
    
    Returns:
    --------
    centralities : dict
        各中心性指標の辞書
    """
    centralities = {}
    
    # PageRank
    try:
        if weighted and nx.is_weighted(G):
            centralities['pagerank'] = nx.pagerank(G, weight='weight')
        else:
            centralities['pagerank'] = nx.pagerank(G)
    except:
        centralities['pagerank'] = {}
    
    # In-Degree（入次数）
    if weighted and nx.is_weighted(G):
        centralities['in_degree'] = dict(G.in_degree(weight='weight'))
    else:
        centralities['in_degree'] = dict(G.in_degree())
    
    # Out-Degree（出次数）
    if weighted and nx.is_weighted(G):
        centralities['out_degree'] = dict(G.out_degree(weight='weight'))
    else:
        centralities['out_degree'] = dict(G.out_degree())
    
    # Betweenness Centrality（媒介中心性）
    try:
        if weighted and nx.is_weighted(G):
            centralities['betweenness'] = nx.betweenness_centrality(
                G, weight='weight')
        else:
            centralities['betweenness'] = nx.betweenness_centrality(G)
    except:
        centralities['betweenness'] = {}
    
    # Closeness Centrality（近接中心性）
    try:
        centralities['closeness'] = nx.closeness_centrality(G)
    except:
        centralities['closeness'] = {}
    
    return centralities

print("  ✓ 関数定義完了")

# =====================================================================
# 3. 各年・各レイヤーの中心性計算
# =====================================================================
print("\n[3] 中心性計算中...")

years = sorted(df['year'].unique())
all_centralities = {}

for year in years:
    print(f"\n  ■ {year}年")
    df_year = df[df['year'] == year]
    all_centralities[year] = {}
    
    for layer in layers:
        print(f"    - {layer_names[layer]}レイヤー...")
        
        # ネットワーク構築
        G = build_network(df_year, layer, weighted=True)
        
        print(f"      ノード数: {G.number_of_nodes()}")
        print(f"      エッジ数: {G.number_of_edges()}")
        
        # 中心性計算
        centralities = calculate_centralities(G, weighted=True)
        
        all_centralities[year][layer] = {
            'network': G,
            'centralities': centralities
        }

print("\n  ✓ 中心性計算完了")

# =====================================================================
# 4. トップ国ランキング
# =====================================================================
print("\n[4] トップ国ランキング作成中...")

def get_top_countries(centralities, metric='pagerank', n=20):
    """
    指定された中心性指標でトップN国を取得
    """
    if metric not in centralities or not centralities[metric]:
        return []
    
    sorted_countries = sorted(centralities[metric].items(), 
                             key=lambda x: x[1], reverse=True)
    return sorted_countries[:n]

# 最新年のランキングを表示
latest_year = max(years)
print(f"\n【{latest_year}年のトップ20カ国】")
print("-"*70)

for layer in layers:
    print(f"\n■ {layer_names[layer]}レイヤー (PageRank)")
    
    centralities = all_centralities[latest_year][layer]['centralities']
    top_countries = get_top_countries(centralities, 'pagerank', 20)
    
    for rank, (country, score) in enumerate(top_countries, 1):
        print(f"  {rank:2d}. {country:3s}: {score:.6f}")

# =====================================================================
# 5. マルチレイヤー統合中心性
# =====================================================================
print("\n[5] マルチレイヤー統合中心性計算中...")

multilayer_centrality = {}

for year in years:
    # 各レイヤーのPageRankを正規化して統合
    all_countries = set()
    
    # 全ての国を収集
    for layer in layers:
        centralities = all_centralities[year][layer]['centralities']
        all_countries.update(centralities['pagerank'].keys())
    
    # 統合スコア計算
    integrated_scores = {}
    
    for country in all_countries:
        scores = []
        
        for layer in layers:
            centralities = all_centralities[year][layer]['centralities']
            score = centralities['pagerank'].get(country, 0)
            scores.append(score)
        
        # 平均（等重み）
        integrated_scores[country] = np.mean(scores)
    
    multilayer_centrality[year] = integrated_scores

print(f"\n【{latest_year}年のマルチレイヤー統合中心性トップ20】")
print("-"*70)

sorted_countries = sorted(multilayer_centrality[latest_year].items(),
                         key=lambda x: x[1], reverse=True)

for rank, (country, score) in enumerate(sorted_countries[:20], 1):
    print(f"  {rank:2d}. {country:3s}: {score:.6f}")

# =====================================================================
# 6. 中心性の時系列変化
# =====================================================================
print("\n[6] 中心性の時系列変化分析中...")

# トップ10カ国を選択（最新年のマルチレイヤー中心性）
top10_countries = [country for country, _ in sorted_countries[:10]]

# 各国の時系列データを収集
time_series_data = []

for country in top10_countries:
    for year in years:
        score = multilayer_centrality[year].get(country, 0)
        time_series_data.append({
            'Country': country,
            'Year': year,
            'Centrality': score
        })

ts_df = pd.DataFrame(time_series_data)

# =====================================================================
# 7. 可視化
# =====================================================================
print("\n[7] 可視化生成中...")

# ----- 図7: レイヤー別トップ10国（最新年） -----
print("  [図7] レイヤー別トップ10国...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, layer in enumerate(layers):
    centralities = all_centralities[latest_year][layer]['centralities']
    top_countries = get_top_countries(centralities, 'pagerank', 10)
    
    countries = [c for c, _ in top_countries]
    scores = [s for _, s in top_countries]
    
    axes[idx].barh(range(len(countries)), scores, color='steelblue', alpha=0.8)
    axes[idx].set_yticks(range(len(countries)))
    axes[idx].set_yticklabels(countries, fontsize=10)
    axes[idx].set_xlabel('PageRank', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'{layer_names[layer]} Layer ({latest_year})', 
                       fontsize=12, fontweight='bold')
    axes[idx].invert_yaxis()
    axes[idx].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
fig_path = OUTPUT_DIRS['figures'] / 'fig7_top10_by_layer.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig_path}")
plt.close()

# ----- 図8: マルチレイヤー統合中心性トップ20 -----
print("  [図8] マルチレイヤー統合中心性トップ20...")

fig, ax = plt.subplots(figsize=(10, 8))

top20 = sorted_countries[:20]
countries = [c for c, _ in top20]
scores = [s for _, s in top20]

ax.barh(range(len(countries)), scores, color='coral', alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(countries)))
ax.set_yticklabels(countries, fontsize=11)
ax.set_xlabel('Integrated Centrality Score', fontsize=12, fontweight='bold')
ax.set_title(f'Top 20 Countries: Multilayer Centrality ({latest_year})', 
             fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
fig_path = OUTPUT_DIRS['figures'] / 'fig8_multilayer_top20.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig_path}")
plt.close()

# ----- 図9: トップ10国の時系列変化 -----
print("  [図9] トップ10国の時系列変化...")

fig, ax = plt.subplots(figsize=(12, 7))

for country in top10_countries:
    country_data = ts_df[ts_df['Country'] == country]
    ax.plot(country_data['Year'], country_data['Centrality'], 
           marker='o', linewidth=2, markersize=6, label=country)

ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Multilayer Centrality', fontsize=12, fontweight='bold')
ax.set_title('Time Series: Top 10 Countries Centrality', 
            fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xticks(years)

plt.tight_layout()
fig_path = OUTPUT_DIRS['figures'] / 'fig9_centrality_time_series.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig_path}")
plt.close()

# ----- 図10: 中心性指標の比較（最新年） -----
print("  [図10] 中心性指標の比較...")

# 外交レイヤーの複数指標を比較
layer = 'diplomatic_relation'
centralities = all_centralities[latest_year][layer]['centralities']

metrics = ['pagerank', 'in_degree', 'betweenness']
metric_names = ['PageRank', 'In-Degree', 'Betweenness']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
    if metric in centralities and centralities[metric]:
        top = sorted(centralities[metric].items(), 
                    key=lambda x: x[1], reverse=True)[:10]
        
        countries = [c for c, _ in top]
        scores = [s for _, s in top]
        
        axes[idx].barh(range(len(countries)), scores, 
                      color='lightseagreen', alpha=0.8)
        axes[idx].set_yticks(range(len(countries)))
        axes[idx].set_yticklabels(countries, fontsize=10)
        axes[idx].set_xlabel(name, fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{name} - {layer_names[layer]} ({latest_year})', 
                          fontsize=12, fontweight='bold')
        axes[idx].invert_yaxis()
        axes[idx].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
fig_path = OUTPUT_DIRS['figures'] / 'fig10_centrality_metrics_comparison.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig_path}")
plt.close()

# ----- 図11: レイヤー間の中心性相関（最新年） -----
print("  [図11] レイヤー間の中心性相関...")

# 各レイヤーのPageRankを収集
all_countries = set()
for layer in layers:
    centralities = all_centralities[latest_year][layer]['centralities']
    all_countries.update(centralities['pagerank'].keys())

centrality_comparison = []

for country in all_countries:
    row = {'Country': country}
    
    for layer in layers:
        centralities = all_centralities[latest_year][layer]['centralities']
        row[layer_names[layer]] = centralities['pagerank'].get(country, 0)
    
    centrality_comparison.append(row)

comp_df = pd.DataFrame(centrality_comparison)

# 相関行列
corr_matrix = comp_df[['Diplomatic', 'Aviation', 'Migration']].corr(method='spearman')

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
           center=0, vmin=-1, vmax=1, square=True,
           cbar_kws={'shrink': 0.8}, ax=ax,
           linewidths=1, linecolor='black')

ax.set_title(f'Layer Centrality Correlation ({latest_year})', 
            fontsize=14, fontweight='bold')

plt.tight_layout()
fig_path = OUTPUT_DIRS['figures'] / 'fig11_layer_centrality_correlation.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig_path}")
plt.close()

# ----- 図12: 中心性のランク変化（2000 vs 2020） -----
print("  [図12] 中心性のランク変化...")

if 2000 in years and 2020 in years:
    # 2000年と2020年のランキング
    rank_2000 = sorted(multilayer_centrality[2000].items(),
                      key=lambda x: x[1], reverse=True)
    rank_2020 = sorted(multilayer_centrality[2020].items(),
                      key=lambda x: x[1], reverse=True)
    
    # トップ20の国を選択
    top20_2000 = set([c for c, _ in rank_2000[:20]])
    top20_2020 = set([c for c, _ in rank_2020[:20]])
    top20_union = top20_2000.union(top20_2020)
    
    # ランク変化を計算
    rank_changes = []
    
    for country in top20_union:
        rank_2000_val = next((i+1 for i, (c, _) in enumerate(rank_2000) 
                             if c == country), None)
        rank_2020_val = next((i+1 for i, (c, _) in enumerate(rank_2020) 
                             if c == country), None)
        
        if rank_2000_val and rank_2020_val:
            change = rank_2000_val - rank_2020_val  # 正=上昇
            rank_changes.append({
                'Country': country,
                'Rank_2000': rank_2000_val,
                'Rank_2020': rank_2020_val,
                'Change': change
            })
    
    rank_changes_df = pd.DataFrame(rank_changes)
    rank_changes_df = rank_changes_df.sort_values('Change', ascending=False)
    
    # 可視化
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_gainers = rank_changes_df.head(10)
    top_losers = rank_changes_df.tail(10).iloc[::-1]
    
    plot_data = pd.concat([top_gainers, top_losers])
    
    colors = ['green' if x > 0 else 'red' for x in plot_data['Change']]
    
    ax.barh(range(len(plot_data)), plot_data['Change'], 
           color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data['Country'], fontsize=10)
    ax.set_xlabel('Rank Change (2000→2020)', fontsize=12, fontweight='bold')
    ax.set_title('Top Gainers and Losers in Centrality Ranking', 
                fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    fig_path = OUTPUT_DIRS['figures'] / 'fig12_rank_changes_2000_2020.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ {fig_path}")
    plt.close()

# =====================================================================
# 8. 統計表の保存
# =====================================================================
print("\n[8] 統計表保存中...")

# 最新年のレイヤー別トップ20
for layer in layers:
    centralities = all_centralities[latest_year][layer]['centralities']
    top20 = get_top_countries(centralities, 'pagerank', 20)
    
    df_top20 = pd.DataFrame(top20, columns=['Country', 'PageRank'])
    df_top20['Rank'] = range(1, len(df_top20) + 1)
    df_top20 = df_top20[['Rank', 'Country', 'PageRank']]
    
    filename = f'centrality_{layer}_{latest_year}.csv'
    output_path = OUTPUT_DIRS['tables'] / filename
    df_top20.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ {output_path}")

# マルチレイヤー統合中心性トップ20
df_multilayer = pd.DataFrame(sorted_countries[:20], 
                            columns=['Country', 'Centrality'])
df_multilayer['Rank'] = range(1, 21)
df_multilayer = df_multilayer[['Rank', 'Country', 'Centrality']]

output_path = OUTPUT_DIRS['tables'] / f'centrality_multilayer_{latest_year}.csv'
df_multilayer.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"  ✓ {output_path}")

# 時系列データ
output_path = OUTPUT_DIRS['tables'] / 'centrality_time_series.csv'
ts_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"  ✓ {output_path}")

# レイヤー間相関
output_path = OUTPUT_DIRS['tables'] / f'layer_centrality_correlation_{latest_year}.csv'
corr_matrix.to_csv(output_path, encoding='utf-8-sig')
print(f"  ✓ {output_path}")

# =====================================================================
# 9. サマリーレポート
# =====================================================================
print("\n[9] サマリーレポート生成中...")

report_path = OUTPUT_DIRS['reports'] / 'network_centrality_report.txt'

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write(" NETWORK CENTRALITY ANALYSIS REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write("1. ANALYSIS OVERVIEW\n")
    f.write("-"*70 + "\n")
    f.write(f"Analysis Period: {min(years)} - {max(years)}\n")
    f.write(f"Number of Years: {len(years)}\n")
    f.write(f"Layers Analyzed: {len(layers)}\n\n")
    
    f.write("2. TOP 20 COUNTRIES (MULTILAYER CENTRALITY)\n")
    f.write("-"*70 + "\n")
    f.write(f"Year: {latest_year}\n\n")
    
    for rank, (country, score) in enumerate(sorted_countries[:20], 1):
        f.write(f"{rank:2d}. {country:3s}: {score:.6f}\n")
    
    f.write("\n3. TOP 10 BY LAYER\n")
    f.write("-"*70 + "\n")
    
    for layer in layers:
        f.write(f"\n{layer_names[layer]} Layer:\n")
        centralities = all_centralities[latest_year][layer]['centralities']
        top10 = get_top_countries(centralities, 'pagerank', 10)
        
        for rank, (country, score) in enumerate(top10, 1):
            f.write(f"  {rank:2d}. {country:3s}: {score:.6f}\n")
    
    f.write("\n4. LAYER CENTRALITY CORRELATION\n")
    f.write("-"*70 + "\n")
    f.write(f"Year: {latest_year}\n\n")
    f.write(corr_matrix.to_string())
    f.write("\n\n")
    
    f.write("5. KEY FINDINGS\n")
    f.write("-"*70 + "\n")
    
    # 相関分析
    diplo_avia_corr = corr_matrix.loc['Diplomatic', 'Aviation']
    diplo_migr_corr = corr_matrix.loc['Diplomatic', 'Migration']
    avia_migr_corr = corr_matrix.loc['Aviation', 'Migration']
    
    f.write(f"- Diplomatic-Aviation correlation: {diplo_avia_corr:.3f}\n")
    f.write(f"- Diplomatic-Migration correlation: {diplo_migr_corr:.3f}\n")
    f.write(f"- Aviation-Migration correlation: {avia_migr_corr:.3f}\n\n")
    
    if avia_migr_corr > 0.5:
        f.write("→ Strong correlation between Aviation and Migration centrality\n")
    
    if abs(diplo_avia_corr) < 0.3 and abs(diplo_migr_corr) < 0.3:
        f.write("→ Diplomatic centrality is relatively independent from other layers\n")
    
    f.write("\n" + "="*70 + "\n")

print(f"  ✓ {report_path}")

# =====================================================================
# 完了
# =====================================================================
print("\n" + "="*70)
print(" ✓ ネットワーク中心性分析完了！ ")
print("="*70)

print("\n【生成されたファイル】")

print(f"\n📈 グラフ ({OUTPUT_DIRS['figures']}):")
print("   - fig7_top10_by_layer.png")
print("   - fig8_multilayer_top20.png")
print("   - fig9_centrality_time_series.png")
print("   - fig10_centrality_metrics_comparison.png")
print("   - fig11_layer_centrality_correlation.png")
print("   - fig12_rank_changes_2000_2020.png")

print(f"\n📊 統計表 ({OUTPUT_DIRS['tables']}):")
print(f"   - centrality_diplomatic_relation_{latest_year}.csv")
print(f"   - centrality_aviation_routes_{latest_year}.csv")
print(f"   - centrality_migrant_stock_{latest_year}.csv")
print(f"   - centrality_multilayer_{latest_year}.csv")
print("   - centrality_time_series.csv")
print(f"   - layer_centrality_correlation_{latest_year}.csv")

print(f"\n📄 レポート ({OUTPUT_DIRS['reports']}):")
print("   - network_centrality_report.txt")

print("\n次のステップ:")
print("  1. outputs/figures/ のグラフを確認")
print("  2. トップ国のランキングを分析")
print("  3. 相関パターンを解釈")
print("  4. 論文のResultsセクションを執筆")

print("\n" + "="*70)