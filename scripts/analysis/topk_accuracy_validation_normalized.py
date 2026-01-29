#!/usr/bin/env python3
"""
統合版：マルチレイヤーネットワーク中心性分析 + Top-k Accuracy検証
【データ量正規化版】

このスクリプトは以下を実行：
【既存機能】
1. 各レイヤーのネットワーク構築と中心性計算
2. マルチレイヤー統合中心性
3. トップ国のランキング
4. 時系列での変化分析
5. 既存の可視化（図7-12）

【新機能】
6. Top-k Accuracy検証（8通りのレイヤー組み合わせ）
7. 訓練/テスト分割による予測精度評価
8. 新規可視化（図13-15）

【重み付け手法】
- 均等重み（1/N）：オリジナル版
- データ量正規化：各レイヤーの総エッジ重みで正規化

使用方法：
  python topk_accuracy_validation_normalized.py
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
print(" 統合版：マルチレイヤーネットワーク分析（データ量正規化版） ")
print("="*70)

# =====================================================================
# 設定
# =====================================================================

# データパス
MIGRATION_FILE = Path('data/processed/migration_network.csv')
DIPLOMACY_FILE = Path('data/processed/diplomatic_network.csv')
AVIATION_FILE = Path('data/midstate/aviation_network_raw.csv')
TRADE_FILE = Path('data/processed/trade_network.csv')

# 出力フォルダ
OUTPUT_DIRS = {
    'figures': Path('outputs/figures'),
    'tables': Path('outputs/tables'),
    'reports': Path('outputs/reports')
}

for path in OUTPUT_DIRS.values():
    path.mkdir(parents=True, exist_ok=True)

# 訓練・テスト分割
TRAIN_YEARS = [2000, 2005, 2010, 2015]
TEST_YEAR = 2020
ALL_YEARS = TRAIN_YEARS + [TEST_YEAR]
TOP_K = 20

# レイヤー設定
LAYER_COLUMNS = {
    'migration': 'migrant_stock',
    'diplomacy': 'diplomatic_relation',
    'aviation': 'aviation_routes',
    'trade': 'trade_value'
}

LAYER_NAMES = {
    'migration': 'Migration',
    'diplomacy': 'Diplomatic',
    'aviation': 'Aviation',
    'trade': 'Trade'
}

# Top-k検証用のレイヤー組み合わせ（8通り）
LAYER_COMBINATIONS = [
    {'name': '移住のみ', 'layers': ['migration'], 'code': 'M'},
    {'name': '移住+外交', 'layers': ['migration', 'diplomacy'], 'code': 'MD'},
    {'name': '移住+航空', 'layers': ['migration', 'aviation'], 'code': 'MA'},
    {'name': '移住+貿易', 'layers': ['migration', 'trade'], 'code': 'MT'},
    {'name': '移住+外交+航空', 'layers': ['migration', 'diplomacy', 'aviation'], 'code': 'MDA'},
    {'name': '移住+外交+貿易', 'layers': ['migration', 'diplomacy', 'trade'], 'code': 'MDT'},
    {'name': '移住+航空+貿易', 'layers': ['migration', 'aviation', 'trade'], 'code': 'MAT'},
    {'name': '全レイヤー', 'layers': ['migration', 'diplomacy', 'aviation', 'trade'], 'code': 'ALL'},
]

# =====================================================================
# 1. データ読み込み
# =====================================================================
print("\n[1] データ読み込み中...")

def load_layer_data(filepath, layer_name):
    """レイヤーデータを読み込み"""
    if not filepath.exists():
        print(f"  ⚠ 警告: {filepath} が見つかりません")
        return None
    
    df = pd.read_csv(filepath)
    print(f"  ✓ {layer_name}: {len(df):,}行")
    return df

# 各レイヤーのデータ読み込み
migration_df = load_layer_data(MIGRATION_FILE, 'Migration')
diplomacy_df = load_layer_data(DIPLOMACY_FILE, 'Diplomacy')
aviation_df = load_layer_data(AVIATION_FILE, 'Aviation')
trade_df = load_layer_data(TRADE_FILE, 'Trade')

# カラム名の統一
if diplomacy_df is not None and 'embassy_level' in diplomacy_df.columns:
    diplomacy_df = diplomacy_df.rename(columns={'embassy_level': 'diplomatic_relation'})

if aviation_df is not None and 'route_count' in aviation_df.columns:
    aviation_df = aviation_df.rename(columns={'route_count': 'aviation_routes'})

# レイヤーデータを辞書にまとめる
layer_data = {
    'migration': migration_df,
    'diplomacy': diplomacy_df,
    'aviation': aviation_df,
    'trade': trade_df
}

# =====================================================================
# 2. レイヤーごとの総重み計算（データ量正規化用）
# =====================================================================
print("\n[2] レイヤーごとの総重み計算（正規化用）...")

def calculate_layer_total_weight(df, layer_name, years):
    """指定された年のレイヤーの総重みを計算"""
    if df is None:
        return 0
    
    df_filtered = df[df['year'].isin(years)].copy()
    col = LAYER_COLUMNS[layer_name]
    
    total = df_filtered[col].sum()
    return total if not pd.isna(total) else 0

# 訓練データの各レイヤーの総重み
layer_total_weights = {}

for layer_name in ['migration', 'diplomacy', 'aviation', 'trade']:
    total_weight = calculate_layer_total_weight(
        layer_data[layer_name], 
        layer_name, 
        TRAIN_YEARS
    )
    layer_total_weights[layer_name] = total_weight
    print(f"  {LAYER_NAMES[layer_name]:12s}: {total_weight:>20,.2f}")

# 正規化係数の計算
print("\n  正規化係数:")
for layer_name, total in layer_total_weights.items():
    if total > 0:
        norm_factor = 1.0 / total
        print(f"  {LAYER_NAMES[layer_name]:12s}: {norm_factor:.2e}")

# =====================================================================
# 3. ネットワーク構築関数（データ量正規化対応）
# =====================================================================
print("\n[3] ネットワーク構築関数定義...")

def build_network(data, layer_name, years=None, weighted=True):
    """指定されたレイヤーと年のネットワークを構築"""
    if data is None:
        return nx.DiGraph()
    
    if years is not None:
        df = data[data['year'].isin(years)].copy()
    else:
        df = data.copy()
    
    G = nx.DiGraph()
    col = LAYER_COLUMNS[layer_name]
    
    for _, row in df.iterrows():
        origin = row['origin']
        dest = row['destination']
        value = row[col]
        
        if pd.isna(value) or value <= 0:
            continue
        
        if weighted:
            if G.has_edge(origin, dest):
                G[origin][dest]['weight'] += value
            else:
                G.add_edge(origin, dest, weight=value)
        else:
            G.add_edge(origin, dest)
    
    return G

def build_multilayer_network(layer_dfs, layer_names, years=None, weight_method='uniform'):
    """
    多層ネットワークを構築
    
    Parameters:
    -----------
    layer_dfs : dict
        各レイヤーのDataFrame
    layer_names : list
        使用するレイヤー名のリスト
    years : list or None
        対象年
    weight_method : str
        重み付け方法
        - 'uniform': 均等重み (1/N)
        - 'normalized': データ量正規化
    
    Returns:
    --------
    G : nx.DiGraph
        構築されたネットワーク
    stats : dict
        統計情報
    """
    G = nx.DiGraph()
    n_layers = len(layer_names)
    
    # 統計情報
    stats = {
        'layer_contributions': {},
        'total_edges': 0,
        'total_weight': 0
    }
    
    for layer_name in layer_names:
        df = layer_dfs[layer_name]
        if df is None:
            continue
        
        # 年でフィルタ
        if years is not None:
            df = df[df['year'].isin(years)].copy()
        
        col = LAYER_COLUMNS[layer_name]
        
        # このレイヤーの寄与を記録
        layer_edges = 0
        layer_weight = 0
        
        for _, row in df.iterrows():
            origin = row['origin']
            dest = row['destination']
            value = row[col]
            
            if pd.isna(value) or value <= 0:
                continue
            
            # 重み付け方法に応じて重みを計算
            if weight_method == 'uniform':
                # 均等重み: 1/N
                edge_weight = value / n_layers
            elif weight_method == 'normalized':
                # データ量正規化: 総重みで割る
                total = layer_total_weights.get(layer_name, 1)
                if total > 0:
                    edge_weight = value / total
                else:
                    edge_weight = 0
            else:
                edge_weight = value
            
            # エッジの追加（重み累積）
            if G.has_edge(origin, dest):
                G[origin][dest]['weight'] += edge_weight
            else:
                G.add_edge(origin, dest, weight=edge_weight)
            
            layer_edges += 1
            layer_weight += edge_weight
        
        stats['layer_contributions'][layer_name] = {
            'edges': layer_edges,
            'total_weight': layer_weight
        }
    
    stats['total_edges'] = G.number_of_edges()
    stats['total_weight'] = sum(data['weight'] for _, _, data in G.edges(data=True))
    
    return G, stats

def calculate_centralities(G, weighted=True):
    """複数の中心性指標を計算"""
    centralities = {}
    
    if G.number_of_nodes() == 0:
        return centralities
    
    # PageRank
    try:
        if weighted:
            centralities['pagerank'] = nx.pagerank(G, weight='weight')
        else:
            centralities['pagerank'] = nx.pagerank(G)
    except:
        centralities['pagerank'] = {}
    
    # 次数中心性
    if weighted:
        centralities['in_degree'] = dict(G.in_degree(weight='weight'))
        centralities['out_degree'] = dict(G.out_degree(weight='weight'))
    else:
        centralities['in_degree'] = dict(G.in_degree())
        centralities['out_degree'] = dict(G.out_degree())
    
    # 媒介中心性
    try:
        if weighted:
            centralities['betweenness'] = nx.betweenness_centrality(G, weight='weight')
        else:
            centralities['betweenness'] = nx.betweenness_centrality(G)
    except:
        centralities['betweenness'] = {}
    
    return centralities

# =====================================================================
# 4. 各レイヤーの中心性計算（全年）
# =====================================================================
print("\n[4] 各レイヤーの中心性計算中...")

all_centralities = {}

for year in ALL_YEARS:
    print(f"\n  【{year}年】")
    all_centralities[year] = {}
    
    for layer_name, layer_df in layer_data.items():
        if layer_df is None:
            continue
        
        print(f"    処理中: {LAYER_NAMES[layer_name]}...")
        
        # ネットワーク構築（単一年）
        G = build_network(layer_df, layer_name, years=[year], weighted=True)
        
        print(f"      ノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}")
        
        # 中心性計算
        centralities = calculate_centralities(G, weighted=True)
        
        all_centralities[year][layer_name] = {
            'network': G,
            'centralities': centralities
        }

print("\n  ✓ 中心性計算完了")

# =====================================================================
# 5. トップ国ランキング（既存機能）
# =====================================================================
print("\n[5] トップ国ランキング作成中...")

def get_top_countries(centralities, metric='pagerank', n=20):
    """指定された中心性指標でトップN国を取得"""
    if metric not in centralities or not centralities[metric]:
        return []
    
    sorted_countries = sorted(centralities[metric].items(), 
                             key=lambda x: x[1], reverse=True)
    return sorted_countries[:n]

# 最新年のランキングを表示
latest_year = TEST_YEAR
print(f"\n【{latest_year}年のトップ20カ国】")
print("-"*70)

for layer_name in ['migration', 'diplomacy', 'aviation', 'trade']:
    if layer_name not in all_centralities[latest_year]:
        continue
    
    print(f"\n■ {LAYER_NAMES[layer_name]}レイヤー (PageRank)")
    
    centralities = all_centralities[latest_year][layer_name]['centralities']
    top_countries = get_top_countries(centralities, 'pagerank', 20)
    
    for rank, (country, score) in enumerate(top_countries, 1):
        print(f"  {rank:2d}. {country:3s}: {score:.6f}")

# =====================================================================
# 6. マルチレイヤー統合中心性（既存機能）
# =====================================================================
print("\n[6] マルチレイヤー統合中心性計算中...")

available_layers = [l for l in ['migration', 'diplomacy', 'aviation', 'trade'] 
                   if layer_data[l] is not None]

multilayer_centrality = {}

for year in ALL_YEARS:
    all_countries = set()
    
    for layer_name in available_layers:
        if layer_name in all_centralities[year]:
            centralities = all_centralities[year][layer_name]['centralities']
            all_countries.update(centralities['pagerank'].keys())
    
    integrated_scores = {}
    
    for country in all_countries:
        scores = []
        
        for layer_name in available_layers:
            if layer_name in all_centralities[year]:
                centralities = all_centralities[year][layer_name]['centralities']
                score = centralities['pagerank'].get(country, 0)
                scores.append(score)
        
        integrated_scores[country] = np.mean(scores) if scores else 0
    
    multilayer_centrality[year] = integrated_scores

print(f"\n【{latest_year}年のマルチレイヤー統合中心性トップ20】")
print("-"*70)

sorted_countries = sorted(multilayer_centrality[latest_year].items(),
                         key=lambda x: x[1], reverse=True)

for rank, (country, score) in enumerate(sorted_countries[:20], 1):
    print(f"  {rank:2d}. {country:3s}: {score:.6f}")

# =====================================================================
# 7. 時系列データの準備
# =====================================================================
print("\n[7] 時系列データの準備...")

top10_countries = [country for country, _ in sorted_countries[:10]]

time_series_data = []
for country in top10_countries:
    for year in ALL_YEARS:
        score = multilayer_centrality[year].get(country, 0)
        time_series_data.append({
            'Country': country,
            'Year': year,
            'Centrality': score
        })

ts_df = pd.DataFrame(time_series_data)

# =====================================================================
# 8. 既存の可視化（図7-12）
# =====================================================================
print("\n[8] 既存の可視化生成中...")

# ----- 図7: レイヤー別トップ10国（最新年） -----
print("  [図7] レイヤー別トップ10国...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, layer_name in enumerate(available_layers):
    if layer_name not in all_centralities[latest_year]:
        continue
    
    centralities = all_centralities[latest_year][layer_name]['centralities']
    top_countries = get_top_countries(centralities, 'pagerank', 10)
    
    countries = [c for c, _ in top_countries]
    scores = [s for _, s in top_countries]
    
    axes[idx].barh(range(len(countries)), scores, color='steelblue', alpha=0.8)
    axes[idx].set_yticks(range(len(countries)))
    axes[idx].set_yticklabels(countries, fontsize=10)
    axes[idx].set_xlabel('PageRank', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'{LAYER_NAMES[layer_name]} Layer ({latest_year})', 
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
ax.set_xticks(ALL_YEARS)

plt.tight_layout()
fig_path = OUTPUT_DIRS['figures'] / 'fig9_centrality_time_series.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig_path}")
plt.close()

# ----- 図10: 中心性指標の比較（最新年） -----
print("  [図10] 中心性指標の比較...")

layer_name = 'migration'
if layer_name in all_centralities[latest_year]:
    centralities = all_centralities[latest_year][layer_name]['centralities']
    
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
            axes[idx].set_title(f'{name} - {LAYER_NAMES[layer_name]} ({latest_year})', 
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

all_countries_corr = set()
for layer_name in available_layers:
    if layer_name in all_centralities[latest_year]:
        centralities = all_centralities[latest_year][layer_name]['centralities']
        all_countries_corr.update(centralities['pagerank'].keys())

centrality_comparison = []

for country in all_countries_corr:
    row = {'Country': country}
    
    for layer_name in available_layers:
        if layer_name in all_centralities[latest_year]:
            centralities = all_centralities[latest_year][layer_name]['centralities']
            row[LAYER_NAMES[layer_name]] = centralities['pagerank'].get(country, 0)
    
    centrality_comparison.append(row)

comp_df = pd.DataFrame(centrality_comparison)

corr_columns = [LAYER_NAMES[l] for l in available_layers if LAYER_NAMES[l] in comp_df.columns]
if len(corr_columns) >= 2:
    corr_matrix = comp_df[corr_columns].corr(method='spearman')
    
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

if 2000 in ALL_YEARS and 2020 in ALL_YEARS:
    rank_2000 = sorted(multilayer_centrality[2000].items(),
                      key=lambda x: x[1], reverse=True)
    rank_2020 = sorted(multilayer_centrality[2020].items(),
                      key=lambda x: x[1], reverse=True)
    
    top20_2000 = set([c for c, _ in rank_2000[:20]])
    top20_2020 = set([c for c, _ in rank_2020[:20]])
    top20_union = top20_2000.union(top20_2020)
    
    rank_changes = []
    
    for country in top20_union:
        rank_2000_val = next((i+1 for i, (c, _) in enumerate(rank_2000) 
                             if c == country), None)
        rank_2020_val = next((i+1 for i, (c, _) in enumerate(rank_2020) 
                             if c == country), None)
        
        if rank_2000_val and rank_2020_val:
            change = rank_2000_val - rank_2020_val
            rank_changes.append({
                'Country': country,
                'Rank_2000': rank_2000_val,
                'Rank_2020': rank_2020_val,
                'Change': change
            })
    
    rank_changes_df = pd.DataFrame(rank_changes)
    rank_changes_df = rank_changes_df.sort_values('Change', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['green' if c > 0 else 'red' if c < 0 else 'gray' 
              for c in rank_changes_df['Change']]
    
    ax.barh(range(len(rank_changes_df)), rank_changes_df['Change'], 
            color=colors, alpha=0.7)
    ax.set_yticks(range(len(rank_changes_df)))
    ax.set_yticklabels(rank_changes_df['Country'], fontsize=9)
    ax.set_xlabel('Rank Change (Positive = Improved)', fontsize=12, fontweight='bold')
    ax.set_title('Centrality Rank Changes: 2000 vs 2020', 
                fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    fig_path = OUTPUT_DIRS['figures'] / 'fig12_rank_changes_2000_vs_2020.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ {fig_path}")
    plt.close()

# =====================================================================
# 9. 正解データ（Ground Truth）の作成
# =====================================================================
print("\n[9] 正解データ（2020年移住Top-20）の作成...")

migration_2020 = migration_df[migration_df['year'] == TEST_YEAR].copy()
inflow_2020 = migration_2020.groupby('destination')['migrant_stock'].sum().sort_values(ascending=False)
actual_top20 = inflow_2020.nlargest(TOP_K).index.tolist()

print(f"  正解Top-{TOP_K}国:")
for i, country in enumerate(actual_top20, 1):
    print(f"    {i:2d}. {country}: {inflow_2020[country]:,}")

# =====================================================================
# 10. Top-k Accuracy評価（2つの重み付け手法で比較）
# =====================================================================
print("\n[10] Top-k Accuracy評価（2つの重み付け手法）...")

results = []

for weight_method in ['uniform', 'normalized']:
    print(f"\n  === 重み付け手法: {weight_method.upper()} ===")
    
    for combo in LAYER_COMBINATIONS:
        combo_name = combo['name']
        combo_layers = combo['layers']
        combo_code = combo['code']
        
        print(f"\n  処理中: {combo_name} ({combo_code})...")
        
        # ネットワーク構築
        G, stats = build_multilayer_network(
            layer_data, combo_layers, 
            years=TRAIN_YEARS, 
            weight_method=weight_method
        )
        
        if G.number_of_nodes() == 0:
            print(f"    ⚠ 警告: ネットワークが空です")
            continue
        
        print(f"    ノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}")
        print(f"    総重み: {stats['total_weight']:.6f}")
        
        # レイヤー寄与度を表示
        for layer_name, contrib in stats['layer_contributions'].items():
            contrib_pct = (contrib['total_weight'] / stats['total_weight'] * 100) if stats['total_weight'] > 0 else 0
            print(f"      {LAYER_NAMES[layer_name]:12s}: {contrib_pct:5.1f}% (重み={contrib['total_weight']:.6f})")
        
        # 中心性計算と評価
        centralities = calculate_centralities(G, weighted=True)
        
        for cent_type, cent_name in [('in_degree', 'degree'), ('betweenness', 'betweenness')]:
            if cent_type not in centralities:
                continue
            
            centrality_series = pd.Series(centralities[cent_type]).sort_values(ascending=False)
            predicted_top20 = centrality_series.nlargest(TOP_K).index.tolist()
            
            matches = set(predicted_top20) & set(actual_top20)
            accuracy = len(matches) / TOP_K
            
            results.append({
                'weight_method': weight_method,
                'combination': combo_name,
                'code': combo_code,
                'layers': '+'.join(combo_layers),
                'n_layers': len(combo_layers),
                'centrality_type': cent_name,
                'accuracy': accuracy,
                'matches': len(matches),
                'predicted_top20': predicted_top20,
                'matched_countries': sorted(list(matches))
            })
            
            print(f"    {cent_name.capitalize()}: {accuracy:.3f} ({len(matches)}/{TOP_K})")

# 結果をDataFrameに変換
results_df = pd.DataFrame(results)

# =====================================================================
# 11. Top-k結果の保存
# =====================================================================
print("\n[11] Top-k結果の保存...")

# 詳細テーブル保存
output_table = OUTPUT_DIRS['tables'] / 'topk_accuracy_results_comparison.csv'
results_df.to_csv(output_table, index=False, encoding='utf-8')
print(f"  ✓ 詳細テーブル保存: {output_table}")

# 各重み付け手法のサマリー
for weight_method in ['uniform', 'normalized']:
    subset = results_df[results_df['weight_method'] == weight_method]
    
    summary_df = subset.pivot(
        index='combination',
        columns='centrality_type',
        values='accuracy'
    ).round(3)
    
    summary_output = OUTPUT_DIRS['tables'] / f'topk_accuracy_summary_{weight_method}.csv'
    summary_df.to_csv(summary_output, encoding='utf-8')
    print(f"  ✓ サマリー保存 ({weight_method}): {summary_output}")
    
    print(f"\n  【{weight_method.upper()}】")
    print(summary_df.to_string())

# =====================================================================
# 12. Top-k可視化（比較版）
# =====================================================================
print("\n[12] Top-k可視化の生成（2手法比較）...")

# ----- 図13: 精度比較（2手法並列） -----
print("  [図13] 精度比較（2手法並列）...")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

for idx, weight_method in enumerate(['uniform', 'normalized']):
    ax = axes[idx]
    subset = results_df[results_df['weight_method'] == weight_method]
    
    x = np.arange(len(LAYER_COMBINATIONS))
    width = 0.35
    
    degree_acc = subset[subset['centrality_type'] == 'degree']['accuracy'].values
    betweenness_acc = subset[subset['centrality_type'] == 'betweenness']['accuracy'].values
    
    bars1 = ax.bar(x - width/2, degree_acc, width, label='Degree Centrality', 
                   alpha=0.8, color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, betweenness_acc, width, label='Betweenness Centrality', 
                   alpha=0.8, color='coral', edgecolor='black')
    
    # 値をバーの上に表示
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel('Layer Combination', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Top-{TOP_K} Accuracy', fontsize=11, fontweight='bold')
    ax.set_title(f'Weight Method: {weight_method.upper()}', 
                fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c['code'] for c in LAYER_COMBINATIONS], rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)

plt.suptitle(f'Top-{TOP_K} Accuracy Comparison: Uniform vs Normalized Weighting', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig13_path = OUTPUT_DIRS['figures'] / 'fig13_topk_accuracy_comparison_methods.png'
plt.savefig(fig13_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig13_path}")
plt.close()

# ----- 図14: レイヤー数と精度の関係（2手法比較） -----
print("  [図14] レイヤー数と精度の関係（2手法比較）...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, cent_type in enumerate(['degree', 'betweenness']):
    ax = axes[idx]
    
    for weight_method in ['uniform', 'normalized']:
        subset = results_df[
            (results_df['weight_method'] == weight_method) &
            (results_df['centrality_type'] == cent_type)
        ]
        
        ax.plot(subset['n_layers'], subset['accuracy'], 
                marker='o', linewidth=2.5, markersize=10, 
                label=f'{weight_method.capitalize()}', alpha=0.8)
    
    ax.set_xlabel('Number of Layers', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Top-{TOP_K} Accuracy', fontsize=11, fontweight='bold')
    ax.set_title(f'{cent_type.capitalize()} Centrality', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0, 1)

plt.suptitle('Accuracy vs Number of Layers: Uniform vs Normalized', 
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
fig14_path = OUTPUT_DIRS['figures'] / 'fig14_accuracy_vs_layers_comparison.png'
plt.savefig(fig14_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig14_path}")
plt.close()

# ----- 図15: 予測成功マトリクス（Normalized, Degree） -----
print("  [図15] 予測成功マトリクス（Normalized）...")

fig, ax = plt.subplots(figsize=(16, 10))

all_countries_hm = sorted(set(actual_top20))
heatmap_data = np.zeros((len(LAYER_COMBINATIONS), len(all_countries_hm)))

for i, combo in enumerate(LAYER_COMBINATIONS):
    combo_results = results_df[
        (results_df['code'] == combo['code']) & 
        (results_df['centrality_type'] == 'degree') &
        (results_df['weight_method'] == 'normalized')
    ]
    
    if len(combo_results) > 0:
        predicted = combo_results.iloc[0]['predicted_top20']
        for j, country in enumerate(all_countries_hm):
            if country in predicted:
                heatmap_data[i, j] = 1

sns.heatmap(heatmap_data, 
            xticklabels=all_countries_hm,
            yticklabels=[c['code'] for c in LAYER_COMBINATIONS],
            cmap='RdYlGn', 
            cbar_kws={'label': 'Predicted (1) or Not (0)'},
            linewidths=0.5,
            ax=ax)

ax.set_title(f'Top-{TOP_K} Prediction Success Matrix (Normalized, Degree)', 
            fontsize=14, fontweight='bold')
ax.set_xlabel('Country', fontsize=12, fontweight='bold')
ax.set_ylabel('Layer Combination', fontsize=12, fontweight='bold')

plt.tight_layout()
fig15_path = OUTPUT_DIRS['figures'] / 'fig15_prediction_heatmap_normalized.png'
plt.savefig(fig15_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig15_path}")
plt.close()

# =====================================================================
# 13. 統合レポートの生成
# =====================================================================
print("\n[13] 統合レポートの生成...")

report_lines = []
report_lines.append("="*70)
report_lines.append(" 統合版：マルチレイヤーネットワーク分析（データ量正規化版） ")
report_lines.append("="*70)

report_lines.append(f"\n実行日時: {pd.Timestamp.now()}")

# セクション1: データ概要
report_lines.append("\n" + "="*70)
report_lines.append(" 1. データ概要 ")
report_lines.append("="*70)

for layer_name, layer_df in layer_data.items():
    if layer_df is not None:
        report_lines.append(f"\n{LAYER_NAMES[layer_name]}:")
        report_lines.append(f"  総行数: {len(layer_df):,}")
        report_lines.append(f"  年範囲: {sorted(layer_df['year'].unique())}")

# セクション2: データ量統計
report_lines.append("\n" + "="*70)
report_lines.append(" 2. データ量統計（訓練データ2000-2015） ")
report_lines.append("="*70)

report_lines.append("\n各レイヤーの総重み:")
for layer_name, total in layer_total_weights.items():
    report_lines.append(f"  {LAYER_NAMES[layer_name]:12s}: {total:>20,.2f}")

report_lines.append("\n正規化係数:")
for layer_name, total in layer_total_weights.items():
    if total > 0:
        norm_factor = 1.0 / total
        report_lines.append(f"  {LAYER_NAMES[layer_name]:12s}: {norm_factor:.2e}")

# セクション3: Top-k Accuracy検証
report_lines.append("\n" + "="*70)
report_lines.append(" 3. Top-k Accuracy検証結果 ")
report_lines.append("="*70)

report_lines.append(f"\n設定:")
report_lines.append(f"  訓練データ: {TRAIN_YEARS}")
report_lines.append(f"  テストデータ: {TEST_YEAR}")
report_lines.append(f"  Top-K: {TOP_K}")

report_lines.append(f"\n正解データ（{TEST_YEAR}年移住流入Top-{TOP_K}）:")
for i, country in enumerate(actual_top20, 1):
    report_lines.append(f"  {i:2d}. {country}: {inflow_2020[country]:,}")

# 2つの手法の結果
for weight_method in ['uniform', 'normalized']:
    report_lines.append(f"\n--- 重み付け手法: {weight_method.upper()} ---")
    
    subset = results_df[results_df['weight_method'] == weight_method]
    summary_df = subset.pivot(
        index='combination',
        columns='centrality_type',
        values='accuracy'
    ).round(3)
    
    report_lines.append("\n" + summary_df.to_string())
    
    # 仮説検証
    single_layer = subset[subset['n_layers'] == 1]['accuracy'].mean()
    multi_layer = subset[subset['n_layers'] > 1]['accuracy'].mean()
    
    report_lines.append(f"\n単層 vs 多層:")
    report_lines.append(f"  単層平均: {single_layer:.3f}")
    report_lines.append(f"  多層平均: {multi_layer:.3f}")
    report_lines.append(f"  差分: {multi_layer - single_layer:+.3f}")
    
    if multi_layer > single_layer:
        report_lines.append(f"  → 仮説支持: 多層ネットワークの方が精度が高い ✓")
    else:
        report_lines.append(f"  → 仮説非支持: 単層ネットワークの方が精度が高い ✗")

# セクション4: 比較分析
report_lines.append("\n" + "="*70)
report_lines.append(" 4. Uniform vs Normalized 比較 ")
report_lines.append("="*70)

comparison_summary = results_df.groupby(['combination', 'weight_method', 'centrality_type'])['accuracy'].mean().unstack(level=[1,2])
report_lines.append("\n" + comparison_summary.to_string())

# セクション5: 出力ファイル一覧
report_lines.append("\n" + "="*70)
report_lines.append(" 5. 出力ファイル一覧 ")
report_lines.append("="*70)

report_lines.append("\n【既存の可視化】")
report_lines.append("  図7-12: レイヤー別分析、時系列、相関など")

report_lines.append("\n【Top-k Accuracy検証（2手法比較）】")
report_lines.append("  図13: 精度比較（Uniform vs Normalized並列）")
report_lines.append("  図14: レイヤー数と精度の関係（2手法比較）")
report_lines.append("  図15: 予測成功マトリクス（Normalized）")

report_lines.append("\n【テーブル】")
report_lines.append("  topk_accuracy_results_comparison.csv（全結果）")
report_lines.append("  topk_accuracy_summary_uniform.csv")
report_lines.append("  topk_accuracy_summary_normalized.csv")

report_text = "\n".join(report_lines)

# レポート保存
report_path = OUTPUT_DIRS['reports'] / 'integrated_analysis_report_normalized.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"  ✓ レポート保存: {report_path}")

print("\n" + report_text)

print("\n" + "="*70)
print(" ✓ すべての処理が完了しました ")
print("="*70)
print(f"\n生成された図:")
print(f"  既存: 図7-12 (6枚)")
print(f"  新規: 図13-15 (3枚、2手法比較版)")
print(f"  合計: 9枚の可視化")
print(f"\n重み付け手法:")
print(f"  - Uniform: 均等重み (1/N)")
print(f"  - Normalized: データ量正規化 (各レイヤーの総重みで正規化)")