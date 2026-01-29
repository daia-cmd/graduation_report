#!/usr/bin/env python3
"""
統合版：マルチレイヤーネットワーク中心性分析 + Top-k Accuracy検証
【移住データ中心の重み付け版】

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
- Uniform: 均等重み (1/N)
- Normalized: データ量正規化
- Migration-Centric: 移住中心（移住=1.0、他=可変）

使用方法：
  python topk_accuracy_validation_migration_centric.py
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
print(" 統合版：マルチレイヤーネットワーク分析（移住データ中心版） ")
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

# 移住中心の重み付け設定
# 複数の比率を試す
MIGRATION_CENTRIC_RATIOS = [
    {'name': '10:1', 'migration': 1.0, 'others': 0.1},   # 他は移住の1/10
    {'name': '5:1', 'migration': 1.0, 'others': 0.2},    # 他は移住の1/5
    {'name': '3:1', 'migration': 1.0, 'others': 0.33},   # 他は移住の1/3
    {'name': '2:1', 'migration': 1.0, 'others': 0.5},    # 他は移住の1/2
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
# 2. レイヤーごとの総重み計算
# =====================================================================
print("\n[2] レイヤーごとの総重み計算...")

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

# =====================================================================
# 3. ネットワーク構築関数（移住中心の重み付け対応）
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

def build_multilayer_network(layer_dfs, layer_names, years=None, weight_method='uniform', 
                            migration_weight=1.0, others_weight=0.1):
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
        - 'migration_centric': 移住中心
    migration_weight : float
        移住レイヤーの重み（migration_centricの場合）
    others_weight : float
        他レイヤーの重み（migration_centricの場合）
    
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
        'total_weight': 0,
        'weight_method': weight_method
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
                    
            elif weight_method == 'migration_centric':
                # 移住中心: 移住は migration_weight、他は others_weight
                if layer_name == 'migration':
                    layer_multiplier = migration_weight
                else:
                    layer_multiplier = others_weight
                
                # 総重みで正規化してから倍率をかける
                total = layer_total_weights.get(layer_name, 1)
                if total > 0:
                    edge_weight = (value / total) * layer_multiplier
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
# 8. 既存の可視化（図7-12）- スキップ（既に生成済み）
# =====================================================================
print("\n[8] 既存の可視化（スキップ - 既に生成済み）")

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
# 10. Top-k Accuracy評価（移住中心の複数比率で評価）
# =====================================================================
print("\n[10] Top-k Accuracy評価（移住中心の重み付け）...")

results = []

# Uniform, Normalized（参照用）
for weight_method in ['uniform', 'normalized']:
    print(f"\n  === 重み付け手法: {weight_method.upper()} ===")
    
    for combo in LAYER_COMBINATIONS:
        combo_name = combo['name']
        combo_layers = combo['layers']
        combo_code = combo['code']
        
        # ネットワーク構築
        G, stats = build_multilayer_network(
            layer_data, combo_layers, 
            years=TRAIN_YEARS, 
            weight_method=weight_method
        )
        
        if G.number_of_nodes() == 0:
            continue
        
        # 中心性計算と評価（Degreeのみ）
        centralities = calculate_centralities(G, weighted=True)
        
        cent_type = 'in_degree'
        cent_name = 'degree'
        
        if cent_type in centralities:
            centrality_series = pd.Series(centralities[cent_type]).sort_values(ascending=False)
            predicted_top20 = centrality_series.nlargest(TOP_K).index.tolist()
            
            matches = set(predicted_top20) & set(actual_top20)
            accuracy = len(matches) / TOP_K
            
            results.append({
                'weight_method': weight_method,
                'ratio': 'N/A',
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

# Migration-Centric（複数比率）
for ratio_config in MIGRATION_CENTRIC_RATIOS:
    ratio_name = ratio_config['name']
    migration_w = ratio_config['migration']
    others_w = ratio_config['others']
    
    print(f"\n  === 移住中心 {ratio_name} (移住={migration_w}, 他={others_w}) ===")
    
    for combo in LAYER_COMBINATIONS:
        combo_name = combo['name']
        combo_layers = combo['layers']
        combo_code = combo['code']
        
        print(f"\n  処理中: {combo_name} ({combo_code})...")
        
        # ネットワーク構築
        G, stats = build_multilayer_network(
            layer_data, combo_layers, 
            years=TRAIN_YEARS, 
            weight_method='migration_centric',
            migration_weight=migration_w,
            others_weight=others_w
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
        
        # 中心性計算と評価（Degreeのみ）
        centralities = calculate_centralities(G, weighted=True)
        
        cent_type = 'in_degree'
        cent_name = 'degree'
        
        if cent_type in centralities:
            centrality_series = pd.Series(centralities[cent_type]).sort_values(ascending=False)
            predicted_top20 = centrality_series.nlargest(TOP_K).index.tolist()
            
            matches = set(predicted_top20) & set(actual_top20)
            accuracy = len(matches) / TOP_K
            
            results.append({
                'weight_method': 'migration_centric',
                'ratio': ratio_name,
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
            
            print(f"    Degree: {accuracy:.3f} ({len(matches)}/{TOP_K})")

# 結果をDataFrameに変換
results_df = pd.DataFrame(results)

# =====================================================================
# 11. Top-k結果の保存
# =====================================================================
print("\n[11] Top-k結果の保存...")

# 詳細テーブル保存
output_table = OUTPUT_DIRS['tables'] / 'topk_accuracy_results_migration_centric.csv'
results_df.to_csv(output_table, index=False, encoding='utf-8')
print(f"  ✓ 詳細テーブル保存: {output_table}")

# 各重み付け手法のサマリー
print("\n" + "="*70)
print(" RESULTS SUMMARY BY WEIGHT METHOD ")
print("="*70)

for weight_method in ['uniform', 'normalized'] + ['migration_centric']:
    if weight_method == 'migration_centric':
        # 各比率ごとに表示
        for ratio_config in MIGRATION_CENTRIC_RATIOS:
            ratio_name = ratio_config['name']
            subset = results_df[
                (results_df['weight_method'] == weight_method) &
                (results_df['ratio'] == ratio_name)
            ]
            
            if len(subset) == 0:
                continue
            
            summary_df = subset.pivot_table(
                index='combination',
                values='accuracy',
                aggfunc='mean'
            ).round(3)
            
            summary_output = OUTPUT_DIRS['tables'] / f'topk_accuracy_summary_migration_{ratio_name.replace(":", "_")}.csv'
            summary_df.to_csv(summary_output, encoding='utf-8')
            
            print(f"\n【移住中心 {ratio_name}】")
            print(summary_df.to_string())
            
            # 仮説検証
            single_layer = subset[subset['n_layers'] == 1]['accuracy'].mean()
            multi_layer = subset[subset['n_layers'] > 1]['accuracy'].mean()
            
            print(f"\n単層 vs 多層:")
            print(f"  単層平均: {single_layer:.3f}")
            print(f"  多層平均: {multi_layer:.3f}")
            print(f"  差分: {multi_layer - single_layer:+.3f}")
            
            if multi_layer > single_layer:
                print(f"  → 仮説支持: 多層ネットワークの方が精度が高い ✓")
            else:
                print(f"  → 仮説非支持: 単層ネットワークの方が精度が高い ✗")
    else:
        subset = results_df[results_df['weight_method'] == weight_method]
        
        if len(subset) == 0:
            continue
        
        summary_df = subset.pivot_table(
            index='combination',
            values='accuracy',
            aggfunc='mean'
        ).round(3)
        
        print(f"\n【{weight_method.upper()}】")
        print(summary_df.to_string())

# =====================================================================
# 12. Top-k可視化（移住中心版）
# =====================================================================
print("\n[12] Top-k可視化の生成（移住中心版）...")

# ----- 図16: 移住中心の重み付け比較 -----
print("  [図16] 移住中心の重み付け比較...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for idx, ratio_config in enumerate(MIGRATION_CENTRIC_RATIOS):
    ax = axes[idx]
    ratio_name = ratio_config['name']
    
    subset = results_df[
        (results_df['weight_method'] == 'migration_centric') &
        (results_df['ratio'] == ratio_name)
    ]
    
    if len(subset) == 0:
        continue
    
    x = np.arange(len(LAYER_COMBINATIONS))
    width = 0.6
    
    accuracies = subset['accuracy'].values
    
    colors = ['green' if acc >= 0.85 else 'orange' if acc >= 0.75 else 'red' 
              for acc in accuracies]
    
    bars = ax.bar(x, accuracies, width, alpha=0.8, color=colors, edgecolor='black')
    
    # 値をバーの上に表示
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Layer Combination', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Top-{TOP_K} Accuracy', fontsize=11, fontweight='bold')
    ax.set_title(f'Migration-Centric {ratio_name} (M={ratio_config["migration"]}, Others={ratio_config["others"]})', 
                fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c['code'] for c in LAYER_COMBINATIONS], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.90, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (M only)')
    ax.legend()

plt.suptitle(f'Top-{TOP_K} Accuracy: Migration-Centric Weighting with Different Ratios', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
fig16_path = OUTPUT_DIRS['figures'] / 'fig16_migration_centric_comparison.png'
plt.savefig(fig16_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig16_path}")
plt.close()

# ----- 図17: 最良の比率での詳細比較 -----
print("  [図17] 全手法の比較...")

# 各手法の最良結果を抽出
comparison_data = []

for combo in LAYER_COMBINATIONS:
    combo_code = combo['code']
    combo_name = combo['name']
    
    # Uniform
    uniform_acc = results_df[
        (results_df['weight_method'] == 'uniform') &
        (results_df['code'] == combo_code)
    ]['accuracy'].values
    if len(uniform_acc) > 0:
        comparison_data.append({
            'combination': combo_name,
            'code': combo_code,
            'method': 'Uniform',
            'accuracy': uniform_acc[0]
        })
    
    # Normalized
    norm_acc = results_df[
        (results_df['weight_method'] == 'normalized') &
        (results_df['code'] == combo_code)
    ]['accuracy'].values
    if len(norm_acc) > 0:
        comparison_data.append({
            'combination': combo_name,
            'code': combo_code,
            'method': 'Normalized',
            'accuracy': norm_acc[0]
        })
    
    # Migration-Centric（各比率の最良）
    for ratio_config in MIGRATION_CENTRIC_RATIOS:
        ratio_name = ratio_config['name']
        mc_acc = results_df[
            (results_df['weight_method'] == 'migration_centric') &
            (results_df['ratio'] == ratio_name) &
            (results_df['code'] == combo_code)
        ]['accuracy'].values
        if len(mc_acc) > 0:
            comparison_data.append({
                'combination': combo_name,
                'code': combo_code,
                'method': f'MC {ratio_name}',
                'accuracy': mc_acc[0]
            })

comp_df = pd.DataFrame(comparison_data)

fig, ax = plt.subplots(figsize=(16, 8))

# 各組み合わせごとにグループ化して棒グラフ
methods = ['Uniform', 'Normalized'] + [f'MC {r["name"]}' for r in MIGRATION_CENTRIC_RATIOS]
n_methods = len(methods)
n_combos = len(LAYER_COMBINATIONS)

x = np.arange(n_combos)
width = 0.12

for i, method in enumerate(methods):
    method_data = comp_df[comp_df['method'] == method].sort_values('code')
    accuracies = [method_data[method_data['code'] == c['code']]['accuracy'].values[0] 
                 if len(method_data[method_data['code'] == c['code']]) > 0 else 0
                 for c in LAYER_COMBINATIONS]
    
    offset = (i - n_methods/2) * width + width/2
    ax.bar(x + offset, accuracies, width, label=method, alpha=0.8)

ax.set_xlabel('Layer Combination', fontsize=12, fontweight='bold')
ax.set_ylabel(f'Top-{TOP_K} Accuracy', fontsize=12, fontweight='bold')
ax.set_title(f'Comparison of All Weighting Methods', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([c['code'] for c in LAYER_COMBINATIONS], rotation=45, ha='right')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1)
ax.axhline(y=0.90, color='green', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
fig17_path = OUTPUT_DIRS['figures'] / 'fig17_all_methods_comparison.png'
plt.savefig(fig17_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig17_path}")
plt.close()

# =====================================================================
# 13. 統合レポートの生成
# =====================================================================
print("\n[13] 統合レポートの生成...")

report_lines = []
report_lines.append("="*70)
report_lines.append(" 統合版：マルチレイヤーネットワーク分析（移住データ中心版） ")
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

# セクション2: 重み付け設定
report_lines.append("\n" + "="*70)
report_lines.append(" 2. 移住中心の重み付け設定 ")
report_lines.append("="*70)

for ratio_config in MIGRATION_CENTRIC_RATIOS:
    report_lines.append(f"\n比率 {ratio_config['name']}:")
    report_lines.append(f"  移住レイヤー: {ratio_config['migration']}")
    report_lines.append(f"  他レイヤー: {ratio_config['others']}")
    report_lines.append(f"  比率: {ratio_config['migration']/ratio_config['others']:.1f}:1")

# セクション3: 結果サマリー
report_lines.append("\n" + "="*70)
report_lines.append(" 3. Top-k Accuracy検証結果 ")
report_lines.append("="*70)

report_lines.append(f"\n正解データ（{TEST_YEAR}年移住流入Top-{TOP_K}）:")
for i, country in enumerate(actual_top20, 1):
    report_lines.append(f"  {i:2d}. {country}: {inflow_2020[country]:,}")

# 各比率の結果
for ratio_config in MIGRATION_CENTRIC_RATIOS:
    ratio_name = ratio_config['name']
    subset = results_df[
        (results_df['weight_method'] == 'migration_centric') &
        (results_df['ratio'] == ratio_name)
    ]
    
    if len(subset) == 0:
        continue
    
    report_lines.append(f"\n--- 移住中心 {ratio_name} ---")
    
    summary_df = subset.pivot_table(
        index='combination',
        values='accuracy',
        aggfunc='mean'
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

# セクション4: 最良の結果
report_lines.append("\n" + "="*70)
report_lines.append(" 4. 最良の結果 ")
report_lines.append("="*70)

best_result = results_df.loc[results_df['accuracy'].idxmax()]
report_lines.append(f"\n最高精度:")
report_lines.append(f"  組み合わせ: {best_result['combination']}")
report_lines.append(f"  重み付け手法: {best_result['weight_method']}")
if best_result['weight_method'] == 'migration_centric':
    report_lines.append(f"  比率: {best_result['ratio']}")
report_lines.append(f"  精度: {best_result['accuracy']:.3f}")

# セクション5: 出力ファイル一覧
report_lines.append("\n" + "="*70)
report_lines.append(" 5. 出力ファイル一覧 ")
report_lines.append("="*70)

report_lines.append("\n【新規可視化】")
report_lines.append("  図16: 移住中心の重み付け比較（4比率）")
report_lines.append("  図17: 全手法の比較")

report_lines.append("\n【テーブル】")
report_lines.append("  topk_accuracy_results_migration_centric.csv（全結果）")
for ratio_config in MIGRATION_CENTRIC_RATIOS:
    ratio_name = ratio_config['name'].replace(":", "_")
    report_lines.append(f"  topk_accuracy_summary_migration_{ratio_name}.csv")

report_text = "\n".join(report_lines)

# レポート保存
report_path = OUTPUT_DIRS['reports'] / 'integrated_analysis_report_migration_centric.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"  ✓ レポート保存: {report_path}")

print("\n" + report_text)

print("\n" + "="*70)
print(" ✓ すべての処理が完了しました ")
print("="*70)
print(f"\n生成された図:")
print(f"  図16: 移住中心の重み付け比較（4比率）")
print(f"  図17: 全手法の比較")
print(f"\n評価した重み付け手法:")
print(f"  - Uniform（参照）")
print(f"  - Normalized（参照）")
print(f"  - Migration-Centric 10:1")
print(f"  - Migration-Centric 5:1")
print(f"  - Migration-Centric 3:1")
print(f"  - Migration-Centric 2:1")