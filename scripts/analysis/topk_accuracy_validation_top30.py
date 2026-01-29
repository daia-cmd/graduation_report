#!/usr/bin/env python3
"""
統合版：マルチレイヤーネットワーク中心性分析 + Top-k Accuracy検証
【移住データ中心版 - Top-30】

このスクリプトは以下を実行：
- Top-20とTop-30の両方で評価
- 移住中心の重み付け（4つの比率）
- 難易度が上がることで多層ネットワークの優位性が明確化

使用方法：
  python topk_accuracy_validation_top30.py
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
print(" マルチレイヤーネットワーク分析（移住中心版 - Top-20/30比較） ")
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

# Top-kの設定（両方評価）
TOP_K_VALUES = [20, 30]

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

# Top-k検証用のレイヤー組み合わせ
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
MIGRATION_CENTRIC_RATIOS = [
    {'name': '10:1', 'migration': 1.0, 'others': 0.1},
    {'name': '5:1', 'migration': 1.0, 'others': 0.2},
    {'name': '3:1', 'migration': 1.0, 'others': 0.33},
    {'name': '2:1', 'migration': 1.0, 'others': 0.5},
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

migration_df = load_layer_data(MIGRATION_FILE, 'Migration')
diplomacy_df = load_layer_data(DIPLOMACY_FILE, 'Diplomacy')
aviation_df = load_layer_data(AVIATION_FILE, 'Aviation')
trade_df = load_layer_data(TRADE_FILE, 'Trade')

# カラム名の統一
if diplomacy_df is not None and 'embassy_level' in diplomacy_df.columns:
    diplomacy_df = diplomacy_df.rename(columns={'embassy_level': 'diplomatic_relation'})

if aviation_df is not None and 'route_count' in aviation_df.columns:
    aviation_df = aviation_df.rename(columns={'route_count': 'aviation_routes'})

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
# 3. ネットワーク構築関数
# =====================================================================
print("\n[3] ネットワーク構築関数定義...")

def build_multilayer_network(layer_dfs, layer_names, years=None, weight_method='uniform', 
                            migration_weight=1.0, others_weight=0.1):
    """多層ネットワークを構築"""
    G = nx.DiGraph()
    n_layers = len(layer_names)
    
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
        
        if years is not None:
            df = df[df['year'].isin(years)].copy()
        
        col = LAYER_COLUMNS[layer_name]
        
        layer_edges = 0
        layer_weight = 0
        
        for _, row in df.iterrows():
            origin = row['origin']
            dest = row['destination']
            value = row[col]
            
            if pd.isna(value) or value <= 0:
                continue
            
            # 重み付け方法に応じて重みを計算
            if weight_method == 'migration_centric':
                if layer_name == 'migration':
                    layer_multiplier = migration_weight
                else:
                    layer_multiplier = others_weight
                
                total = layer_total_weights.get(layer_name, 1)
                if total > 0:
                    edge_weight = (value / total) * layer_multiplier
                else:
                    edge_weight = 0
            else:
                edge_weight = value / n_layers
            
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
    
    # 次数中心性
    if weighted:
        centralities['in_degree'] = dict(G.in_degree(weight='weight'))
    else:
        centralities['in_degree'] = dict(G.in_degree())
    
    return centralities

# =====================================================================
# 4. 正解データ（Ground Truth）の作成
# =====================================================================
print("\n[4] 正解データ（2020年移住Top-20/30）の作成...")

migration_2020 = migration_df[migration_df['year'] == TEST_YEAR].copy()
inflow_2020 = migration_2020.groupby('destination')['migrant_stock'].sum().sort_values(ascending=False)

# Top-20とTop-30の正解データ
actual_top = {}
for k in TOP_K_VALUES:
    actual_top[k] = inflow_2020.nlargest(k).index.tolist()
    print(f"\n  正解Top-{k}国:")
    for i, country in enumerate(actual_top[k], 1):
        print(f"    {i:2d}. {country}: {inflow_2020[country]:,}")

# =====================================================================
# 5. Top-k Accuracy評価（Top-20とTop-30の両方）
# =====================================================================
print("\n[5] Top-k Accuracy評価（Top-20 & Top-30）...")

results = []

for ratio_config in MIGRATION_CENTRIC_RATIOS:
    ratio_name = ratio_config['name']
    migration_w = ratio_config['migration']
    others_w = ratio_config['others']
    
    print(f"\n  === 移住中心 {ratio_name} ===")
    
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
        
        # 中心性計算
        centralities = calculate_centralities(G, weighted=True)
        
        if 'in_degree' not in centralities:
            continue
        
        centrality_series = pd.Series(centralities['in_degree']).sort_values(ascending=False)
        
        # Top-20とTop-30の両方で評価
        for k in TOP_K_VALUES:
            predicted_topk = centrality_series.nlargest(k).index.tolist()
            
            matches = set(predicted_topk) & set(actual_top[k])
            accuracy = len(matches) / k
            
            results.append({
                'top_k': k,
                'ratio': ratio_name,
                'combination': combo_name,
                'code': combo_code,
                'layers': '+'.join(combo_layers),
                'n_layers': len(combo_layers),
                'accuracy': accuracy,
                'matches': len(matches),
                'predicted': predicted_topk,
                'matched_countries': sorted(list(matches))
            })
            
            print(f"    Top-{k}: {accuracy:.3f} ({len(matches)}/{k})")

# 結果をDataFrameに変換
results_df = pd.DataFrame(results)

# =====================================================================
# 6. 結果の保存
# =====================================================================
print("\n[6] 結果の保存...")

output_table = OUTPUT_DIRS['tables'] / 'topk_accuracy_results_top20_30_comparison.csv'
results_df.to_csv(output_table, index=False, encoding='utf-8')
print(f"  ✓ 詳細テーブル保存: {output_table}")

# =====================================================================
# 7. サマリー表示
# =====================================================================
print("\n" + "="*70)
print(" RESULTS SUMMARY BY TOP-K AND RATIO ")
print("="*70)

for k in TOP_K_VALUES:
    print(f"\n{'='*70}")
    print(f" TOP-{k} RESULTS ")
    print(f"{'='*70}")
    
    for ratio_config in MIGRATION_CENTRIC_RATIOS:
        ratio_name = ratio_config['name']
        subset = results_df[
            (results_df['top_k'] == k) &
            (results_df['ratio'] == ratio_name)
        ]
        
        if len(subset) == 0:
            continue
        
        summary_df = subset.pivot_table(
            index='combination',
            values='accuracy',
            aggfunc='mean'
        ).round(3)
        
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

# =====================================================================
# 8. Top-20 vs Top-30 比較可視化
# =====================================================================
print("\n[8] Top-20 vs Top-30 比較可視化...")

# ----- 図18: Top-20 vs Top-30 比較（4比率） -----
print("  [図18] Top-20 vs Top-30 比較（4比率）...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for idx, ratio_config in enumerate(MIGRATION_CENTRIC_RATIOS):
    ax = axes[idx]
    ratio_name = ratio_config['name']
    
    x = np.arange(len(LAYER_COMBINATIONS))
    width = 0.35
    
    # Top-20
    subset_20 = results_df[
        (results_df['top_k'] == 20) &
        (results_df['ratio'] == ratio_name)
    ]
    acc_20 = subset_20['accuracy'].values
    
    # Top-30
    subset_30 = results_df[
        (results_df['top_k'] == 30) &
        (results_df['ratio'] == ratio_name)
    ]
    acc_30 = subset_30['accuracy'].values
    
    bars1 = ax.bar(x - width/2, acc_20, width, label='Top-20', 
                   alpha=0.8, color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, acc_30, width, label='Top-30', 
                   alpha=0.8, color='coral', edgecolor='black')
    
    # 値をバーの上に表示
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Layer Combination', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title(f'Migration-Centric {ratio_name}', 
                fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c['code'] for c in LAYER_COMBINATIONS], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.90, color='green', linestyle='--', linewidth=1, alpha=0.5)

plt.suptitle('Top-20 vs Top-30 Accuracy Comparison', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
fig18_path = OUTPUT_DIRS['figures'] / 'fig18_top20_vs_top30_comparison.png'
plt.savefig(fig18_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig18_path}")
plt.close()

# ----- 図19: 精度差分（Top-30 - Top-20） -----
print("  [図19] 精度差分（Top-30 - Top-20）...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for idx, ratio_config in enumerate(MIGRATION_CENTRIC_RATIOS):
    ax = axes[idx]
    ratio_name = ratio_config['name']
    
    x = np.arange(len(LAYER_COMBINATIONS))
    
    # Top-20
    subset_20 = results_df[
        (results_df['top_k'] == 20) &
        (results_df['ratio'] == ratio_name)
    ]
    acc_20 = subset_20['accuracy'].values
    
    # Top-30
    subset_30 = results_df[
        (results_df['top_k'] == 30) &
        (results_df['ratio'] == ratio_name)
    ]
    acc_30 = subset_30['accuracy'].values
    
    # 差分
    diff = acc_30 - acc_20
    
    colors = ['green' if d > 0 else 'red' if d < 0 else 'gray' for d in diff]
    
    bars = ax.bar(x, diff, alpha=0.8, color=colors, edgecolor='black')
    
    # 値をバーの上/下に表示
    for bar, d in zip(bars, diff):
        height = bar.get_height()
        va = 'bottom' if d >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{d:+.2f}',
                ha='center', va=va, fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Layer Combination', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy Difference (Top-30 - Top-20)', fontsize=11, fontweight='bold')
    ax.set_title(f'Migration-Centric {ratio_name}', 
                fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c['code'] for c in LAYER_COMBINATIONS], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Accuracy Difference: Top-30 vs Top-20 (Positive = Top-30 Better)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
fig19_path = OUTPUT_DIRS['figures'] / 'fig19_accuracy_difference.png'
plt.savefig(fig19_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig19_path}")
plt.close()

# ----- 図20: 最良比率（5:1）でのTop-20 vs Top-30詳細比較 -----
print("  [図20] 最良比率（5:1）でのTop-20 vs Top-30詳細比較...")

fig, ax = plt.subplots(figsize=(14, 8))

ratio_name = '5:1'

x = np.arange(len(LAYER_COMBINATIONS))
width = 0.35

subset_20 = results_df[
    (results_df['top_k'] == 20) &
    (results_df['ratio'] == ratio_name)
]
acc_20 = subset_20['accuracy'].values

subset_30 = results_df[
    (results_df['top_k'] == 30) &
    (results_df['ratio'] == ratio_name)
]
acc_30 = subset_30['accuracy'].values

bars1 = ax.bar(x - width/2, acc_20, width, label='Top-20', 
               alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, acc_30, width, label='Top-30', 
               alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)

# 値をバーの上に表示
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Layer Combination', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax.set_title(f'Top-20 vs Top-30 Accuracy: Migration-Centric 5:1 (Best Ratio)', 
            fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([c['code'] for c in LAYER_COMBINATIONS], rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=12, loc='lower left')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1)
ax.axhline(y=0.90, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Baseline (M only, Top-20)')

plt.tight_layout()
fig20_path = OUTPUT_DIRS['figures'] / 'fig20_best_ratio_detailed_comparison.png'
plt.savefig(fig20_path, dpi=300, bbox_inches='tight')
print(f"    ✓ {fig20_path}")
plt.close()

# =====================================================================
# 9. 統合レポートの生成
# =====================================================================
print("\n[9] 統合レポートの生成...")

report_lines = []
report_lines.append("="*70)
report_lines.append(" Top-20 vs Top-30 比較分析レポート ")
report_lines.append("="*70)

report_lines.append(f"\n実行日時: {pd.Timestamp.now()}")

# 最良結果のまとめ
report_lines.append("\n" + "="*70)
report_lines.append(" 最良結果 ")
report_lines.append("="*70)

for k in TOP_K_VALUES:
    best = results_df[results_df['top_k'] == k].loc[results_df[results_df['top_k'] == k]['accuracy'].idxmax()]
    report_lines.append(f"\nTop-{k} 最高精度:")
    report_lines.append(f"  組み合わせ: {best['combination']}")
    report_lines.append(f"  比率: {best['ratio']}")
    report_lines.append(f"  精度: {best['accuracy']:.3f} ({best['matches']}/{k})")

# 比較分析
report_lines.append("\n" + "="*70)
report_lines.append(" Top-20 vs Top-30 比較 ")
report_lines.append("="*70)

for ratio_config in MIGRATION_CENTRIC_RATIOS:
    ratio_name = ratio_config['name']
    
    report_lines.append(f"\n【比率 {ratio_name}】")
    
    for combo in LAYER_COMBINATIONS:
        combo_code = combo['code']
        
        acc_20 = results_df[
            (results_df['top_k'] == 20) &
            (results_df['ratio'] == ratio_name) &
            (results_df['code'] == combo_code)
        ]['accuracy'].values
        
        acc_30 = results_df[
            (results_df['top_k'] == 30) &
            (results_df['ratio'] == ratio_name) &
            (results_df['code'] == combo_code)
        ]['accuracy'].values
        
        if len(acc_20) > 0 and len(acc_30) > 0:
            diff = acc_30[0] - acc_20[0]
            report_lines.append(f"  {combo_code:6s}: Top-20={acc_20[0]:.3f}, Top-30={acc_30[0]:.3f}, Diff={diff:+.3f}")

report_text = "\n".join(report_lines)

# レポート保存
report_path = OUTPUT_DIRS['reports'] / 'top20_vs_top30_comparison_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"  ✓ レポート保存: {report_path}")

print("\n" + report_text)

print("\n" + "="*70)
print(" ✓ すべての処理が完了しました ")
print("="*70)
print(f"\n生成された図:")
print(f"  図18: Top-20 vs Top-30 比較（4比率）")
print(f"  図19: 精度差分（Top-30 - Top-20）")
print(f"  図20: 最良比率（5:1）での詳細比較")