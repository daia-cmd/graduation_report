import pandas as pd
import numpy as np
import os
import sys
from collections import defaultdict

# --- CONFIGURATION ---
DIPLOMATIC_FILE = 'diplomatic_network.csv'
AVIATION_FILE = 'aviation_network_raw.csv'
MIGRATION_FILE = 'data/processed/migration_network.csv'
TRADE_FILE = 'data/processed/trade_network.csv'
OUTPUT_FILE = 'multilayer_network.csv'
QUALITY_REPORT_FILE = 'multilayer_integration_quality_report.txt'

TARGET_YEARS = [2000, 2005, 2010, 2015, 2020]

# データ品質トラッキング
class IntegrationQualityTracker:
    def __init__(self):
        self.stats = {}
        self.warnings = []
        self.common_countries = {}
        
    def record_dataset_stats(self, name, stats):
        self.stats[name] = stats
    
    def add_warning(self, message):
        self.warnings.append(message)
    
    def set_common_countries(self, common):
        self.common_countries = common
    
    def generate_report(self, final_df):
        report = []
        report.append("="*70)
        report.append(" MULTILAYER NETWORK INTEGRATION QUALITY REPORT ")
        report.append("="*70)
        
        # 各データセットの統計
        report.append("\n" + "-"*70)
        report.append(" 入力データセットの統計 ")
        report.append("-"*70)
        
        for name, stats in self.stats.items():
            report.append(f"\n【{name}】")
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    report.append(f"  {key}: {value:,}")
                else:
                    report.append(f"  {key}: {value}")
        
        # 共通国の分析
        report.append("\n" + "-"*70)
        report.append(" 国カバレッジ分析 ")
        report.append("-"*70)
        
        for key, countries in self.common_countries.items():
            report.append(f"\n{key}: {len(countries)}カ国")
            if len(countries) <= 20:
                report.append(f"  {sorted(countries)}")
            else:
                sample = sorted(list(countries))[:10]
                report.append(f"  サンプル: {sample}...")
        
        # 統合後の統計
        report.append("\n" + "-"*70)
        report.append(" 統合後のデータ統計 ")
        report.append("-"*70)
        
        report.append(f"\n総行数: {len(final_df):,}")
        report.append(f"年範囲: {sorted(final_df['year'].unique())}")
        
        # レイヤーごとの統計
        for year in sorted(final_df['year'].unique()):
            df_year = final_df[final_df['year'] == year]
            report.append(f"\n【{year}年】")
            
            # 外交
            diplo = df_year[df_year['diplomatic_relation'].notna()]
            report.append(f"  外交関係: {len(diplo):,}行")
            
            # 航空
            avia = df_year[df_year['aviation_routes'].notna()]
            report.append(f"  航空路: {len(avia):,}行")
            
            # 移民
            migr = df_year[df_year['migrant_stock'].notna()]
            report.append(f"  移民: {len(migr):,}行")
            
            # 貿易
            trade = df_year[df_year['trade_value'].notna()]
            report.append(f"  貿易: {len(trade):,}行")
            
            # 全レイヤー
            all_layers = df_year[
                df_year['diplomatic_relation'].notna() &
                df_year['aviation_routes'].notna() &
                df_year['migrant_stock'].notna() &
                df_year['trade_value'].notna()
            ]
            report.append(f"  4レイヤー全て: {len(all_layers):,}行 ({len(all_layers)/len(df_year)*100:.1f}%)")
        
        # データ品質評価
        report.append("\n" + "-"*70)
        report.append(" データ品質評価 ")
        report.append("-"*70)
        
        report.append(self._evaluate_quality(final_df))
        
        # 警告
        if self.warnings:
            report.append("\n" + "-"*70)
            report.append(" 警告 ")
            report.append("-"*70)
            for warning in self.warnings:
                report.append(f"  ⚠ {warning}")
        
        return "\n".join(report)
    
    def _evaluate_quality(self, df):
        lines = []
        
        # チェック1: データ量
        if len(df) > 100000:
            lines.append("✓ 十分なデータ量（> 100,000行）")
        elif len(df) > 50000:
            lines.append("⚠ データ量が中程度（50,000-100,000行）")
        else:
            lines.append("✗ データ量が少ない（< 50,000行）")
        
        # チェック2: 欠損値率
        total_cells = len(df) * 4  # 4つの値列
        missing = df[['diplomatic_relation', 'aviation_routes', 'migrant_stock', 'trade_value']].isna().sum().sum()
        missing_rate = missing / total_cells * 100
        
        if missing_rate < 70:
            lines.append(f"✓ 欠損値率が低い（{missing_rate:.1f}%）")
        elif missing_rate < 85:
            lines.append(f"⚠ 欠損値率が中程度（{missing_rate:.1f}%）")
        else:
            lines.append(f"⚠ 欠損値率が高い（{missing_rate:.1f}%）- これは正常（レイヤーごとにデータが異なる）")
        
        # チェック3: 4レイヤー全てのカバレッジ
        all_four = df[
            df['diplomatic_relation'].notna() &
            df['aviation_routes'].notna() &
            df['migrant_stock'].notna() &
            df['trade_value'].notna()
        ]
        coverage = len(all_four) / len(df) * 100
        
        if coverage > 10:
            lines.append(f"✓ 4レイヤー全てのカバレッジ: {coverage:.1f}%")
        elif coverage > 5:
            lines.append(f"⚠ 4レイヤー全てのカバレッジ: {coverage:.1f}%")
        else:
            lines.append(f"⚠ 4レイヤー全てのカバレッジ: {coverage:.1f}% - 各レイヤーの国カバレッジが異なる")
        
        return "\n".join(lines)

def load_and_validate(filepath, name, required_cols):
    """データセットをロードして検証"""
    print(f"\nLoading {name}...")
    
    if not os.path.exists(filepath):
        print(f"ERROR: {filepath} not found")
        return None
    
    df = pd.read_csv(filepath, encoding='utf-8')
    print(f"  Loaded {len(df):,} rows")
    
    # 列の確認
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  ERROR: Missing columns: {missing_cols}")
        return None
    
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  Countries: {len(set(df['origin'].unique()) | set(df['destination'].unique()))}")
    
    return df

def replicate_aviation_data(df_aviation, target_years):
    """航空路データを複数年に複製"""
    print("\n航空路データを複数年に複製中...")
    
    original_year = df_aviation['year'].iloc[0]
    print(f"  元の年: {original_year}")
    print(f"  複製先の年: {target_years}")
    
    # 元データから年列を除く
    df_base = df_aviation.drop(columns=['year']).copy()
    
    # 各年のコピーを作成
    dfs = []
    for year in target_years:
        df_copy = df_base.copy()
        df_copy['year'] = year
        dfs.append(df_copy)
    
    df_replicated = pd.concat(dfs, ignore_index=True)
    print(f"  複製後の行数: {len(df_replicated):,}")
    
    return df_replicated

def main():
    tracker = IntegrationQualityTracker()
    
    print("="*70)
    print(" MULTILAYER NETWORK INTEGRATION ")
    print("="*70)
    
    # 1. データロード
    print("\n[Step 1] Loading datasets...")
    
    df_diplomatic = load_and_validate(
        DIPLOMATIC_FILE, 
        "Diplomatic Network",
        ['year', 'origin', 'destination', 'embassy_level']
    )
    
    df_aviation = load_and_validate(
        AVIATION_FILE,
        "Aviation Network",
        ['year', 'origin', 'destination', 'route_count']
    )
    
    df_migration = load_and_validate(
        MIGRATION_FILE,
        "Migration Network",
        ['year', 'origin', 'destination', 'migrant_stock']
    )
    
    df_trade = load_and_validate(
        TRADE_FILE,
        "Trade Network",
        ['year', 'origin', 'destination', 'trade_value']
    )
    
    # エラーチェック
    if df_diplomatic is None or df_aviation is None or df_migration is None or df_trade is None:
        print("\nERROR: Failed to load one or more datasets")
        sys.exit(1)
    
    # 統計記録
    tracker.record_dataset_stats('外交', {
        '行数': len(df_diplomatic),
        '国数': len(set(df_diplomatic['origin']) | set(df_diplomatic['destination'])),
        '年範囲': f"{df_diplomatic['year'].min()}-{df_diplomatic['year'].max()}"
    })
    
    tracker.record_dataset_stats('航空路', {
        '行数': len(df_aviation),
        '国数': len(set(df_aviation['origin']) | set(df_aviation['destination'])),
        '年': df_aviation['year'].iloc[0]
    })
    
    tracker.record_dataset_stats('移民', {
        '行数': len(df_migration),
        '国数': len(set(df_migration['origin']) | set(df_migration['destination'])),
        '年範囲': f"{df_migration['year'].min()}-{df_migration['year'].max()}"
    })
    
    tracker.record_dataset_stats('貿易', {
        '行数': len(df_trade),
        '国数': len(set(df_trade['origin']) | set(df_trade['destination'])),
        '年範囲': f"{df_trade['year'].min()}-{df_trade['year'].max()}"
    })
    
    # 2. 航空路データを複製
    print("\n[Step 2] Replicating aviation data to multiple years...")
    df_aviation_multi = replicate_aviation_data(df_aviation, TARGET_YEARS)
    
    # 3. 国カバレッジ分析
    print("\n[Step 3] Analyzing country coverage...")
    
    countries_diplo = set(df_diplomatic['origin']) | set(df_diplomatic['destination'])
    countries_avia = set(df_aviation['origin']) | set(df_aviation['destination'])
    countries_migr = set(df_migration['origin']) | set(df_migration['destination'])
    countries_trade = set(df_trade['origin']) | set(df_trade['destination'])
    
    common_all = countries_diplo & countries_avia & countries_migr & countries_trade
    common_diplo_migr = countries_diplo & countries_migr
    all_countries = countries_diplo | countries_avia | countries_migr | countries_trade
    
    tracker.set_common_countries({
        '全データセット共通': common_all,
        '外交+移民 共通': common_diplo_migr,
        '全体（和集合）': all_countries
    })
    
    print(f"  全データセット共通: {len(common_all)}カ国")
    print(f"  外交+移民 共通: {len(common_diplo_migr)}カ国")
    print(f"  全体（和集合）: {len(all_countries)}カ国")
    
    if len(common_all) < 100:
        tracker.add_warning(f"3データセット全てに共通する国が少ない（{len(common_all)}カ国）")
    
    # 4. データ統合
    print("\n[Step 4] Integrating datasets...")
    
    # 4.1 外交データをベースに
    df_base = df_diplomatic[['year', 'origin', 'destination', 'embassy_level']].copy()
    df_base = df_base.rename(columns={'embassy_level': 'diplomatic_relation'})
    
    print(f"  ベース（外交）: {len(df_base):,}行")
    
    # 4.2 航空路データをマージ
    df_aviation_merge = df_aviation_multi[['year', 'origin', 'destination', 'route_count']].copy()
    df_aviation_merge = df_aviation_merge.rename(columns={'route_count': 'aviation_routes'})
    
    df_merged = df_base.merge(
        df_aviation_merge,
        on=['year', 'origin', 'destination'],
        how='outer'
    )
    
    print(f"  外交+航空路: {len(df_merged):,}行")
    
    # 4.3 移民データをマージ
    df_migration_merge = df_migration[['year', 'origin', 'destination', 'migrant_stock']].copy()
    
    df_merged = df_merged.merge(
        df_migration_merge,
        on=['year', 'origin', 'destination'],
        how='outer'
    )
    
    print(f"  外交+航空路+移民: {len(df_merged):,}行")
    
    # 4.4 貿易データをマージ
    df_trade_merge = df_trade[['year', 'origin', 'destination', 'trade_value']].copy()
    
    df_final = df_merged.merge(
        df_trade_merge,
        on=['year', 'origin', 'destination'],
        how='outer'
    )
    
    print(f"  最終（4レイヤー統合）: {len(df_final):,}行")
    
    # 5. データクリーニング
    print("\n[Step 5] Cleaning integrated data...")
    
    # 年でフィルタ
    pre_filter = len(df_final)
    df_final = df_final[df_final['year'].isin(TARGET_YEARS)]
    print(f"  年フィルタ後: {len(df_final):,}行（除外: {pre_filter - len(df_final):,}）")
    
    # 自己ループ除去
    pre_loops = len(df_final)
    df_final = df_final[df_final['origin'] != df_final['destination']]
    print(f"  自己ループ除去後: {len(df_final):,}行（除外: {pre_loops - len(df_final):,}）")
    
    # 少なくとも1つのレイヤーにデータがある行のみ保持
    pre_empty = len(df_final)
    df_final = df_final[
        df_final['diplomatic_relation'].notna() |
        df_final['aviation_routes'].notna() |
        df_final['migrant_stock'].notna() |
        df_final['trade_value'].notna()
    ]
    print(f"  空行除去後: {len(df_final):,}行（除外: {pre_empty - len(df_final):,}）")
    
    # 6. 最終フォーマット
    print("\n[Step 6] Final formatting...")
    
    # 列の順序
    df_final = df_final[['year', 'origin', 'destination', 
                         'diplomatic_relation', 'aviation_routes', 'migrant_stock', 'trade_value']]
    
    # ソート
    df_final = df_final.sort_values(['year', 'origin', 'destination'])
    
    # データ型
    df_final['year'] = df_final['year'].astype(int)
    df_final['diplomatic_relation'] = df_final['diplomatic_relation'].astype('Int64')  # nullable int
    df_final['aviation_routes'] = df_final['aviation_routes'].astype('Int64')
    df_final['migrant_stock'] = df_final['migrant_stock'].astype('Int64')
    df_final['trade_value'] = df_final['trade_value'].astype('Float64')  # nullable float for trade
    
    # 7. 統計サマリー
    print("\n" + "="*70)
    print(" INTEGRATION SUMMARY ")
    print("="*70)
    
    print(f"\n総行数: {len(df_final):,}")
    print(f"年: {sorted(df_final['year'].unique())}")
    print(f"国数: {len(set(df_final['origin']) | set(df_final['destination']))}")
    
    print("\n--- 各レイヤーのカバレッジ ---")
    print(f"外交関係あり: {df_final['diplomatic_relation'].notna().sum():,}行 ({df_final['diplomatic_relation'].notna().sum()/len(df_final)*100:.1f}%)")
    print(f"航空路あり: {df_final['aviation_routes'].notna().sum():,}行 ({df_final['aviation_routes'].notna().sum()/len(df_final)*100:.1f}%)")
    print(f"移民データあり: {df_final['migrant_stock'].notna().sum():,}行 ({df_final['migrant_stock'].notna().sum()/len(df_final)*100:.1f}%)")
    print(f"貿易データあり: {df_final['trade_value'].notna().sum():,}行 ({df_final['trade_value'].notna().sum()/len(df_final)*100:.1f}%)")
    
    all_four = df_final[
        df_final['diplomatic_relation'].notna() &
        df_final['aviation_routes'].notna() &
        df_final['migrant_stock'].notna() &
        df_final['trade_value'].notna()
    ]
    print(f"4レイヤー全て: {len(all_four):,}行 ({len(all_four)/len(df_final)*100:.1f}%)")
    
    print("\n--- サンプルデータ（4レイヤー全て） ---")
    if len(all_four) > 0:
        print(all_four.head(10))
    else:
        print("  該当なし")
    
    # 8. レポート生成
    print("\n[Step 7] Generating quality report...")
    report = tracker.generate_report(df_final)
    print("\n" + report)
    
    # レポート保存
    with open(QUALITY_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n品質レポート保存: {QUALITY_REPORT_FILE}")
    
    # 9. データ保存
    print(f"\n[Step 8] Saving integrated data...")
    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"データ保存: {OUTPUT_FILE}")
    
    print("\n" + "="*70)
    print(" ✓ 統合完了 ")
    print("="*70)
    print(f"\n出力ファイル:")
    print(f"  - {OUTPUT_FILE}")
    print(f"  - {QUALITY_REPORT_FILE}")

if __name__ == "__main__":
    main()