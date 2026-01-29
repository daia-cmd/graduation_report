#!/usr/bin/env python3
"""
貿易データ前処理スクリプト

このスクリプトは以下を実行：
1. BACI貿易データの読み込み
2. データクリーニング（自己ループ、欠損値の除去）
3. 2000-2020年のデータフィルタリング
4. ISO3コード形式の確認
5. 品質レポートの生成
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
from collections import defaultdict

warnings.simplefilter(action='ignore', category=UserWarning)

# --- CONFIGURATION ---
INPUT_FILE = r'data/raw/trade.csv'
OUTPUT_FILE = r'data/processed/trade_network.csv'
QUALITY_REPORT_FILE = r'data/quality_reports/trade_data_quality_report.txt'
TARGET_YEARS = [2000, 2005, 2010, 2015, 2020]

# データ品質トラッキング
class DataQualityTracker:
    def __init__(self):
        self.initial_rows = 0
        self.removed = defaultdict(lambda: {'count': 0, 'details': []})
        self.warnings = []
        
    def set_initial(self, count):
        self.initial_rows = count
    
    def record_removal(self, reason, count, details=None):
        self.removed[reason]['count'] = count
        if details:
            self.removed[reason]['details'] = details
    
    def add_warning(self, message):
        self.warnings.append(message)
    
    def generate_report(self, final_count, df_final):
        report = []
        report.append("="*70)
        report.append(" DATA QUALITY REPORT - TRADE NETWORK ")
        report.append("="*70)
        report.append(f"\n初期行数（全データ）: {self.initial_rows:,}")
        report.append(f"最終行数（有効な貿易フロー）: {final_count:,}")
        report.append(f"保持率: {final_count/self.initial_rows*100:.2f}%")
        
        report.append("\n" + "-"*70)
        report.append(" フィルタリング・除外の内訳 ")
        report.append("-"*70)
        
        for reason, data in sorted(self.removed.items(),
                                   key=lambda x: x[1]['count'],
                                   reverse=True):
            count = data['count']
            pct = count / self.initial_rows * 100
            report.append(f"\n【{reason}】")
            report.append(f"  除外: {count:,} ({pct:.2f}%)")
            
            if data['details']:
                report.append(f"  詳細:")
                for detail in data['details'][:10]:
                    report.append(f"    - {detail}")
                if len(data['details']) > 10:
                    report.append(f"    ... and {len(data['details'])-10} more")
        
        if self.warnings:
            report.append("\n" + "-"*70)
            report.append(" 警告 ")
            report.append("-"*70)
            for warning in self.warnings:
                report.append(f"  ⚠ {warning}")
        
        # 統計情報
        report.append("\n" + "="*70)
        report.append(" 貿易ネットワーク統計 ")
        report.append("="*70)
        
        countries = set(df_final['origin']) | set(df_final['destination'])
        report.append(f"国数: {len(countries)}")
        report.append(f"年: {sorted(df_final['year'].unique())}")
        
        # 貿易額統計
        report.append(f"\n貿易額統計（USD）:")
        report.append(f"  最小値: ${df_final['trade_value'].min():,.2f}")
        report.append(f"  最大値: ${df_final['trade_value'].max():,.2f}")
        report.append(f"  平均値: ${df_final['trade_value'].mean():,.2f}")
        report.append(f"  中央値: ${df_final['trade_value'].median():,.2f}")
        
        # 年別統計
        report.append(f"\n年別貿易フロー数:")
        for year in sorted(df_final['year'].unique()):
            count = len(df_final[df_final['year'] == year])
            report.append(f"  {year}: {count:,}")
        
        # トップ10貿易回廊（2020年）
        if 2020 in df_final['year'].values:
            report.append(f"\nトップ10貿易回廊（2020年）:")
            top_2020 = df_final[df_final['year'] == 2020].nlargest(10, 'trade_value')
            for idx, row in top_2020.iterrows():
                report.append(f"  {row['origin']} → {row['destination']}: ${row['trade_value']:,.0f}")
        
        report.append("\n" + "="*70)
        report.append(" データ品質評価 ")
        report.append("="*70)
        report.append(self._evaluate_quality(final_count, df_final))
        
        return "\n".join(report)
    
    def _evaluate_quality(self, final_count, df_final):
        lines = []
        
        # チェック1: データ量
        if final_count > 100000:
            lines.append("✓ 十分なデータ量（> 100,000行）")
        elif final_count > 50000:
            lines.append("⚠ データ量がやや少ない（50,000-100,000行）")
        else:
            lines.append("✗ データ量が少ない（< 50,000行）")
        
        # チェック2: 年カバレッジ
        years_covered = len(df_final['year'].unique())
        if years_covered == 21:
            lines.append("✓ 完全な年カバレッジ（2000-2020）")
        elif years_covered >= 15:
            lines.append(f"⚠ 部分的な年カバレッジ（{years_covered}/21年）")
        else:
            lines.append(f"✗ 不完全な年カバレッジ（{years_covered}/21年）")
        
        # チェック3: 欠損値
        missing_pct = (df_final.isnull().sum().sum() / (len(df_final) * len(df_final.columns))) * 100
        if missing_pct == 0:
            lines.append("✓ 欠損値なし")
        elif missing_pct < 1:
            lines.append(f"⚠ 少量の欠損値（{missing_pct:.2f}%）")
        else:
            lines.append(f"✗ 多数の欠損値（{missing_pct:.2f}%）")
        
        return "\n".join(lines)

def main():
    tracker = DataQualityTracker()
    
    print("="*70)
    print(" TRADE DATA PREPROCESSING WITH QUALITY TRACKING ")
    print("="*70)
    
    # 1. Load CSV
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found at {INPUT_FILE}")
        sys.exit(1)
    
    print(f"\nLoading {INPUT_FILE}...")
    print("（このファイルは大きいため、読み込みに時間がかかる場合があります）")
    
    df = pd.read_csv(INPUT_FILE)
    
    tracker.set_initial(len(df))
    print(f"初期行数: {len(df):,}")
    print(f"列: {list(df.columns)}")
    
    # 2. データ構造の確認
    print(f"\nデータ構造:")
    print(f"  年の範囲: {df['year'].min()} - {df['year'].max()}")
    print(f"  ユニークな起点国: {df['iso3_o'].nunique()}")
    print(f"  ユニークな終点国: {df['iso3_d'].nunique()}")
    
    # 3. 年フィルタリング（2000-2020）
    print(f"\nFiltering for years 2000-2020...")
    pre_year_filter = len(df)
    df = df[df['year'].isin(TARGET_YEARS)].copy()
    tracker.record_removal(
        "対象年外（2000-2020以外）",
        pre_year_filter - len(df)
    )
    print(f"除外: {pre_year_filter - len(df):,}行（対象年外）")
    print(f"残り: {len(df):,}行")
    
    # 4. 自己ループの除去
    print(f"\nRemoving self-loops...")
    pre_loops = len(df)
    df = df[df['iso3_o'] != df['iso3_d']].copy()
    tracker.record_removal("自己ループ", pre_loops - len(df))
    print(f"除外: {pre_loops - len(df):,}行（自己ループ）")
    print(f"残り: {len(df):,}行")
    
    # 5. 欠損値・ゼロ値の除去
    print(f"\nCleaning missing/zero trade values...")
    pre_clean = len(df)
    
    # tradeflow_usdを使用（より正確なUSD値）
    df['tradeflow_usd'] = pd.to_numeric(df['tradeflow_usd'], errors='coerce')
    
    # 欠損値とゼロ値を除去
    df = df.dropna(subset=['tradeflow_usd'])
    df = df[df['tradeflow_usd'] > 0].copy()
    
    tracker.record_removal("無効/ゼロの貿易額", pre_clean - len(df))
    print(f"除外: {pre_clean - len(df):,}行（無効/ゼロの貿易額）")
    print(f"残り: {len(df):,}行")
    
    # 6. 最終フォーマット
    print(f"\nFormatting output...")
    df_final = df[['year', 'iso3_o', 'iso3_d', 'tradeflow_usd']].copy()
    df_final = df_final.rename(columns={
        'iso3_o': 'origin',
        'iso3_d': 'destination',
        'tradeflow_usd': 'trade_value'
    })
    
    # ソート
    df_final = df_final.sort_values(['year', 'origin', 'destination'])
    
    # データ型の最適化
    df_final['year'] = df_final['year'].astype(int)
    df_final['trade_value'] = df_final['trade_value'].astype(float)
    
    final_count = len(df_final)
    
    # 7. 統計情報の表示
    print("\n" + "="*70)
    print(" TRADE NETWORK STATISTICS ")
    print("="*70)
    print(f"Total trade flows: {final_count:,}")
    print(f"Years: {sorted(df_final['year'].unique())}")
    
    countries = set(df_final['origin']) | set(df_final['destination'])
    print(f"Countries: {len(countries)}")
    
    print(f"\nTrade value statistics (USD):")
    print(f"  Min: ${df_final['trade_value'].min():,.2f}")
    print(f"  Max: ${df_final['trade_value'].max():,.2f}")
    print(f"  Mean: ${df_final['trade_value'].mean():,.2f}")
    print(f"  Median: ${df_final['trade_value'].median():,.2f}")
    
    print("\n--- Top 10 Trade Corridors (2020) ---")
    if 2020 in df_final['year'].values:
        top_2020 = df_final[df_final['year'] == 2020].nlargest(10, 'trade_value')
        for _, row in top_2020.iterrows():
            print(f"  {row['origin']} → {row['destination']}: ${row['trade_value']:,.0f}")
    
    # 8. レポート生成
    report = tracker.generate_report(final_count, df_final)
    print("\n" + report)
    
    # 9. レポート保存
    with open(QUALITY_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n品質レポート保存: {QUALITY_REPORT_FILE}")
    
    # 10. データ保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(QUALITY_REPORT_FILE), exist_ok=True) 
    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"データ保存: {OUTPUT_FILE}")
    print(f"\n✓ 完了")

if __name__ == "__main__":
    main()
