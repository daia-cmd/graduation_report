# 貿易レイヤー追加 - 実装ガイド（更新版）

## 概要

このプロジェクトでは、多層ネットワーク分析に貿易データレイヤーを追加します。外交、航空、移住の既存レイヤーに加えて、貿易フローデータを統合し、包括的な国際関係ネットワーク分析を実現します。

## ディレクトリ構造

```
graduation_report/
├── .venv/                              # Python仮想環境
├── data/
│   ├── raw/                            # 生データ
│   │   ├── trade.csv                   # 生の貿易データ（1.1M+ 行）
│   │   ├── diplometrics_ddr.csv        # 外交データ
│   │   ├── airports.dat                # 空港データ
│   │   ├── routes.dat                  # 航空路線データ
│   │   └── un_migrant_stock.xlsx       # 移住データ
│   ├── midstate/                       # 中間処理データ
│   │   ├── aviation_network_raw.csv
│   │   ├── diplomatic_network.csv
│   │   └── multilayer_network.csv
│   ├── processed/                      # 処理済みデータ
│   │   ├── trade_network.csv           # [新規] 処理済み貿易データ
│   │   └── migration_network.csv
│   └── quality_reports/                # データ品質レポート
│       ├── trade_data_quality_report.txt                      # [新規]
│       └── multilayer_integration_quality_report.txt          # [更新]
├── scripts/
│   ├── preprocess/                     # 前処理スクリプト
│   │   ├── preprocess_trade.py         # [新規] 貿易データ前処理
│   │   ├── preprocess_diplomacy.py
│   │   ├── preprocess_aviation.py
│   │   └── preprocess_migration.py
│   └── analysis/                       # 分析スクリプト
│       ├── integrate_multilayer_network.py     # [修正] 多層統合
│       ├── network_centrality_analysis.py      # [修正] 中心性分析
│       └── centrality_detailed_analysis.py     # [修正] 詳細分析
├── outputs/
│   ├── figures/                        # 可視化図表
│   ├── tables/                         # 分析結果テーブル
│   └── reports/                        # 分析レポート
├── notebooks/                          # Jupyter notebooks
├── docs/                               # ドキュメント
├── .gitignore
├── README.md
└── directory_structure.txt
```

## 実装された変更

### 1. データ前処理（新規）

**ファイル**: `scripts/preprocess/preprocess_trade.py`

**機能**:
- `data/raw/trade.csv` を読み込み（1.1M+ 行）
- 2000-2020年のデータをフィルタリング
- 自己ループ（origin == destination）を削除
- 貿易額の欠損値を削除
- `tradeflow_usd` を `trade_value` として使用
- `data/processed/trade_network.csv` に出力

**出力**:
- `data/processed/trade_network.csv` - 処理済みデータ
  - カラム: `year`, `origin`, `destination`, `trade_value`
- `data/quality_reports/trade_data_quality_report.txt` - 品質レポート

**品質レポート内容**:
- 総行数、処理済み行数、フィルタ率
- ユニーク国数（起点国、終点国）
- 貿易額の分布統計（最小、最大、平均、中央値）
- Top 10 貿易回廊
- 欠損値の割合
- 年カバレッジ（2000-2020）

### 2. 多層ネットワーク統合（修正）

**ファイル**: `scripts/analysis/integrate_multilayer_network.py`

**変更点**:
- 貿易レイヤーを統合対象に追加
- `data/processed/trade_network.csv` を読み込み
- 外交、航空、移住レイヤーとマージ
- `data/midstate/multilayer_network.csv` に `trade_value` カラムを追加
- 統合レポートに貿易レイヤー統計を追加

**入力**:
- `data/midstate/diplomatic_network.csv`
- `data/midstate/aviation_network_raw.csv`
- `data/processed/migration_network.csv`
- `data/processed/trade_network.csv` [新規]

**出力**:
- `data/midstate/multilayer_network.csv` - 更新版統合データ
- `data/quality_reports/multilayer_integration_quality_report.txt` - 更新版レポート

### 3. 中心性分析（修正）

**ファイル**: `scripts/analysis/network_centrality_analysis.py`

**変更点**:
- `layers` リストに `'trade_value'` を追加
- 日本語名マッピングに「貿易」を追加
- 貿易レイヤーの中心性指標を計算
  - PageRank
  - 媒介中心性（Betweenness）
  - 固有ベクトル中心性
  - 入次数・出次数
  - クラスタリング係数
- 貿易レイヤーの可視化を生成

**入力**:
- `data/midstate/multilayer_network.csv`

**出力**:
- `outputs/tables/centrality_*.csv` - 全レイヤーの中心性指標
- `outputs/figures/fig*_trade_*.png` - 貿易レイヤー可視化
- `outputs/reports/network_centrality_report.txt` - 分析レポート

### 4. 詳細分析（修正）

**ファイル**: `scripts/analysis/centrality_detailed_analysis.py`

**変更点**:
- `layers` リストに `'trade_value'` を追加
- 日本語名マッピングに「貿易」を追加
- 相関分析に貿易レイヤーを含める
- 貿易レイヤーのTop 20ランキングを生成
- レイヤー間・中心性指標間の相関分析

**入力**:
- `data/midstate/multilayer_network.csv`
- `outputs/tables/centrality_*.csv`

**出力**:
- `outputs/figures/fig*_correlation*.png` - 相関分析可視化
- `outputs/tables/top20_*_trade_*.csv` - Top 20ランキング
- `outputs/reports/centrality_detailed_report.txt` - 詳細レポート

## 使用方法

### 前提条件

#### 必要なPythonパッケージ

```bash
# 仮想環境の作成（初回のみ）
python -m venv .venv

# 仮想環境の有効化
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# パッケージのインストール
pip install pandas numpy networkx matplotlib seaborn openpyxl
```

#### 必要なデータファイル

`data/raw/` ディレクトリに以下のファイルを配置：
- `trade.csv` - BACI貿易データ（必須）
- `diplometrics_ddr.csv` - 外交データ
- `airports.dat`, `routes.dat` - 航空データ
- `un_migrant_stock.xlsx` - 移住データ

### 実行手順

#### ステップ1: 貿易データの前処理

```bash
# プロジェクトルートで実行
python scripts/preprocess/preprocess_trade.py
```

**確認項目**:
- ✅ `data/processed/trade_network.csv` が作成されたか
- ✅ `data/quality_reports/trade_data_quality_report.txt` が作成されたか
- ✅ データに自己ループがないか
- ✅ 年の範囲が2000-2020か
- ✅ 欠損値が削除されているか

**期待される出力**:
```
============================================================
貿易データ前処理スクリプト
============================================================
貿易データを読み込み中: data/raw/trade.csv
読み込み完了: 1,100,000 行

=== データフィルタリング ===
年フィルタ (2000-2020): 1,100,000 → 850,000 行
自己ループ削除: 850,000 → 845,000 行 (5,000 行削除)
欠損値削除: 845,000 → 840,000 行 (5,000 行削除)

=== データ変換 ===
最終データ行数: 840,000

処理済みデータを保存中: data/processed/trade_network.csv
保存完了

=== 品質レポート生成 ===
品質レポート保存: data/quality_reports/trade_data_quality_report.txt

処理完了!
```

#### ステップ2: 多層ネットワークの統合

```bash
python scripts/analysis/integrate_multilayer_network.py
```

**確認項目**:
- ✅ `data/midstate/multilayer_network.csv` が更新されたか
- ✅ `trade_value` カラムが追加されたか
- ✅ 統合レポートに貿易レイヤー統計が含まれているか

**期待される出力**:
```
============================================================
多層ネットワーク統合スクリプト
============================================================
=== レイヤーデータ読み込み ===
外交レイヤー: 50,000 行読み込み
航空レイヤー: 60,000 行読み込み
移住レイヤー: 40,000 行読み込み
貿易レイヤー: 840,000 行読み込み

=== カラム名標準化 ===
外交レイヤー: 標準化完了
航空レイヤー: 標準化完了
移住レイヤー: 標準化完了
貿易レイヤー: 標準化完了

=== レイヤー統合 ===
基準レイヤー: trade (840,000 行)
diplomatic レイヤー追加: 840,000 → 850,000 行
aviation レイヤー追加: 850,000 → 860,000 行
migration レイヤー追加: 860,000 → 865,000 行

統合完了: 865,000 行, 7 列

統合データを保存中: data/midstate/multilayer_network.csv
保存完了

統合完了!
```

#### ステップ3: 中心性分析の実行

```bash
python scripts/analysis/network_centrality_analysis.py
```

**確認項目**:
- ✅ `outputs/tables/centrality_*.csv` に貿易レイヤーのデータがあるか
- ✅ 貿易レイヤーの可視化ファイルが作成されたか
- ✅ 図表が `outputs/figures/` に保存されたか

**期待される出力**:
```
============================================================
ネットワーク中心性分析
============================================================
多層ネットワークデータ読み込み: data/midstate/multilayer_network.csv
読み込み完了: 865,000 行, 7 列

分析対象年: [2000, 2001, ..., 2020]
利用可能なレイヤー: ['diplomatic_ties', 'aviation_connections', 'migration_flow', 'trade_value']

============================================================
中心性指標計算
============================================================

2020年:
  diplomatic_ties (2020年): ノード数=180, エッジ数=15,234
  aviation_connections (2020年): ノード数=200, エッジ数=20,456
  migration_flow (2020年): ノード数=190, エッジ数=18,789
  trade_value (2020年): ノード数=195, エッジ数=35,678

中心性指標保存: outputs/tables/centrality_multilayer_2020.csv

============================================================
可視化
============================================================

外交レイヤー:
  保存: trade_value_pagerank_top20_2020.png
  保存: trade_value_betweenness_top20_2020.png
  保存: trade_value_eigenvector_top20_2020.png

分析完了!
出力ディレクトリ: outputs/
```

#### ステップ4: 詳細分析の実行

```bash
python scripts/analysis/centrality_detailed_analysis.py
```

**確認項目**:
- ✅ Top 20ランキングに貿易レイヤーが含まれているか
- ✅ 相関分析に貿易レイヤーが含まれているか
- ✅ 可視化が日本語で適切にラベル付けされているか
- ✅ レポートが `outputs/reports/` に保存されたか

### 自動検証

全ての実装を自動的に検証するには:

```bash
python verify_implementation.py
```

このスクリプトは以下を確認します:
1. ファイルの存在
2. データ構造の正確性
3. 必須カラムの存在
4. データ品質（自己ループ、欠損値、年範囲）
5. レポートの完全性

## データ仕様

### 入力データ: `data/raw/trade.csv`

**必須カラム**:
- `year`: 年（整数）
- `iso3_o`: 起点国のISO3コード
- `iso3_d`: 終点国のISO3コード
- `tradeflow_usd`: 貿易額（米ドル）

**データ形式例**:
```csv
year,iso3_o,iso3_d,tradeflow_usd
2020,USA,CHN,120500000000
2020,CHN,USA,435600000000
```

### 出力データ: `data/processed/trade_network.csv`

**カラム**:
- `year`: 年
- `origin`: 起点国（ISO3コード）
- `destination`: 終点国（ISO3コード）
- `trade_value`: 貿易額（米ドル）

**データ条件**:
- 年: 2000-2020
- 自己ループなし（origin ≠ destination）
- 欠損値なし
- trade_value ≥ 0

### 統合データ: `data/midstate/multilayer_network.csv`

**カラム**:
- `year`: 年
- `origin`: 起点国
- `destination`: 終点国
- `diplomatic_ties`: 外交関係値
- `aviation_connections`: 航空接続値
- `migration_flow`: 移住フロー値
- `trade_value`: 貿易額（新規追加）

## トラブルシューティング

### エラー: `KeyError: 'trade_value'`

**原因**: `multilayer_network.csv` に `trade_value` カラムが存在しない

**解決策**:
1. `python scripts/preprocess/preprocess_trade.py` を実行して `trade_network.csv` を作成
2. `python scripts/analysis/integrate_multilayer_network.py` を実行して統合データを更新
3. その後、分析スクリプトを実行

### エラー: `FileNotFoundError: data/raw/trade.csv`

**原因**: 生データファイルが配置されていない

**解決策**:
- `data/raw/trade.csv` が存在することを確認
- BACI貿易データベースからダウンロード
- ファイルパスが正しいことを確認

### エラー: `ModuleNotFoundError: No module named 'pandas'`

**原因**: 必要なパッケージがインストールされていない

**解決策**:
```bash
# 仮想環境を有効化
.venv\Scripts\activate  # Windows
# または
source .venv/bin/activate  # macOS/Linux

# パッケージをインストール
pip install pandas numpy networkx matplotlib seaborn openpyxl
```

### 警告: データがない

**原因**: 特定の年やレイヤーにデータが存在しない

**解決策**:
- 品質レポートを確認してデータカバレッジを把握
- `data/quality_reports/trade_data_quality_report.txt` を確認
- 必要に応じてフィルタ条件を調整

### メモリエラー

**原因**: 大量データ（1.1M+行）の処理でメモリ不足

**解決策**:
- チャンク処理を使用（スクリプト内で実装済み）
- 年範囲を限定（例: 2015-2020のみ）
- 不要なレイヤーを一時的に除外

## 品質保証

### 自動テスト

`verify_implementation.py` は以下を検証:

1. **前処理の検証**
   - ✅ 出力ファイルの存在
   - ✅ データ構造の正確性（year, origin, destination, trade_value）
   - ✅ 自己ループの不在
   - ✅ 年範囲の正確性（2000-2020）
   - ✅ 欠損値の不在
   - ✅ 品質レポートの完全性

2. **統合の検証**
   - ✅ 統合ファイルの更新
   - ✅ `trade_value` カラムの存在
   - ✅ 統合レポートの更新

3. **中心性分析の検証**
   - ✅ 中心性指標の計算
   - ✅ 貿易レイヤーのデータ存在
   - ✅ 可視化ファイルの作成

4. **詳細分析の検証**
   - ✅ Top 20ランキングの生成
   - ✅ 相関分析の実施
   - ✅ 可視化の作成

### 手動確認チェックリスト

#### 1. 品質レポートのレビュー
- [ ] `data/quality_reports/trade_data_quality_report.txt` を開く
- [ ] 統計値が妥当か確認（国数: 150-200, 年数: 21）
- [ ] Top 10貿易回廊が論理的か確認（USA-CHN, CHN-USA等）
- [ ] 欠損値率が許容範囲内か確認（< 5%）

#### 2. 上位国の確認
- [ ] 中心性ランキングで上位に来る国が妥当か
- [ ] 主要経済国（USA, CHN, DEU, JPN, GBR等）が含まれているか
- [ ] 予想外の国が上位にいる場合、理由を調査

#### 3. レイヤー間比較
- [ ] 貿易レイヤーが他のレイヤーと比較可能か
- [ ] 相関分析の結果が解釈可能か
- [ ] レイヤー間で整合性があるか

#### 4. 可視化の確認
- [ ] グラフのタイトルや軸ラベルが日本語で表示されているか
- [ ] 文字化けがないか
- [ ] カラースキームが適切か
- [ ] 図表が `outputs/figures/` に保存されているか

#### 5. 出力ファイルの確認
- [ ] `outputs/tables/` にCSVファイルが生成されているか
- [ ] `outputs/reports/` にレポートが生成されているか
- [ ] ファイル名が規則に従っているか
- [ ] ファイルサイズが妥当か

## パフォーマンス最適化

### 大規模データ処理のヒント

1. **メモリ効率化**
```python
# チャンク処理
chunksize = 100000
for chunk in pd.read_csv('data/raw/trade.csv', chunksize=chunksize):
    process_chunk(chunk)
```

2. **並列処理**
```python
from multiprocessing import Pool
with Pool(4) as p:
    results = p.map(analyze_year, years)
```

3. **キャッシング**
- 中間結果を `data/midstate/` に保存
- 再実行時は既存データを活用

## 拡張性

### 新しいレイヤーの追加手順

1. **前処理スクリプトを作成**
   - `scripts/preprocess/preprocess_<layer>.py` を作成
   - `preprocess_trade.py` を参考に実装

2. **統合スクリプトを更新**
   - `scripts/analysis/integrate_multilayer_network.py` の `load_layer_data()` にレイヤーを追加
   - `standardize_layer_columns()` にカラム名マッピングを追加

3. **分析スクリプトを更新**
   - `network_centrality_analysis.py` の `layers` リストに追加
   - `layer_names_jp` に日本語名を追加
   - `centrality_detailed_analysis.py` も同様に更新

4. **検証スクリプトを更新**
   - `verify_implementation.py` に新レイヤーの検証を追加

### カスタム分析の追加

1. `scripts/analysis/` に新しいスクリプトを作成
2. `data/midstate/multilayer_network.csv` を入力として使用
3. 結果を `outputs/` の適切なサブディレクトリに出力

## 参考情報

### 中心性指標の解釈

- **PageRank**: 重要なノードからのリンクを重視した影響力
- **媒介中心性**: ネットワーク内の情報フローの媒介役としての重要性
- **固有ベクトル中心性**: 影響力のあるノードとの接続を重視
- **次数中心性**: 直接接続の数（入次数・出次数）
- **クラスタリング係数**: ローカルな密度、近隣ノード間の接続度

### データソース

- **貿易データ**: BACI (Base pour l'Analyse du Commerce International)
- **外交データ**: Correlates of War Diplomatic Exchange Dataset
- **航空データ**: OpenFlights Database
- **移住データ**: UN DESA International Migrant Stock

### 推奨文献

- Barabási, A. L. (2016). Network Science. Cambridge University Press.
- Newman, M. E. J. (2018). Networks (2nd ed.). Oxford University Press.
- De Domenico, M., et al. (2013). Mathematical formulation of multilayer networks. Physical Review X.

## ライセンスと引用

このコードを使用する場合は、適切にプロジェクトを引用してください。

```
@misc{graduation_report_2025,
  title={多層ネットワーク分析による国際関係の可視化},
  author={あなたの名前},
  year={2025},
  howpublished={\url{https://github.com/your-repo}}
}
```

## サポート

問題が発生した場合:
1. `verify_implementation.py` を実行して問題を特定
2. エラーメッセージとログを確認
3. `data/quality_reports/` の品質レポートを確認
4. このREADMEのトラブルシューティングセクションを参照

---

**実装日**: 2026年1月29日  
**バージョン**: 2.0  
**対応レイヤー**: 外交、航空、移住、貿易  
**Pythonバージョン**: 3.10+  
**主要依存パッケージ**: pandas 2.0+, networkx 3.0+, matplotlib 3.5+