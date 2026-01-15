# Multilayer Network Analysis Project

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

Created: 2026-01-15
