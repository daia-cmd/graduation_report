import os

print("="*70)
print(" ファイル配置確認 ")
print("="*70)

# 確認するファイル
files_to_check = {
    '外交データ': [
        'diplomatic_network.csv',
        'data/processed/diplomatic_network.csv',
        './diplomatic_network.csv'
    ],
    '航空路データ': [
        'aviation_network_raw.csv',
        'data/processed/aviation_network_raw.csv',
        './aviation_network_raw.csv'
    ],
    '移民データ': [
        'migration_network.csv',
        'data/processed/migration_network.csv',
        './migration_network.csv'
    ]
}

print("\n現在のディレクトリ:", os.getcwd())
print("\n検索結果:")

found_files = {}

for name, paths in files_to_check.items():
    print(f"\n【{name}】")
    found = False
    for path in paths:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024 / 1024  # MB
            print(f"  ✅ 見つかった: {path} ({size:.2f} MB)")
            found_files[name] = path
            found = True
            break
    if not found:
        print(f"  ❌ 見つからない")
        print(f"     検索したパス: {paths}")

print("\n" + "="*70)
print(" 推奨アクション ")
print("="*70)

if len(found_files) == 3:
    print("\n✅ すべてのファイルが見つかりました！")
    print("\n統合スクリプトを以下のように修正してください：")
    print("\n```python")
    print(f"DIPLOMATIC_FILE = '{found_files['外交データ']}'")
    print(f"AVIATION_FILE = '{found_files['航空路データ']}'")
    print(f"MIGRATION_FILE = '{found_files['移民データ']}'")
    print("```")
else:
    print("\n⚠️ 一部のファイルが見つかりません")
    print("\n以下のいずれかを実行してください：")
    print("\n1. ファイルを移動:")
    print("   - diplomatic_network.csv → カレントディレクトリ")
    print("   - aviation_network_raw.csv → カレントディレクトリ")
    print("   - migration_network.csv → data/processed/")
    print("\n2. または、統合スクリプトのパスを修正")

# ディレクトリ構造を表示
print("\n" + "="*70)
print(" ディレクトリ構造 ")
print("="*70)

print("\n推奨構造:")
print("""
your_project/
├── data/
│   ├── raw/
│   │   ├── diplometrics_ddr.csv
│   │   ├── routes.dat
│   │   ├── airports.dat
│   │   └── un_migrant_stock.xlsx
│   └── processed/
│       └── migration_network.csv
├── diplomatic_network.csv  ← 外交データ（処理済み）
├── aviation_network_raw.csv  ← 航空路データ（処理済み）
├── preprocess_*.py  ← 前処理スクリプト
└── integrate_multilayer_network.py  ← 統合スクリプト
""")