import pandas as pd

encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

for enc in encodings:
    try:
        df = pd.read_csv('data/raw/diplometrics_ddr.csv', encoding=enc, nrows=5)
        print(f"\n✓ {enc} works!")
        print(df.head())
        break
    except:
        print(f"✗ {enc} failed")