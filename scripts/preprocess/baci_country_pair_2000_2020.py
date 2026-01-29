import pandas as pd

# 使う列だけ
usecols = [
    "year",
    "iso3_o",
    "iso3_d",
    "country_exists_o",
    "country_exists_d",
    "tradeflow_baci"
]

# chunksize を使って安全に読み込む
chunks = pd.read_csv(
    "Gravity_V202211.csv",
    usecols=usecols,
    chunksize=1_000_000
)

dfs = []
for chunk in chunks:
    chunk = chunk[
        (chunk["year"] >= 2000) &
        (chunk["year"] <= 2020) &
        (chunk["country_exists_o"] == 1) &
        (chunk["country_exists_d"] == 1)
    ]
    dfs.append(chunk)

df = pd.concat(dfs, ignore_index=True)

# 千ドル → ドル（任意）
df["tradeflow_usd"] = df["tradeflow_baci"] * 1000

# 最終的に6列（or 7列）に
df = df[[
    "year",
    "iso3_o",
    "iso3_d",
    "tradeflow_baci",
    "tradeflow_usd"
]]

print(df.head())
print(len(df))

df.to_csv("baci_country_pair_2000_2020.csv", index=False)
