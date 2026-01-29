import pandas as pd
import numpy as np
import os
import sys
import warnings
from collections import defaultdict

warnings.simplefilter(action='ignore', category=UserWarning)

# --- CONFIGURATION ---
INPUT_FILE = r'data/raw/un_migrant_stock.xlsx'
OUTPUT_FILE = r'data/processed/migration_network.csv'
QUALITY_REPORT_FILE =  r'data/quality_reports/migration_data_quality_report.txt'
SHEET_NAME = 'Table 1'
HEADER_ROW = 10
TARGET_YEARS = [2000, 2005, 2010, 2015, 2020]

# --- COUNTRY MAPPING DICTIONARY ---
COUNTRY_MAPPING = {
    "United States of America": "USA", "United Kingdom": "GBR", "Russian Federation": "RUS",
    "Republic of Korea": "KOR", "Democratic People's Republic of Korea": "PRK",
    "Viet Nam": "VNM", "Lao People's Democratic Republic": "LAO", "Syrian Arab Republic": "SYR",
    "Iran (Islamic Republic of)": "IRN", "United Republic of Tanzania": "TZA",
    "Bolivia (Plurinational State of)": "BOL", "Venezuela (Bolivarian Republic of)": "VEN",
    "Republic of Moldova": "MDA", "Democratic Republic of the Congo": "COD", "Congo": "COG",
    "Gambia": "GMB", "Bahamas": "BHS", "China, Hong Kong SAR": "HKG", "China, Macao SAR": "MAC",
    "State of Palestine": "PSE", "Holy See": "VAT", "Türkiye": "TUR", "Czechia": "CZE",
    "Côte d'Ivoire": "CIV", "Réunion": "REU", "Curaçao": "CUW",
    
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "Andorra": "AND", "Angola": "AGO",
    "Antigua and Barbuda": "ATG", "Argentina": "ARG", "Armenia": "ARM", "Australia": "AUS",
    "Austria": "AUT", "Azerbaijan": "AZE", "Bahrain": "BHR", "Bangladesh": "BGD",
    "Barbados": "BRB", "Belarus": "BLR", "Belgium": "BEL", "Belize": "BLZ", "Benin": "BEN",
    "Bhutan": "BTN", "Bolivia": "BOL", "Bosnia and Herzegovina": "BIH", "Botswana": "BWA",
    "Brazil": "BRA", "Brunei Darussalam": "BRN", "Bulgaria": "BGR", "Burkina Faso": "BFA",
    "Burundi": "BDI", "Cabo Verde": "CPV", "Cambodia": "KHM", "Cameroon": "CMR", "Canada": "CAN",
    "Central African Republic": "CAF", "Chad": "TCD", "Chile": "CHL", "China": "CHN",
    "Colombia": "COL", "Comoros": "COM", "Costa Rica": "CRI", "Croatia": "HRV", "Cuba": "CUB",
    "Cyprus": "CYP", "Denmark": "DNK", "Djibouti": "DJI", "Dominica": "DMA",
    "Dominican Republic": "DOM", "Ecuador": "ECU", "Egypt": "EGY", "El Salvador": "SLV",
    "Equatorial Guinea": "GNQ", "Eritrea": "ERI", "Estonia": "EST", "Eswatini": "SWZ",
    "Ethiopia": "ETH", "Fiji": "FJI", "Finland": "FIN", "France": "FRA", "Gabon": "GAB",
    "Georgia": "GEO", "Germany": "DEU", "Ghana": "GHA", "Greece": "GRC", "Grenada": "GRD",
    "Guatemala": "GTM", "Guinea": "GIN", "Guinea-Bissau": "GNB", "Guyana": "GUY", "Haiti": "HTI",
    "Honduras": "HND", "Hungary": "HUN", "Iceland": "ISL", "India": "IND", "Indonesia": "IDN",
    "Iraq": "IRQ", "Ireland": "IRL", "Israel": "ISR", "Italy": "ITA", "Jamaica": "JAM",
    "Japan": "JPN", "Jordan": "JOR", "Kazakhstan": "KAZ", "Kenya": "KEN", "Kiribati": "KIR",
    "Kuwait": "KWT", "Kyrgyzstan": "KGZ", "Latvia": "LVA", "Lebanon": "LBN", "Lesotho": "LSO",
    "Liberia": "LBR", "Libya": "LBY", "Liechtenstein": "LIE", "Lithuania": "LTU", "Luxembourg": "LUX",
    "Madagascar": "MDG", "Malawi": "MWI", "Malaysia": "MYS", "Maldives": "MDV", "Mali": "MLI",
    "Malta": "MLT", "Marshall Islands": "MHL", "Mauritania": "MRT", "Mauritius": "MUS",
    "Mexico": "MEX", "Micronesia (Federated States of)": "FSM", "Monaco": "MCO",
    "Mongolia": "MNG", "Montenegro": "MNE", "Morocco": "MAR", "Mozambique": "MOZ",
    "Myanmar": "MMR", "Namibia": "NAM", "Nauru": "NRU", "Nepal": "NPL", "Netherlands": "NLD",
    "New Zealand": "NZL", "Nicaragua": "NIC", "Niger": "NER", "Nigeria": "NGA",
    "North Macedonia": "MKD", "Norway": "NOR", "Oman": "OMN", "Pakistan": "PAK",
    "Palau": "PLW", "Panama": "PAN", "Papua New Guinea": "PNG", "Paraguay": "PRY", "Peru": "PER",
    "Philippines": "PHL", "Poland": "POL", "Portugal": "PRT", "Qatar": "QAT", "Romania": "ROU",
    "Rwanda": "RWA", "Saint Kitts and Nevis": "KNA", "Saint Lucia": "LCA",
    "Saint Vincent and the Grenadines": "VCT", "Samoa": "WSM", "San Marino": "SMR",
    "Sao Tome and Principe": "STP", "Saudi Arabia": "SAU", "Senegal": "SEN", "Serbia": "SRB",
    "Seychelles": "SYC", "Sierra Leone": "SLE", "Singapore": "SGP", "Slovakia": "SVK",
    "Slovenia": "SVN", "Solomon Islands": "SLB", "Somalia": "SOM", "South Africa": "ZAF",
    "South Sudan": "SSD", "Spain": "ESP", "Sri Lanka": "LKA", "Sudan": "SDN", "Suriname": "SUR",
    "Sweden": "SWE", "Switzerland": "CHE", "Tajikistan": "TJK", "Thailand": "THA",
    "Timor-Leste": "TLS", "Togo": "TGO", "Tonga": "TON", "Trinidad and Tobago": "TTO",
    "Tunisia": "TUN", "Turkmenistan": "TKM", "Tuvalu": "TUV", "Uganda": "UGA", "Ukraine": "UKR",
    "United Arab Emirates": "ARE", "Uruguay": "URY", "Uzbekistan": "UZB", "Vanuatu": "VUT",
    "Yemen": "YEM", "Zambia": "ZMB", "Zimbabwe": "ZWE",
}

REGIONAL_KEYWORDS = [
    'World', 'AFRICA', 'ASIA', 'EUROPE', 'LATIN AMERICA', 'NORTHERN AMERICA', 'OCEANIA',
    'Sub-Saharan', 'Northern Africa', 'Eastern Africa', 'Middle Africa', 'Southern Africa',
    'Western Africa', 'Western Asia', 'Eastern Asia', 'Southern Asia', 'South-Eastern Asia',
    'Central Asia', 'Central and Southern Asia', 'Eastern and South-Eastern Asia',
    'Latin America and the Caribbean', 'Central America', 'South America', 'Caribbean',
    'Oceania (excluding', 'Australia/New Zealand',
    'Europe and Northern America', 'Eastern Europe', 'Northern Europe', 'Southern Europe',
    'Western Europe', 'developed regions', 'developing countries', 'Least developed',
    'Land-locked', 'Small Island', 'income countries', 'income group', '-income',
    'OECD', 'European Union', 'No income group'
]

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
    
    def generate_report(self, final_count):
        report = []
        report.append("="*70)
        report.append(" DATA QUALITY REPORT - MIGRATION NETWORK ")
        report.append("="*70)
        report.append(f"\n初期行数（全データ）: {self.initial_rows:,}")
        report.append(f"最終行数（個別国ペア）: {final_count:,}")
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
            report.append(f"  除外/変換: {count:,} ({pct:.2f}%)")
            report.append(f"  正当性: {'✓' if self._is_justified(reason) else '⚠'}")
            
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
        
        report.append("\n" + "="*70)
        report.append(" 正当性評価 ")
        report.append("="*70)
        report.append(self._evaluate_justification(final_count))
        
        return "\n".join(report)
    
    def _is_justified(self, reason):
        justified_reasons = [
            "地域集計（個別国ではない）",
            "ISO変換失敗",
            "自己ループ",
            "無効/ゼロの移民数"
        ]
        return any(j in reason for j in justified_reasons)
    
    def _evaluate_justification(self, final_count):
        lines = []
        
        # チェック1: 地域除外率
        regional = self.removed.get('地域集計（個別国ではない）', {}).get('count', 0)
        if regional > 0:
            regional_pct = regional / self.initial_rows * 100
            if regional_pct > 50:
                lines.append(f"✓ 地域集計の除外: {regional_pct:.1f}% - 正常（UN DESAは地域集計を含む）")
            else:
                lines.append(f"⚠ 地域集計の除外: {regional_pct:.1f}% - 予想より少ない")
        
        # チェック2: ISO変換失敗率
        iso_failed = self.removed.get('ISO変換失敗', {}).get('count', 0)
        if iso_failed > 0:
            failure_rate = iso_failed / self.initial_rows * 100
            if failure_rate < 5:
                lines.append("✓ ISO変換失敗率が低い（< 5%）")
            else:
                lines.append(f"⚠ ISO変換失敗率: {failure_rate:.1f}%")
        
        # チェック3: 最終データ量
        if final_count > 30000:
            lines.append("✓ 十分なデータ量（> 30,000行）")
        elif final_count > 20000:
            lines.append("⚠ データ量がやや少ない（20,000-30,000行）")
        else:
            lines.append("✗ データ量が少ない（< 20,000行）")
        
        return "\n".join(lines)

try:
    import pycountry
    HAS_PYCOUNTRY = True
except ImportError:
    HAS_PYCOUNTRY = False

def get_iso_code(country_name):
    if not isinstance(country_name, str):
        return None
    clean_name = country_name.strip().rstrip('*').strip()
    if clean_name in COUNTRY_MAPPING:
        return COUNTRY_MAPPING[clean_name]
    if HAS_PYCOUNTRY:
        try:
            country = pycountry.countries.get(name=clean_name)
            if country:
                return country.alpha_3
            matches = pycountry.countries.search_fuzzy(clean_name)
            if matches:
                return matches[0].alpha_3
        except (LookupError, AttributeError):
            pass
    return None

def is_regional_name(name):
    if not isinstance(name, str):
        return True
    name_clean = str(name).strip()
    for keyword in REGIONAL_KEYWORDS:
        if keyword.lower() in name_clean.lower():
            return True
    return False

def main():
    tracker = DataQualityTracker()
    
    print("="*70)
    print(" MIGRATION DATA PREPROCESSING WITH QUALITY TRACKING ")
    print("="*70)
    
    # 1. Load Excel
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found at {INPUT_FILE}")
        sys.exit(1)
    
    print(f"\nLoading {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, header=HEADER_ROW)
    
    tracker.set_initial(len(df))
    print(f"初期行数: {len(df):,}")
    
    # 2. Extract columns
    dest_col = 'Region, development group, country or area of destination'
    origin_col = 'Region, development group, country or area of origin'
    
    cols_to_keep = [dest_col, origin_col] + TARGET_YEARS
    df_subset = df[cols_to_keep].copy()
    
    # 3. Filter regions
    print(f"\nFiltering regional aggregates...")
    initial_rows = len(df_subset)
    
    df_countries = df_subset[
        ~df_subset[dest_col].apply(is_regional_name) &
        ~df_subset[origin_col].apply(is_regional_name)
    ].copy()
    
    regional_removed = initial_rows - len(df_countries)
    tracker.record_removal(
        "地域集計（個別国ではない）",
        regional_removed,
        ["World→各国", "地域→各国", "集計グループ"]
    )
    print(f"除外: {regional_removed:,}行（地域集計）")
    print(f"残り: {len(df_countries):,}行（個別国ペア）")
    
    # 4. Clean names
    df_countries[dest_col] = df_countries[dest_col].str.strip().str.rstrip('*').str.strip()
    df_countries[origin_col] = df_countries[origin_col].str.strip().str.rstrip('*').str.strip()
    
    # 5. Melt
    df_long = df_countries.melt(
        id_vars=[dest_col, origin_col],
        value_vars=TARGET_YEARS,
        var_name='year',
        value_name='migrant_stock'
    )
    
    df_long = df_long.rename(columns={
        dest_col: 'destination',
        origin_col: 'origin'
    })
    
    # 6. Clean data
    df_long['migrant_stock'] = pd.to_numeric(df_long['migrant_stock'], errors='coerce')
    df_long['year'] = df_long['year'].astype(int)
    
    pre_clean = len(df_long)
    df_long = df_long.dropna(subset=['migrant_stock'])
    df_long = df_long[df_long['migrant_stock'] > 0]
    tracker.record_removal("無効/ゼロの移民数", pre_clean - len(df_long))
    
    # 7. ISO conversion
    print(f"\nConverting to ISO codes...")
    
    unconvertible_dest = set()
    unconvertible_origin = set()
    
    def convert_dest(name):
        code = get_iso_code(name)
        if not code:
            unconvertible_dest.add(name)
        return code
    
    def convert_origin(name):
        code = get_iso_code(name)
        if not code:
            unconvertible_origin.add(name)
        return code
    
    df_long['destination_iso'] = df_long['destination'].apply(convert_dest)
    df_long['origin_iso'] = df_long['origin'].apply(convert_origin)
    
    all_unconvertible = list(unconvertible_dest | unconvertible_origin)
    pre_iso = len(df_long)
    df_long = df_long.dropna(subset=['destination_iso', 'origin_iso'])
    
    if all_unconvertible:
        tracker.record_removal(
            "ISO変換失敗",
            pre_iso - len(df_long),
            all_unconvertible[:20]
        )
        if len(all_unconvertible) > 10:
            tracker.add_warning(f"多数の国名変換失敗: {len(all_unconvertible)}カ国")
    
    df_long['destination'] = df_long['destination_iso']
    df_long['origin'] = df_long['origin_iso']
    df_long = df_long.drop(columns=['destination_iso', 'origin_iso'])
    
    # 8. Remove self-loops
    pre_loops = len(df_long)
    df_long = df_long[df_long['origin'] != df_long['destination']]
    tracker.record_removal("自己ループ", pre_loops - len(df_long))
    
    # 9. Final format
    df_long = df_long[['year', 'origin', 'destination', 'migrant_stock']]
    df_long = df_long.sort_values(['year', 'origin', 'destination'])
    df_long['migrant_stock'] = df_long['migrant_stock'].astype(int)
    
    final_count = len(df_long)
    
    # 10. Statistics
    print("\n" + "="*70)
    print(" MIGRATION NETWORK STATISTICS ")
    print("="*70)
    print(f"Total migration flows: {final_count:,}")
    print(f"Years: {sorted(df_long['year'].unique())}")
    print(f"Countries: {len(set(df_long['origin']) | set(df_long['destination']))}")
    
    print("\n--- Top 10 Migration Corridors (2020) ---")
    top_2020 = df_long[df_long['year'] == 2020].nlargest(10, 'migrant_stock')
    for _, row in top_2020.iterrows():
        print(f"  {row['origin']} → {row['destination']}: {row['migrant_stock']:,}")
    
    # レポート生成
    report = tracker.generate_report(final_count)
    print("\n" + report)
    
    # レポート保存
    with open(QUALITY_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n品質レポート保存: {QUALITY_REPORT_FILE}")
    
    # データ保存
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(QUALITY_REPORT_FILE), exist_ok=True)
    df_long.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"データ保存: {OUTPUT_FILE}")
    print(f"\n✓ 完了")

if __name__ == "__main__":
    main()