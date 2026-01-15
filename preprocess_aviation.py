import pandas as pd
import numpy as np
import os
import sys
from collections import defaultdict

# --- CONFIGURATION ---
ROUTES_FILE = r'data/raw/routes.dat'
AIRPORTS_FILE = r'data/raw/airports.dat'
OUTPUT_FILE = 'aviation_network_raw.csv'
QUALITY_REPORT_FILE = 'aviation_data_quality_report.txt'
SNAPSHOT_YEAR = 2014

# --- COMPLETE COUNTRY MAPPING DICTIONARY ---
COUNTRY_MAPPING = {
    # 主要国
    "United States": "USA", "United Kingdom": "GBR", "Russia": "RUS", "Russian Federation": "RUS",
    "South Korea": "KOR", "Republic of Korea": "KOR", "North Korea": "PRK",
    "Democratic People's Republic of Korea": "PRK",
    "China": "CHN", "Hong Kong": "HKG", "Macau": "MAC", "Taiwan": "TWN",
    "Japan": "JPN", "Germany": "DEU", "France": "FRA", "Canada": "CAN",
    "Australia": "AUS", "Brazil": "BRA", "India": "IND", "Mexico": "MEX",
    
    # ヨーロッパ
    "Spain": "ESP", "Italy": "ITA", "Netherlands": "NLD", "Turkey": "TUR", "Türkiye": "TUR",
    "Switzerland": "CHE", "Austria": "AUT", "Belgium": "BEL", "Portugal": "PRT",
    "Greece": "GRC", "Ireland": "IRL", "Sweden": "SWE", "Norway": "NOR",
    "Denmark": "DNK", "Finland": "FIN", "Poland": "POL", "Czech Republic": "CZE",
    "Czechia": "CZE", "Hungary": "HUN", "Romania": "ROU", "Bulgaria": "BGR",
    "Croatia": "HRV", "Serbia": "SRB", "Slovakia": "SVK", "Slovenia": "SVN",
    "Estonia": "EST", "Latvia": "LVA", "Lithuania": "LTU", "Iceland": "ISL",
    "Luxembourg": "LUX", "Malta": "MLT", "Cyprus": "CYP", "Albania": "ALB",
    "Bosnia and Herzegovina": "BIH", "Montenegro": "MNE", "North Macedonia": "MKD",
    "Macedonia": "MKD", "Moldova": "MDA", "Republic of Moldova": "MDA",
    "Ukraine": "UKR", "Belarus": "BLR",
    
    # アジア
    "Singapore": "SGP", "Indonesia": "IDN", "Thailand": "THA", "Malaysia": "MYS",
    "Philippines": "PHL", "Vietnam": "VNM", "Viet Nam": "VNM", "Pakistan": "PAK",
    "Bangladesh": "BGD", "Myanmar": "MMR", "Burma": "MMR", "Cambodia": "KHM",
    "Laos": "LAO", "Lao People's Democratic Republic": "LAO",
    "Nepal": "NPL", "Sri Lanka": "LKA", "Afghanistan": "AFG", "Mongolia": "MNG",
    "Bhutan": "BTN", "Maldives": "MDV", "Brunei": "BRN", "Brunei Darussalam": "BRN",
    "Timor-Leste": "TLS", "East Timor": "TLS",
    
    # 中東
    "United Arab Emirates": "ARE", "Saudi Arabia": "SAU", "Israel": "ISR",
    "Iran": "IRN", "Iran (Islamic Republic of)": "IRN", "Iraq": "IRQ",
    "Kuwait": "KWT", "Qatar": "QAT", "Bahrain": "BHR", "Oman": "OMN",
    "Jordan": "JOR", "Lebanon": "LBN", "Syria": "SYR", "Syrian Arab Republic": "SYR",
    "Yemen": "YEM", "Palestine": "PSE", "State of Palestine": "PSE",
    
    # アフリカ
    "South Africa": "ZAF", "Egypt": "EGY", "Morocco": "MAR", "Kenya": "KEN",
    "Nigeria": "NGA", "Ethiopia": "ETH", "Ghana": "GHA", "Tanzania": "TZA",
    "United Republic of Tanzania": "TZA", "Algeria": "DZA", "Tunisia": "TUN",
    "Libya": "LBY", "Uganda": "UGA", "Senegal": "SEN", "Ivory Coast": "CIV",
    "Cote d'Ivoire": "CIV", "Côte d'Ivoire": "CIV", "Cameroon": "CMR",
    "Zimbabwe": "ZWE", "Madagascar": "MDG", "Angola": "AGO", "Mozambique": "MOZ",
    "Zambia": "ZMB", "Mauritius": "MUS", "Botswana": "BWA", "Namibia": "NAM",
    "Rwanda": "RWA", "Malawi": "MWI", "Mali": "MLI", "Burkina Faso": "BFA",
    "Niger": "NER", "Chad": "TCD", "Congo": "COG", "Congo, Rep.": "COG",
    "Democratic Republic of the Congo": "COD", "Congo, Dem. Rep.": "COD",
    "Gabon": "GAB", "Benin": "BEN", "Togo": "TGO", "Sierra Leone": "SLE",
    "Liberia": "LBR", "Guinea": "GIN", "Eritrea": "ERI", "Somalia": "SOM",
    "Djibouti": "DJI", "Seychelles": "SYC", "Comoros": "COM",
    
    # 南北アメリカ
    "Argentina": "ARG", "Colombia": "COL", "Chile": "CHL", "Peru": "PER",
    "Venezuela": "VEN", "Venezuela (Bolivarian Republic of)": "VEN",
    "Ecuador": "ECU", "Bolivia": "BOL", "Bolivia (Plurinational State of)": "BOL",
    "Paraguay": "PRY", "Uruguay": "URY", "Guyana": "GUY", "Suriname": "SUR",
    "Costa Rica": "CRI", "Panama": "PAN", "Guatemala": "GTM", "Honduras": "HND",
    "Nicaragua": "NIC", "El Salvador": "SLV", "Belize": "BLZ",
    "Cuba": "CUB", "Jamaica": "JAM", "Haiti": "HTI", "Dominican Republic": "DOM",
    "Trinidad and Tobago": "TTO", "Bahamas": "BHS", "Barbados": "BRB",
    "Saint Lucia": "LCA", "Grenada": "GRD", "Saint Vincent and the Grenadines": "VCT",
    "Antigua and Barbuda": "ATG", "Dominica": "DMA", "Saint Kitts and Nevis": "KNA",
    
    # オセアニア
    "New Zealand": "NZL", "Papua New Guinea": "PNG", "Fiji": "FJI",
    "Solomon Islands": "SLB", "Vanuatu": "VUT", "Samoa": "WSM",
    "Tonga": "TON", "Kiribati": "KIR", "Marshall Islands": "MHL",
    "Micronesia": "FSM", "Micronesia (Federated States of)": "FSM",
    "Palau": "PLW", "Nauru": "NRU", "Tuvalu": "TUV",
    
    # 中央アジア
    "Kazakhstan": "KAZ", "Uzbekistan": "UZB", "Turkmenistan": "TKM",
    "Tajikistan": "TJK", "Kyrgyzstan": "KGZ", "Armenia": "ARM",
    "Azerbaijan": "AZE", "Georgia": "GEO",
    
    # 特殊領域・属領
    "Greenland": "GRL", "French Polynesia": "PYF", "New Caledonia": "NCL",
    "Guam": "GUM", "Puerto Rico": "PRI", "US Virgin Islands": "VIR",
    "United States Virgin Islands": "VIR", "American Samoa": "ASM",
    "Northern Mariana Islands": "MNP", "Aruba": "ABW", "Curacao": "CUW",
    "Curaçao": "CUW", "Sint Maarten": "SXM", "Martinique": "MTQ",
    "Guadeloupe": "GLP", "French Guiana": "GUF", "Reunion": "REU",
    "Réunion": "REU", "Mayotte": "MYT", "Bermuda": "BMU", "Cayman Islands": "CYM",
    "Turks and Caicos Islands": "TCA", "British Virgin Islands": "VGB",
    "Anguilla": "AIA", "Montserrat": "MSR", "Falkland Islands": "FLK",
    "Gibraltar": "GIB", "Isle of Man": "IMN", "Jersey": "JEY", "Guernsey": "GGY",
    "Faroe Islands": "FRO", "Saint Pierre and Miquelon": "SPM",
    
    # その他
    "Vatican City": "VAT", "Holy See": "VAT", "Monaco": "MCO",
    "Liechtenstein": "LIE", "San Marino": "SMR", "Andorra": "AND",
}

# データ品質トラッキング
class DataQualityTracker:
    def __init__(self):
        self.initial_routes = 0
        self.initial_airports = 0
        self.removed = defaultdict(lambda: {'count': 0, 'details': []})
        self.warnings = []
        
    def set_initial_routes(self, count):
        self.initial_routes = count
    
    def set_initial_airports(self, count):
        self.initial_airports = count
    
    def record_removal(self, reason, count, details=None):
        self.removed[reason]['count'] = count
        if details:
            self.removed[reason]['details'] = details
    
    def add_warning(self, message):
        self.warnings.append(message)
    
    def generate_report(self, final_country_pairs, final_routes):
        report = []
        report.append("="*70)
        report.append(" DATA QUALITY REPORT - AVIATION NETWORK ")
        report.append("="*70)
        report.append(f"\n初期路線数: {self.initial_routes:,}")
        report.append(f"初期空港数: {self.initial_airports:,}")
        report.append(f"最終国ペア数: {final_country_pairs:,}")
        report.append(f"最終路線数: {final_routes:,}")
        
        if self.initial_routes > 0:
            route_retention = final_routes / self.initial_routes * 100
            report.append(f"路線保持率: {route_retention:.2f}%")
        
        report.append("\n" + "-"*70)
        report.append(" 除外/変換の内訳 ")
        report.append("-"*70)
        
        for reason, data in sorted(self.removed.items(),
                                   key=lambda x: x[1]['count'],
                                   reverse=True):
            count = data['count']
            if self.initial_routes > 0:
                pct = count / self.initial_routes * 100
                report.append(f"\n【{reason}】")
                report.append(f"  影響: {count:,} ({pct:.2f}%)")
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
        report.append(self._evaluate_justification(final_routes, final_country_pairs))
        
        return "\n".join(report)
    
    def _is_justified(self, reason):
        """除外理由が正当かチェック"""
        justified_reasons = [
            "空港コードマッピング失敗",
            "ISO変換失敗",
            "自己ループ（同一国内路線）",
            "\\N（データなし）"
        ]
        return any(j in reason for j in justified_reasons)
    
    def _evaluate_justification(self, final_routes, final_country_pairs):
        """全体的な正当性を評価"""
        lines = []
        
        # チェック1: 路線保持率
        if self.initial_routes > 0:
            retention = final_routes / self.initial_routes * 100
            if retention > 80:
                lines.append("✓ 路線保持率が高い（> 80%）- 良好")
            elif retention > 60:
                lines.append("⚠ 路線保持率が中程度（60-80%）")
            else:
                lines.append("✗ 路線保持率が低い（< 60%）- マッピング辞書の拡張推奨")
        
        # チェック2: ISO変換失敗
        iso_failed = self.removed.get('ISO変換失敗', {}).get('count', 0)
        if iso_failed > 0:
            failure_rate = iso_failed / self.initial_routes * 100
            if failure_rate < 10:
                lines.append("✓ ISO変換失敗率が低い（< 10%）")
            elif failure_rate < 20:
                lines.append("⚠ ISO変換失敗率がやや高い（10-20%）")
            else:
                lines.append(f"✗ ISO変換失敗率が高い（{failure_rate:.1f}%）- 要改善")
        
        # チェック3: 最終的な国カバレッジ
        lines.append(f"✓ 最終的な国ペア数: {final_country_pairs:,}")
        
        return "\n".join(lines)

# Pycountryフォールバック
try:
    import pycountry
    HAS_PYCOUNTRY = True
    print("Pycountry library detected.")
except ImportError:
    HAS_PYCOUNTRY = False
    print("Pycountry not found. Using comprehensive internal dictionary.")

def get_iso_code(country_name):
    if not isinstance(country_name, str):
        return None
    clean_name = country_name.strip()
    
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

def find_file(filepath):
    if os.path.exists(filepath):
        return filepath
    basename = os.path.basename(filepath)
    if os.path.exists(basename):
        return basename
    raw_path = os.path.join('data', 'raw', basename)
    if os.path.exists(raw_path):
        return raw_path
    return None

def load_airports(filepath, tracker):
    print(f"Loading airports from {filepath}...")
    try:
        df = pd.read_csv(filepath, header=None, encoding='utf-8', on_bad_lines='skip')
        
        tracker.set_initial_airports(len(df))
        
        if len(df.columns) < 6:
            print("Error: Airports file has too few columns.")
            return None
        
        mapping = {}
        for idx, row in df.iterrows():
            country = str(row[3]).strip()
            iata = str(row[4]).strip()
            icao = str(row[5]).strip()
            
            if len(iata) == 3 and iata != '\\N':
                mapping[iata] = country
            if len(icao) == 4 and icao != '\\N':
                mapping[icao] = country
        
        print(f"Parsed {len(df)} airports. Mapped {len(mapping)} airport codes.")
        return mapping
    
    except Exception as e:
        print(f"Error loading airports: {e}")
        return None

def load_routes(filepath, tracker):
    print(f"Loading routes from {filepath}...")
    try:
        df = pd.read_csv(filepath, header=None, encoding='utf-8', on_bad_lines='skip')
        tracker.set_initial_routes(len(df))
        print(f"Loaded {len(df)} raw route rows.")
        return df
    except Exception as e:
        print(f"Error loading routes: {e}")
        return None

def main():
    tracker = DataQualityTracker()
    
    print("="*70)
    print(" AVIATION DATA PREPROCESSING WITH QUALITY TRACKING ")
    print("="*70)
    
    # 1. ファイル検索
    airports_path = find_file(AIRPORTS_FILE)
    routes_path = find_file(ROUTES_FILE)
    
    if not airports_path or not routes_path:
        print(f"ERROR: Could not find data files.")
        sys.exit(1)
    
    # 2. 空港マッピング
    airport_country_map = load_airports(airports_path, tracker)
    if not airport_country_map:
        sys.exit(1)
    
    # 3. ルートロード
    routes_df = load_routes(routes_path, tracker)
    if routes_df is None:
        sys.exit(1)
    
    # 4. ルートを国にマッピング
    print("\nMapping routes to countries...")
    
    routes_work = routes_df[[2, 4]].copy()
    routes_work.columns = ['src', 'dst']
    
    # \\N除外
    null_routes = routes_work[(routes_work['src'] == '\\N') | (routes_work['dst'] == '\\N')]
    tracker.record_removal("\\N（データなし）", len(null_routes))
    
    routes_work = routes_work[(routes_work['src'] != '\\N') & (routes_work['dst'] != '\\N')]
    
    # 国名マッピング
    routes_work['src_country'] = routes_work['src'].map(airport_country_map)
    routes_work['dst_country'] = routes_work['dst'].map(airport_country_map)
    
    # マッピング失敗
    unmapped_src = routes_work[routes_work['src_country'].isna()]['src'].unique()
    unmapped_dst = routes_work[routes_work['dst_country'].isna()]['dst'].unique()
    
    unmapped_routes = routes_work[routes_work['src_country'].isna() | routes_work['dst_country'].isna()]
    tracker.record_removal(
        "空港コードマッピング失敗",
        len(unmapped_routes),
        list(set(unmapped_src) | set(unmapped_dst))[:20]
    )
    
    routes_mapped = routes_work.dropna(subset=['src_country', 'dst_country'])
    print(f"Routes mapped successfully: {len(routes_mapped)} / {tracker.initial_routes} ({len(routes_mapped)/tracker.initial_routes*100:.1f}%)")
    
    # 5. ISO変換
    print("\nConverting countries to ISO codes...")
    
    unique_countries = pd.unique(routes_mapped[['src_country', 'dst_country']].values.ravel('K'))
    print(f"Unique country names: {len(unique_countries)}")
    
    iso_map = {c: get_iso_code(c) for c in unique_countries}
    
    failed_countries = [c for c, iso in iso_map.items() if iso is None]
    if failed_countries:
        tracker.add_warning(f"ISO変換失敗: {len(failed_countries)}カ国")
        tracker.record_removal(
            "ISO変換失敗",
            len(routes_mapped[routes_mapped['src_country'].isin(failed_countries) | 
                              routes_mapped['dst_country'].isin(failed_countries)]),
            failed_countries[:20]
        )
    
    routes_mapped_copy = routes_mapped.copy()
    routes_mapped_copy['origin'] = routes_mapped_copy['src_country'].map(iso_map)
    routes_mapped_copy['destination'] = routes_mapped_copy['dst_country'].map(iso_map)
    
    final_df = routes_mapped_copy.dropna(subset=['origin', 'destination']).copy()
    print(f"After ISO conversion: {len(final_df)} routes")
    
    # 6. クリーニング
    print("\nCleaning data...")
    pre_loops = len(final_df)
    final_df = final_df[final_df['origin'] != final_df['destination']]
    tracker.record_removal("自己ループ（同一国内路線）", pre_loops - len(final_df))
    
    # 7. 集約
    print("\nAggregating routes...")
    agg_df = final_df.groupby(['origin', 'destination']).size().reset_index(name='route_count')
    agg_df['year'] = SNAPSHOT_YEAR
    agg_df = agg_df[['year', 'origin', 'destination', 'route_count']]
    
    # 8. 統計
    print("\n" + "="*70)
    print(" AVIATION NETWORK STATISTICS ")
    print("="*70)
    print(f"Total Unique Country Pairs: {len(agg_df)}")
    print(f"Total Routes: {agg_df['route_count'].sum()}")
    print(f"Number of Countries: {len(set(agg_df['origin']) | set(agg_df['destination']))}")
    
    print("\n--- Top 10 Country Pairs ---")
    print(agg_df.sort_values('route_count', ascending=False).head(10))
    
    # レポート生成
    report = tracker.generate_report(len(agg_df), agg_df['route_count'].sum())
    print("\n" + report)
    
    # レポート保存
    with open(QUALITY_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n品質レポート保存: {QUALITY_REPORT_FILE}")
    
    # データ保存
    agg_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"データ保存: {OUTPUT_FILE}")
    print(f"\n✓ 完了")

if __name__ == "__main__":
    main()