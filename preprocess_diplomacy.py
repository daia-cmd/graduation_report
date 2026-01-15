import pandas as pd
import numpy as np
import os
import sys
from collections import defaultdict

# --- CONFIGURATION ---
INPUT_FILE = r'data/raw/diplometrics_ddr.csv'
OUTPUT_FILE = 'diplomatic_network.csv'
QUALITY_REPORT_FILE = 'diplomatic_data_quality_report.txt'
START_YEAR = 2000
END_YEAR = 2020

# --- COUNTRY MAPPING DICTIONARY ---
# (完全版 - 省略せず全部含める)
COUNTRY_MAPPING = {
    "United States": "USA", "United Kingdom": "GBR", "Russia": "RUS",
    "South Korea": "KOR", "North Korea": "PRK", "Vietnam": "VNM",
    "Laos": "LAO", "Syria": "SYR", "Iran": "IRN", "Tanzania": "TZA",
    "Bolivia": "BOL", "Venezuela": "VEN", "Moldova": "MDA",
    "Congo, Dem. Rep.": "COD", "Congo, Rep.": "COG",
    "Gambia, The": "GMB", "Bahamas, The": "BHS",
    "USSR": "RUS", "Soviet Union": "RUS", "Yugoslavia": "SRB",
    "Czechoslovakia": "CZE", "East Germany": "DEU", "West Germany": "DEU",
    
    # 追加: Diplometrics特有の表記
    "Kosovo": "XKX", "Serbia and Montenegro": "SRB",
    "Northern Cyprus, Turkish Republic of": "CYP",
    "United KIngdom": "GBR",  # タイポ対応
    "C?te d�fIvoire": "CIV",  # エンコーディング破損
    "Micronesia, Federated States of": "FSM",
    "Korea, Republic of": "KOR",
    "Korea, Democratic People's Republic of": "PRK",
    "Viet Nam": "VNM",
    "Lao People's Democratic Republic": "LAO",
    "Iran, Islamic Republic of": "IRN",
    "Tanzania, United Republic of": "TZA",
    "Congo, Democratic Republic of the": "COD",
    "Congo, Republic of the": "COG",
    "Congo-Brazzaville": "COG",
    "Congo-Kinshasa": "COD",
    "Bolivia, Plurinational State of": "BOL",
    "Venezuela, Bolivarian Republic of": "VEN",
    "Moldova, Republic of": "MDA",
    "Niue": "NIU",
    "Montserrat": "MSR",
    "Cook Islands": "COK",
    "Holy See (Vatican)": "VAT",
    "Holy See": "VAT",
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "Andorra": "AND",
    "Angola": "AGO", "Antigua and Barbuda": "ATG", "Argentina": "ARG",
    "Armenia": "ARM", "Australia": "AUS", "Austria": "AUT", "Azerbaijan": "AZE",
    "Bahamas": "BHS", "Bahrain": "BHR", "Bangladesh": "BGD", "Barbados": "BRB",
    "Belarus": "BLR", "Belgium": "BEL", "Belize": "BLZ", "Benin": "BEN",
    "Bhutan": "BTN", "Bolivia": "BOL", "Bosnia and Herzegovina": "BIH",
    "Botswana": "BWA", "Brazil": "BRA", "Brunei": "BRN", "Bulgaria": "BGR",
    "Burkina Faso": "BFA", "Burundi": "BDI", "Cabo Verde": "CPV",
    "Cambodia": "KHM", "Cameroon": "CMR", "Canada": "CAN",
    "Central African Republic": "CAF", "Chad": "TCD", "Chile": "CHL",
    "China": "CHN", "Colombia": "COL", "Comoros": "COM", "Congo": "COG",
    "Costa Rica": "CRI", "Croatia": "HRV", "Cuba": "CUB", "Cyprus": "CYP",
    "Czech Republic": "CZE", "Denmark": "DNK", "Djibouti": "DJI",
    "Dominica": "DMA", "Dominican Republic": "DOM", "Ecuador": "ECU",
    "Egypt": "EGY", "El Salvador": "SLV", "Equatorial Guinea": "GNQ",
    "Eritrea": "ERI", "Estonia": "EST", "Eswatini": "SWZ", "Ethiopia": "ETH",
    "Fiji": "FJI", "Finland": "FIN", "France": "FRA", "Gabon": "GAB",
    "Gambia": "GMB", "Georgia": "GEO", "Germany": "DEU", "Ghana": "GHA",
    "Greece": "GRC", "Grenada": "GRD", "Guatemala": "GTM", "Guinea": "GIN",
    "Guinea-Bissau": "GNB", "Guyana": "GUY", "Haiti": "HTI", "Honduras": "HND",
    "Hungary": "HUN", "Iceland": "ISL", "India": "IND", "Indonesia": "IDN",
    "Iraq": "IRQ", "Ireland": "IRL", "Israel": "ISR", "Italy": "ITA",
    "Jamaica": "JAM", "Japan": "JPN", "Jordan": "JOR", "Kazakhstan": "KAZ",
    "Kenya": "KEN", "Kiribati": "KIR", "Korea, North": "PRK",
    "Korea, South": "KOR", "Kuwait": "KWT", "Kyrgyzstan": "KGZ",
    "Latvia": "LVA", "Lebanon": "LBN", "Lesotho": "LSO", "Liberia": "LBR",
    "Libya": "LBY", "Liechtenstein": "LIE", "Lithuania": "LTU",
    "Luxembourg": "LUX", "Madagascar": "MDG", "Malawi": "MWI",
    "Malaysia": "MYS", "Maldives": "MDV", "Mali": "MLI", "Malta": "MLT",
    "Marshall Islands": "MHL", "Mauritania": "MRT", "Mauritius": "MUS",
    "Mexico": "MEX", "Micronesia": "FSM", "Monaco": "MCO", "Mongolia": "MNG",
    "Montenegro": "MNE", "Morocco": "MAR", "Mozambique": "MOZ",
    "Myanmar": "MMR", "Namibia": "NAM", "Nauru": "NRU", "Nepal": "NPL",
    "Netherlands": "NLD", "New Zealand": "NZL", "Nicaragua": "NIC",
    "Niger": "NER", "Nigeria": "NGA", "North Macedonia": "MKD",
    "Norway": "NOR", "Oman": "OMN", "Pakistan": "PAK", "Palau": "PLW",
    "Panama": "PAN", "Papua New Guinea": "PNG", "Paraguay": "PRY",
    "Peru": "PER", "Philippines": "PHL", "Poland": "POL", "Portugal": "PRT",
    "Qatar": "QAT", "Romania": "ROU", "Russian Federation": "RUS",
    "Rwanda": "RWA", "Saint Kitts and Nevis": "KNA", "Saint Lucia": "LCA",
    "Saint Vincent and the Grenadines": "VCT", "Samoa": "WSM",
    "San Marino": "SMR", "Sao Tome and Principe": "STP", "Saudi Arabia": "SAU",
    "Senegal": "SEN", "Serbia": "SRB", "Seychelles": "SYC",
    "Sierra Leone": "SLE", "Singapore": "SGP", "Slovakia": "SVK",
    "Slovenia": "SVN", "Solomon Islands": "SLB", "Somalia": "SOM",
    "South Africa": "ZAF", "South Sudan": "SSD", "Spain": "ESP",
    "Sri Lanka": "LKA", "Sudan": "SDN", "Suriname": "SUR", "Sweden": "SWE",
    "Switzerland": "CHE", "Taiwan": "TWN", "Tajikistan": "TJK",
    "Thailand": "THA", "Timor-Leste": "TLS", "Togo": "TGO", "Tonga": "TON",
    "Trinidad and Tobago": "TTO", "Tunisia": "TUN", "Turkey": "TUR",
    "Turkmenistan": "TKM", "Tuvalu": "TUV", "Uganda": "UGA", "Ukraine": "UKR",
    "United Arab Emirates": "ARE", "Uruguay": "URY", "Uzbekistan": "UZB",
    "Vanuatu": "VUT", "Vatican City": "VAT", "Yemen": "YEM", "Zambia": "ZMB",
    "Zimbabwe": "ZWE", "Holy See": "VAT", "Swaziland": "SWZ",
    "Macedonia": "MKD", "Burma": "MMR", "Ivory Coast": "CIV",
    "Cote d'Ivoire": "CIV", "Cape Verde": "CPV"
}

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
        report.append(" DATA QUALITY REPORT - DIPLOMATIC NETWORK ")
        report.append("="*70)
        report.append(f"\n初期行数: {self.initial_rows:,}")
        report.append(f"最終行数: {final_count:,}")
        report.append(f"保持率: {final_count/self.initial_rows*100:.2f}%")
        report.append(f"除外率: {(self.initial_rows-final_count)/self.initial_rows*100:.2f}%")
        
        report.append("\n" + "-"*70)
        report.append(" 除外理由の内訳 ")
        report.append("-"*70)
        
        total_removed = self.initial_rows - final_count
        for reason, data in sorted(self.removed.items(), 
                                   key=lambda x: x[1]['count'], 
                                   reverse=True):
            count = data['count']
            pct = count / self.initial_rows * 100
            report.append(f"\n【{reason}】")
            report.append(f"  除外行数: {count:,} ({pct:.2f}%)")
            report.append(f"  正当性: {'✓' if self._is_justified(reason) else '⚠'}")
            
            if data['details']:
                report.append(f"  詳細:")
                for detail in data['details'][:10]:  # 最初の10件
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
        report.append(self._evaluate_justification())
        
        return "\n".join(report)
    
    def _is_justified(self, reason):
        """除外理由が正当かチェック"""
        justified_reasons = [
            "時間範囲外（2000-2020年外）",
            "自己ループ（origin == destination）",
            "弱い外交関係（embassy_level < 4）",
            "重複エントリ",
            "エンコーディング問題（< 1%）"
        ]
        return any(j in reason for j in justified_reasons)
    
    def _evaluate_justification(self):
        """全体的な正当性を評価"""
        lines = []
        
        # チェック1: 除外率が妥当か
        exclusion_rate = (self.initial_rows - final_count) / self.initial_rows * 100
        if exclusion_rate < 20:
            lines.append("✓ 除外率が低い（< 20%）- 良好")
        elif exclusion_rate < 50:
            lines.append("⚠ 除外率がやや高い（20-50%）- 要確認")
        else:
            lines.append("✗ 除外率が高い（> 50%）- データソース確認推奨")
        
        # チェック2: ISO変換失敗率
        iso_failed = self.removed.get('ISO変換失敗', {}).get('count', 0)
        if iso_failed > 0:
            failure_rate = iso_failed / self.initial_rows * 100
            if failure_rate < 5:
                lines.append("✓ ISO変換失敗率が低い（< 5%）")
            else:
                lines.append(f"⚠ ISO変換失敗率が高い（{failure_rate:.1f}%）")
        
        # チェック3: 不正な除外理由がないか
        unjustified = [r for r in self.removed.keys() if not self._is_justified(r)]
        if unjustified:
            lines.append(f"⚠ 正当化できない除外: {unjustified}")
        else:
            lines.append("✓ すべての除外理由が正当化されている")
        
        return "\n".join(lines)

# Pycountry
try:
    import pycountry
    HAS_PYCOUNTRY = True
except ImportError:
    HAS_PYCOUNTRY = False

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

def main():
    global final_count
    
    tracker = DataQualityTracker()
    
    print("="*70)
    print(" DIPLOMATIC DATA PREPROCESSING WITH QUALITY TRACKING ")
    print("="*70)
    
    # 1. ロード
    if not os.path.exists(INPUT_FILE):
        alt_path = 'diplometrics_ddr.csv'
        input_path = alt_path if os.path.exists(alt_path) else None
        if not input_path:
            print(f"ERROR: Input file not found")
            sys.exit(1)
    else:
        input_path = INPUT_FILE
    
    print(f"\nLoading {input_path}...")
    df = pd.read_csv(input_path, encoding='utf-8', encoding_errors='replace', low_memory=False)
    
    tracker.set_initial(len(df))
    print(f"初期行数: {len(df):,}")
    
    # 2. エンコーディング問題チェック
    mask = df.astype(str).apply(lambda x: x.str.contains('\ufffd', na=False)).any(axis=1)
    problematic = df[mask]
    
    if len(problematic) > 0:
        pct = len(problematic) / len(df) * 100
        if pct < 1:
            tracker.record_removal(
                "エンコーディング問題（< 1%）",
                len(problematic),
                [f"Row {i}" for i in problematic.index[:5]]
            )
            df = df[~mask]
        else:
            tracker.add_warning(f"エンコーディング問題: {len(problematic)}行 ({pct:.2f}%)")
    
    # 3. 列リネーム
    col_map = {
        'Year': 'year', 'Sending Country': 'origin',
        'Destination': 'destination', 'Embassy': 'embassy_level',
        'Focus': 'focus'
    }
    df = df.rename(columns=col_map)
    df.columns = [c.strip() for c in df.columns]
    
    needed_cols = ['year', 'origin', 'destination', 'embassy_level', 'focus']
    available_cols = [c for c in needed_cols if c in df.columns]
    df = df[available_cols].copy()
    
    # 4. 時間フィルタ
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    pre_time = len(df)
    df = df.dropna(subset=['year'])
    df = df[(df['year'] >= START_YEAR) & (df['year'] <= END_YEAR)]
    tracker.record_removal(
        "時間範囲外（2000-2020年外）",
        pre_time - len(df)
    )
    
    # 5. ISO変換
    unconvertible = []
    
    def track_conversion(name):
        code = get_iso_code(name)
        if not code:
            unconvertible.append(name)
        return code
    
    df['origin_iso'] = df['origin'].apply(track_conversion)
    df['destination_iso'] = df['destination'].apply(track_conversion)
    
    pre_iso = len(df)
    df = df.dropna(subset=['origin_iso', 'destination_iso'])
    
    if unconvertible:
        unique_failed = list(set(unconvertible))
        tracker.record_removal(
            "ISO変換失敗",
            pre_iso - len(df),
            unique_failed[:20]
        )
        if len(unique_failed) > 10:
            tracker.add_warning(f"多数の国名変換失敗: {len(unique_failed)}カ国")
    
    df['origin'] = df['origin_iso']
    df['destination'] = df['destination_iso']
    df = df.drop(columns=['origin_iso', 'destination_iso'])
    
    # 6. 自己ループ
    pre_loops = len(df)
    df = df[df['origin'] != df['destination']]
    tracker.record_removal(
        "自己ループ（origin == destination）",
        pre_loops - len(df)
    )
    
    # 7. 弱い関係
    df['embassy_level'] = pd.to_numeric(df['embassy_level'], errors='coerce')
    df = df.dropna(subset=['embassy_level'])
    pre_weak = len(df)
    df = df[df['embassy_level'] >= 4]
    tracker.record_removal(
        "弱い外交関係（embassy_level < 4）",
        pre_weak - len(df)
    )
    
    # 8. 欠損値
    pre_na = len(df)
    df = df.dropna()
    tracker.record_removal("欠損値", pre_na - len(df))
    
    # 9. 重複
    pre_dup = len(df)
    df = df.sort_values('embassy_level', ascending=False)
    df = df.drop_duplicates(subset=['year', 'origin', 'destination'], keep='first')
    tracker.record_removal("重複エントリ", pre_dup - len(df))
    
    # 10. 最終処理
    df = df.sort_values(['year', 'origin', 'destination'])
    df['year'] = df['year'].astype(int)
    df['embassy_level'] = df['embassy_level'].astype(int)
    if 'focus' in df.columns:
        df['focus'] = df['focus'].astype(int)
    
    final_count = len(df)
    
    # レポート生成
    report = tracker.generate_report(final_count)
    print("\n" + report)
    
    # レポート保存
    with open(QUALITY_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n品質レポート保存: {QUALITY_REPORT_FILE}")
    
    # データ保存
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"データ保存: {OUTPUT_FILE}")
    print(f"\n✓ 完了")

if __name__ == "__main__":
    main()