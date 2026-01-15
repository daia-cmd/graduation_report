import pandas as pd
import numpy as np
import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=UserWarning)

# --- CONFIGURATION ---
INPUT_FILE = r'data/raw/un_migrant_stock.xlsx'
OUTPUT_FILE = r'data/processed/migration_network.csv'
SHEET_NAME = 'Table 1'
HEADER_ROW = 10
TARGET_YEARS = [2000, 2005, 2010, 2015, 2020]

# --- COUNTRY MAPPING DICTIONARY ---
# Reused from previous scripts for consistency
COUNTRY_MAPPING = {
    # Special cases
    "United States of America": "USA", "United Kingdom": "GBR", "Russian Federation": "RUS", 
    "Republic of Korea": "KOR", "Democratic People's Republic of Korea": "PRK",
    "Viet Nam": "VNM", "Lao People's Democratic Republic": "LAO", "Syrian Arab Republic": "SYR",
    "Iran (Islamic Republic of)": "IRN", "United Republic of Tanzania": "TZA",
    "Bolivia (Plurinational State of)": "BOL", "Venezuela (Bolivarian Republic of)": "VEN",
    "Republic of Moldova": "MDA", "Democratic Republic of the Congo": "COD", "Congo": "COG",
    "Gambia": "GMB", "Bahamas": "BHS", "China, Hong Kong SAR": "HKG", "China, Macao SAR": "MAC",
    "State of Palestine": "PSE", "Holy See": "VAT", "Türkiye": "TUR", "Czechia": "CZE",
    "Côte d'Ivoire": "CIV", "Réunion": "REU", "Curaçao": "CUW",
    
    # Standard country names
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "Andorra": "AND", "Angola": "AGO",
    "Antigua and Barbuda": "ATG", "Argentina": "ARG", "Armenia": "ARM", "Australia": "AUS", "Austria": "AUT",
    "Azerbaijan": "AZE", "Bahrain": "BHR", "Bangladesh": "BGD", "Barbados": "BRB",
    "Belarus": "BLR", "Belgium": "BEL", "Belize": "BLZ", "Benin": "BEN", "Bhutan": "BTN",
    "Bolivia": "BOL", "Bosnia and Herzegovina": "BIH", "Botswana": "BWA", "Brazil": "BRA", 
    "Brunei Darussalam": "BRN", "Bulgaria": "BGR", "Burkina Faso": "BFA", "Burundi": "BDI",
    "Cabo Verde": "CPV", "Cambodia": "KHM", "Cameroon": "CMR", "Canada": "CAN",
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
    "Malta": "MLT", "Marshall Islands": "MHL", "Mauritania": "MRT", "Mauritius": "MUS", "Mexico": "MEX",
    "Micronesia (Federated States of)": "FSM", "Monaco": "MCO", "Mongolia": "MNG", "Montenegro": "MNE",
    "Morocco": "MAR", "Mozambique": "MOZ", "Myanmar": "MMR", "Namibia": "NAM", "Nauru": "NRU",
    "Nepal": "NPL", "Netherlands": "NLD", "New Zealand": "NZL", "Nicaragua": "NIC", "Niger": "NER",
    "Nigeria": "NGA", "North Macedonia": "MKD", "Norway": "NOR", "Oman": "OMN", "Pakistan": "PAK",
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

# Regional/aggregate keywords to exclude
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

# Try pycountry
try:
    import pycountry
    HAS_PYCOUNTRY = True
except ImportError:
    HAS_PYCOUNTRY = False

def get_iso_code(country_name):
    """Convert country name to ISO 3-letter code"""
    if not isinstance(country_name, str):
        return None
    
    # Clean: strip whitespace and asterisks
    clean_name = country_name.strip().rstrip('*').strip()
    
    # Direct lookup
    if clean_name in COUNTRY_MAPPING:
        return COUNTRY_MAPPING[clean_name]
    
    # Pycountry fallback
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
    """Check if name is a regional/aggregate identifier"""
    if not isinstance(name, str):
        return True
    name_clean = str(name).strip()
    for keyword in REGIONAL_KEYWORDS:
        if keyword.lower() in name_clean.lower():
            return True
    return False

def main():
    print("="*60)
    print(" UN DESA MIGRATION DATA PREPROCESSING ")
    print("="*60)
    
    # 1. Load Excel
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found at {INPUT_FILE}")
        sys.exit(1)
    
    print(f"\nLoading {INPUT_FILE}...")
    try:
        df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, header=HEADER_ROW)
    except Exception as e:
        print(f"ERROR loading file: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} rows")
    
    # 2. Extract relevant columns
    dest_col = 'Region, development group, country or area of destination'
    origin_col = 'Region, development group, country or area of origin'
    
    # Check columns exist
    if dest_col not in df.columns or origin_col not in df.columns:
        print(f"ERROR: Required columns not found")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    # Select columns
    cols_to_keep = [dest_col, origin_col] + TARGET_YEARS
    df_subset = df[cols_to_keep].copy()
    
    # 3. Filter to individual countries only
    print(f"\nFiltering regional aggregates...")
    initial_rows = len(df_subset)
    
    df_countries = df_subset[
        ~df_subset[dest_col].apply(is_regional_name) &
        ~df_subset[origin_col].apply(is_regional_name)
    ].copy()
    
    regional_removed = initial_rows - len(df_countries)
    print(f"Removed {regional_removed} regional/aggregate rows")
    print(f"Remaining country pairs: {len(df_countries)}")
    
    # 4. Clean country names (strip asterisks)
    print(f"\nCleaning country names...")
    df_countries[dest_col] = df_countries[dest_col].str.strip().str.rstrip('*').str.strip()
    df_countries[origin_col] = df_countries[origin_col].str.strip().str.rstrip('*').str.strip()
    
    # 5. Melt to long format
    print(f"\nConverting to long format...")
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
    
    print(f"Long format rows: {len(df_long)}")
    
    # 6. Data cleaning
    print(f"\nCleaning data...")
    
    # Convert to numeric
    df_long['migrant_stock'] = pd.to_numeric(df_long['migrant_stock'], errors='coerce')
    df_long['year'] = df_long['year'].astype(int)
    
    # Remove invalid stocks
    pre_clean = len(df_long)
    df_long = df_long.dropna(subset=['migrant_stock'])
    df_long = df_long[df_long['migrant_stock'] > 0]
    print(f"Removed {pre_clean - len(df_long)} rows with invalid/zero stock")
    
    # 7. Convert to ISO codes
    print(f"\nConverting country names to ISO codes...")
    
    unconvertible_dest = set()
    unconvertible_origin = set()
    
    def convert_and_track_dest(name):
        code = get_iso_code(name)
        if not code:
            unconvertible_dest.add(name)
        return code
    
    def convert_and_track_origin(name):
        code = get_iso_code(name)
        if not code:
            unconvertible_origin.add(name)
        return code
    
    df_long['destination_iso'] = df_long['destination'].apply(convert_and_track_dest)
    df_long['origin_iso'] = df_long['origin'].apply(convert_and_track_origin)
    
    # Report unconvertible
    if unconvertible_dest or unconvertible_origin:
        print(f"\nWARNING: Some countries could not be converted to ISO codes")
        if unconvertible_dest:
            print(f"  Destination: {list(unconvertible_dest)[:10]}")
        if unconvertible_origin:
            print(f"  Origin: {list(unconvertible_origin)[:10]}")
    
    # Filter out failed conversions
    pre_iso = len(df_long)
    df_long = df_long.dropna(subset=['destination_iso', 'origin_iso'])
    print(f"Dropped {pre_iso - len(df_long)} rows due to ISO conversion failure")
    
    # Replace with ISO codes
    df_long['destination'] = df_long['destination_iso']
    df_long['origin'] = df_long['origin_iso']
    df_long = df_long.drop(columns=['destination_iso', 'origin_iso'])
    
    # 8. Remove self-loops
    pre_loops = len(df_long)
    df_long = df_long[df_long['origin'] != df_long['destination']]
    print(f"Removed {pre_loops - len(df_long)} self-loops")
    
    # 9. Final formatting
    df_long = df_long[['year', 'origin', 'destination', 'migrant_stock']]
    df_long = df_long.sort_values(['year', 'origin', 'destination'])
    df_long['migrant_stock'] = df_long['migrant_stock'].astype(int)
    
    # 10. Statistics
    print("\n" + "="*60)
    print(" STATISTICAL REPORT ")
    print("="*60)
    
    print(f"\nTotal migration flows: {len(df_long)}")
    
    print(f"\n[Flows per Year]")
    print(df_long.groupby('year').size())
    
    print(f"\n[Migrant Stock Distribution]")
    print(df_long['migrant_stock'].describe())
    
    print(f"\n[Top 20 Migration Corridors - 2020]")
    top_2020 = df_long[df_long['year'] == 2020].nlargest(20, 'migrant_stock')
    for idx, row in top_2020.iterrows():
        print(f"  {row['origin']} → {row['destination']}: {row['migrant_stock']:,}")
    
    print(f"\n[Top 10 Destination Countries - 2020]")
    top_dest = df_long[df_long['year'] == 2020].groupby('destination')['migrant_stock'].sum().nlargest(10)
    print(top_dest)
    
    print(f"\n[Top 10 Origin Countries - 2020]")
    top_origin = df_long[df_long['year'] == 2020].groupby('origin')['migrant_stock'].sum().nlargest(10)
    print(top_origin)
    
    # 11. Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_long.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    
    print("\n✓ Done!")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Final rows: {len(df_long)}")

if __name__ == "__main__":
    main()