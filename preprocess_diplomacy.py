import pandas as pd
import numpy as np
import os
import sys

# --- CONFIGURATION ---
INPUT_FILE = r'data/raw/diplometrics_ddr.csv'
OUTPUT_FILE = 'diplomatic_network.csv'
START_YEAR = 2000
END_YEAR = 2020

# --- COUNTRY MAPPING DICTIONARY ---
# Comprehensive mapping including historical and special cases
COUNTRY_MAPPING = {
    # Special cases requested
    "United States": "USA",
    "United Kingdom": "GBR",
    "Russia": "RUS",
    "South Korea": "KOR",
    "North Korea": "PRK",
    "Vietnam": "VNM",
    "Laos": "LAO",
    "Syria": "SYR",
    "Iran": "IRN",
    "Tanzania": "TZA",
    "Bolivia": "BOL",
    "Venezuela": "VEN",
    "Moldova": "MDA",
    "Congo, Dem. Rep.": "COD",
    "Congo, Rep.": "COG",
    "Gambia, The": "GMB",
    "Bahamas, The": "BHS",
    
    # Historical
    "USSR": "RUS", # Mapping USSR to Russia for continuity if desired, or use 'SUN'
    "Soviet Union": "RUS",
    "Yugoslavia": "SRB", # Often mapped to Serbia, or 'YUG'
    "Czechoslovakia": "CZE", # or 'CSK', mapping to Czechia usually
    "East Germany": "DEU", # Mapped to Germany
    "West Germany": "DEU",
    
    # Standard Names (Partial List - enhanced with common variations)
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "Andorra": "AND", "Angola": "AGO",
    "Antigua and Barbuda": "ATG", "Argentina": "ARG", "Armenia": "ARM", "Australia": "AUS", "Austria": "AUT",
    "Azerbaijan": "AZE", "Bahamas": "BHS", "Bahrain": "BHR", "Bangladesh": "BGD", "Barbados": "BRB",
    "Belarus": "BLR", "Belgium": "BEL", "Belize": "BLZ", "Benin": "BEN", "Bhutan": "BTN",
    "Bolivia": "BOL", "Bosnia and Herzegovina": "BIH", "Botswana": "BWA", "Brazil": "BRA", "Brunei": "BRN",
    "Bulgaria": "BGR", "Burkina Faso": "BFA", "Burundi": "BDI", "Cabo Verde": "CPV", "Cambodia": "KHM",
    "Cameroon": "CMR", "Canada": "CAN", "Central African Republic": "CAF", "Chad": "TCD", "Chile": "CHL",
    "China": "CHN", "Colombia": "COL", "Comoros": "COM", "Congo": "COG", "Costa Rica": "CRI",
    "Croatia": "HRV", "Cuba": "CUB", "Cyprus": "CYP", "Czech Republic": "CZE", "Denmark": "DNK",
    "Djibouti": "DJI", "Dominica": "DMA", "Dominican Republic": "DOM", "Ecuador": "ECU", "Egypt": "EGY",
    "El Salvador": "SLV", "Equatorial Guinea": "GNQ", "Eritrea": "ERI", "Estonia": "EST", "Eswatini": "SWZ",
    "Ethiopia": "ETH", "Fiji": "FJI", "Finland": "FIN", "France": "FRA", "Gabon": "GAB",
    "Gambia": "GMB", "Georgia": "GEO", "Germany": "DEU", "Ghana": "GHA", "Greece": "GRC",
    "Grenada": "GRD", "Guatemala": "GTM", "Guinea": "GIN", "Guinea-Bissau": "GNB", "Guyana": "GUY",
    "Haiti": "HTI", "Honduras": "HND", "Hungary": "HUN", "Iceland": "ISL", "India": "IND",
    "Indonesia": "IDN", "Iran": "IRN", "Iraq": "IRQ", "Ireland": "IRL", "Israel": "ISR",
    "Italy": "ITA", "Jamaica": "JAM", "Japan": "JPN", "Jordan": "JOR", "Kazakhstan": "KAZ",
    "Kenya": "KEN", "Kiribati": "KIR", "Korea, North": "PRK", "Korea, South": "KOR", "Kuwait": "KWT",
    "Kyrgyzstan": "KGZ", "Laos": "LAO", "Latvia": "LVA", "Lebanon": "LBN", "Lesotho": "LSO",
    "Liberia": "LBR", "Libya": "LBY", "Liechtenstein": "LIE", "Lithuania": "LTU", "Luxembourg": "LUX",
    "Madagascar": "MDG", "Malawi": "MWI", "Malaysia": "MYS", "Maldives": "MDV", "Mali": "MLI",
    "Malta": "MLT", "Marshall Islands": "MHL", "Mauritania": "MRT", "Mauritius": "MUS", "Mexico": "MEX",
    "Micronesia": "FSM", "Moldova": "MDA", "Monaco": "MCO", "Mongolia": "MNG", "Montenegro": "MNE",
    "Morocco": "MAR", "Mozambique": "MOZ", "Myanmar": "MMR", "Namibia": "NAM", "Nauru": "NRU",
    "Nepal": "NPL", "Netherlands": "NLD", "New Zealand": "NZL", "Nicaragua": "NIC", "Niger": "NER",
    "Nigeria": "NGA", "North Macedonia": "MKD", "Norway": "NOR", "Oman": "OMN", "Pakistan": "PAK",
    "Palau": "PLW", "Panama": "PAN", "Papua New Guinea": "PNG", "Paraguay": "PRY", "Peru": "PER",
    "Philippines": "PHL", "Poland": "POL", "Portugal": "PRT", "Qatar": "QAT", "Romania": "ROU",
    "Russian Federation": "RUS", "Rwanda": "RWA", "Saint Kitts and Nevis": "KNA", "Saint Lucia": "LCA",
    "Saint Vincent and the Grenadines": "VCT", "Samoa": "WSM", "San Marino": "SMR", "Sao Tome and Principe": "STP",
    "Saudi Arabia": "SAU", "Senegal": "SEN", "Serbia": "SRB", "Seychelles": "SYC", "Sierra Leone": "SLE",
    "Singapore": "SGP", "Slovakia": "SVK", "Slovenia": "SVN", "Solomon Islands": "SLB", "Somalia": "SOM",
    "South Africa": "ZAF", "South Sudan": "SSD", "Spain": "ESP", "Sri Lanka": "LKA", "Sudan": "SDN",
    "Suriname": "SUR", "Sweden": "SWE", "Switzerland": "CHE", "Syria": "SYR", "Taiwan": "TWN",
    "Tajikistan": "TJK", "Tanzania": "TZA", "Thailand": "THA", "Timor-Leste": "TLS", "Togo": "TGO",
    "Tonga": "TON", "Trinidad and Tobago": "TTO", "Tunisia": "TUN", "Turkey": "TUR", "Turkmenistan": "TKM",
    "Tuvalu": "TUV", "Uganda": "UGA", "Ukraine": "UKR", "United Arab Emirates": "ARE", "Uruguay": "URY",
    "Uzbekistan": "UZB", "Vanuatu": "VUT", "Vatican City": "VAT", "Venezuela": "VEN", "Vietnam": "VNM",
    "Yemen": "YEM", "Zambia": "ZMB", "Zimbabwe": "ZWE",
    "Holy See": "VAT", "Swaziland": "SWZ", "Macedonia": "MKD", "Burma": "MMR", "Ivory Coast": "CIV",
    "Cote d'Ivoire": "CIV", "Cape Verde": "CPV"
}

# Try to use pycountry for missing entries
try:
    import pycountry
    HAS_PYCOUNTRY = True
    print("Pycountry library detected. It will be used for fallback lookups.")
except ImportError:
    HAS_PYCOUNTRY = False
    print("Pycountry not found. Using internal dictionary only.")

def get_iso_code(country_name):
    """
    Convert country name to ISO 3-letter code.
    Prioritizes dictionary mapping, then fuzzy/pycountry lookup if available.
    """
    if not isinstance(country_name, str):
        return None
        
    clean_name = country_name.strip()
    
    # 1. Direct dictionary lookup
    if clean_name in COUNTRY_MAPPING:
        return COUNTRY_MAPPING[clean_name]
        
    # 2. Pycountry fallback
    if HAS_PYCOUNTRY:
        try:
            # Try exact match first
            country = pycountry.countries.get(name=clean_name)
            if country:
                return country.alpha_3
            
            # Try fuzzy search
            matches = pycountry.countries.search_fuzzy(clean_name)
            if matches:
                return matches[0].alpha_3
        except (LookupError, AttributeError):
            pass
            
    return None

def main():
    print(f"Starting processing...")
    print(f"Looking for data in: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found at {INPUT_FILE}")
        # Try finding it in current dir just in case
        alt_path = 'diplometrics_ddr.csv'
        if os.path.exists(alt_path):
            print(f"Found at {alt_path}, using that instead.")
            input_path = alt_path
        else:
            sys.exit(1)
    else:
        input_path = INPUT_FILE
        
    # 1. Load Data
    print("Loading CSV...")
    try:
        df = pd.read_csv(input_path, encoding='utf-8', encoding_errors='replace')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
        
    initial_rows = len(df)
    print(f"Loaded {initial_rows} rows.")
    
    # Step 2: Check for problematic rows with encoding issues
    print("\nChecking for encoding issues...")
    mask = df.astype(str).apply(lambda x: x.str.contains('\ufffd', na=False)).any(axis=1)
    problematic_rows = df[mask]
    
    print(f"Problematic rows with encoding issues: {len(problematic_rows)}")
    if len(problematic_rows) > 0:
        print(f"Percentage: {len(problematic_rows)/len(df)*100:.2f}%")
    
    # Step 3: Remove if minor issue
    if len(problematic_rows) / len(df) < 0.01:  # Less than 1%
        print("✓ Removing problematic rows (< 1% of data)")
        df = df[~mask]
        print(f"Rows after removing encoding issues: {len(df)}")
    elif len(problematic_rows) > 0:
        print("⚠ WARNING: Many rows have encoding issues, but continuing...")
    else:
        print("✓ No encoding issues detected.")
    
    # 2. Rename Columns
    # Expected: Destination, Sending Country, Year, Location, Embassy, Focus, EmbassyFocus, LOR
    # Map to: year, origin, destination, embassy_level, focus
    
    # Check current columns
    print(f"\nColumns: {list(df.columns)}")
    
    # Flexible mapping to handle potential variations
    col_map = {
        'Year': 'year',
        'Sending Country': 'origin',
        'Destination': 'destination',
        'Embassy': 'embassy_level',
        'Focus': 'focus' 
    }
    
    # Verify columns exist
    missing_cols = [c for c in col_map.keys() if c not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing expected columns: {missing_cols}")
        # Try to guess? Or fail? Let's try to proceed if critical ones exist
        if 'year' not in df.columns and 'Year' not in df.columns:
            print("Critical column 'Year' missing.")
            sys.exit(1)
            
    df = df.rename(columns=col_map)
    
    # Keep only needed columns
    needed_cols = ['year', 'origin', 'destination', 'embassy_level', 'focus']
    # If some are missing (e.g. 'focus'), fill with default or drop?
    # User said "Expected columns: ... Focus ...", so assume it's there.
    
    # Handle case where columns might not have been renamed if keys didn't match exactly
    # (e.g. extra spaces). Let's clean column names first.
    df.columns = [c.strip() for c in df.columns]
    # Re-apply rename with stripped keys if needed, but the map above used clean keys 
    # assuming the CSV has clean keys. 
    
    current_cols = df.columns.tolist()
    available_cols = [c for c in needed_cols if c in current_cols]
    df = df[available_cols].copy()
    
    # 3. Filter Time Range
    print(f"\nFiltering years {START_YEAR}-{END_YEAR}...")
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year']) 
    df = df[(df['year'] >= START_YEAR) & (df['year'] <= END_YEAR)]
    print(f"Rows after year filter: {len(df)}")
    
    # 4. Country Conversion
    print("\nConverting country names to ISO codes...")
    
    # Track unconvertible
    unconvertible = set()
    
    def convert_with_track(name):
        code = get_iso_code(name)
        if not code:
            unconvertible.add(name)
        return code
        
    df['origin_iso'] = df['origin'].apply(convert_with_track)
    df['destination_iso'] = df['destination'].apply(convert_with_track)
    
    # Report unconvertible
    if unconvertible:
        print(f"WARNING: {len(unconvertible)} country names could not be converted.")
        print(f"Sample of unconvertible: {list(unconvertible)[:10]}")
    
    # Drop rows where conversion failed
    rows_pre_iso_drop = len(df)
    df = df.dropna(subset=['origin_iso', 'destination_iso'])
    print(f"Dropped {rows_pre_iso_drop - len(df)} rows due to invalid country codes.")
    
    # Replace columns
    df['origin'] = df['origin_iso']
    df['destination'] = df['destination_iso']
    df = df.drop(columns=['origin_iso', 'destination_iso'])
    
    # 5. Cleaning
    print("\nCleaning data...")
    rows_step5_start = len(df)
    
    # Remove self-loops
    df = df[df['origin'] != df['destination']]
    loops_removed = rows_step5_start - len(df)
    print(f"Removed {loops_removed} self-loops.")
    
    # Remove weak relations (embassy_level < 4)
    rows_pre_level = len(df)
    df['embassy_level'] = pd.to_numeric(df['embassy_level'], errors='coerce')
    df = df.dropna(subset=['embassy_level'])
    df = df[df['embassy_level'] >= 4]
    weak_removed = rows_pre_level - len(df)
    print(f"Removed {weak_removed} weak relations (level < 4).")
    
    # Remove rows with any remaining missing values
    rows_pre_na = len(df)
    df = df.dropna()
    na_removed = rows_pre_na - len(df)
    print(f"Removed {na_removed} rows with missing values.")
    
    # 6. Deduplication
    print("\nHandling duplicates...")
    # Keep row with max embassy_level for same (year, origin, destination)
    # Sort by embassy_level desc, then drop duplicates keeping first
    rows_pre_dedup = len(df)
    df = df.sort_values('embassy_level', ascending=False)
    df = df.drop_duplicates(subset=['year', 'origin', 'destination'], keep='first')
    duplicates_removed = rows_pre_dedup - len(df)
    print(f"Removed {duplicates_removed} duplicate entries.")
    
    # 7. Final Formatting
    df = df.sort_values(['year', 'origin', 'destination'])
    # Ensure types
    df['year'] = df['year'].astype(int)
    df['embassy_level'] = df['embassy_level'].astype(int)
    if 'focus' in df.columns:
        df['focus'] = df['focus'].astype(int)
    
    # 8. Statistics
    print("\n" + "="*30)
    print("FINAL STATISTICS")
    print("="*30)
    print(f"Original Rows: {initial_rows}")
    print(f"Final Rows: {len(df)}")
    print(f"Percent retained: {len(df)/initial_rows*100:.2f}%")
    
    print("\n--- Edges per Year ---")
    year_stats = df.groupby('year').size().describe()
    print(year_stats[['mean', 'min', 'max']])
    
    print("\n--- Top 10 Countries by Missions (Origin) ---")
    top_origins = df.groupby('origin').size().sort_values(ascending=False).head(10)
    print(top_origins)
    
    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print("Done.")

if __name__ == "__main__":
    main()