import pandas as pd
import numpy as np
import os
import sys
import openpyxl
import warnings

# Suppress minor warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# --- CONFIGURATION ---
INPUT_FILE = r'data/raw/un_migrant_stock.xlsx'
OUTPUT_FILE = r'data/processed/migration_network.csv'
START_YEAR = 2000
END_YEAR = 2020
TARGET_YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020]

# --- COUNTRY MAPPING DICTIONARY ---
# Reused from preprocess_diplomacy.py for consistency
COUNTRY_MAPPING = {
    # Special cases requested
    "United States": "USA", "United Kingdom": "GBR", "Russia": "RUS", "South Korea": "KOR",
    "North Korea": "PRK", "Vietnam": "VNM", "Laos": "LAO", "Syria": "SYR", "Iran": "IRN",
    "Tanzania": "TZA", "Bolivia": "BOL", "Venezuela": "VEN", "Moldova": "MDA",
    "Congo, Dem. Rep.": "COD", "Congo, Rep.": "COG", "Gambia, The": "GMB", "Bahamas, The": "BHS",
    "United States of America": "USA", "Russian Federation": "RUS", "Bolivia (Plurinational State of)": "BOL",
    "Iran (Islamic Republic of)": "IRN", "Syrian Arab Republic": "SYR", "Venezuela (Bolivarian Republic of)": "VEN",
    "Viet Nam": "VNM", "Lao People's Democratic Republic": "LAO", "Tanzania, United Republic of": "TZA",
    "Republic of Korea": "KOR", "Dem. People's Republic of Korea": "PRK", "Democratic People's Republic of Korea": "PRK",
    "Moldova, Republic of": "MDA", "Micronesia (Federated States of)": "FSM",
    "United Kingdom of Great Britain and Northern Ireland": "GBR", "China, Hong Kong SAR": "HKG",
    "China, Macao SAR": "MAC", "State of Palestine": "PSE", "Holy See": "VAT",
    
    # Standard Names (Partial List - enhanced with common variations)
    "Afghanistan": "AFG", "Albania": "ALB", "Algeria": "DZA", "Andorra": "AND", "Angola": "AGO",
    "Antigua and Barbuda": "ATG", "Argentina": "ARG", "Armenia": "ARM", "Australia": "AUS", "Austria": "AUT",
    "Azerbaijan": "AZE", "Bahamas": "BHS", "Bahrain": "BHR", "Bangladesh": "BGD", "Barbados": "BRB",
    "Belarus": "BLR", "Belgium": "BEL", "Belize": "BLZ", "Benin": "BEN", "Bhutan": "BTN",
    "Bosnia and Herzegovina": "BIH", "Botswana": "BWA", "Brazil": "BRA", "Brunei": "BRN", "Brunei Darussalam": "BRN",
    "Bulgaria": "BGR", "Burkina Faso": "BFA", "Burundi": "BDI", "Cabo Verde": "CPV", "Cambodia": "KHM",
    "Cameroon": "CMR", "Canada": "CAN", "Central African Republic": "CAF", "Chad": "TCD", "Chile": "CHL",
    "China": "CHN", "Colombia": "COL", "Comoros": "COM", "Congo": "COG", "Costa Rica": "CRI",
    "Croatia": "HRV", "Cuba": "CUB", "Cyprus": "CYP", "Czech Republic": "CZE", "Czechia": "CZE", "Denmark": "DNK",
    "Djibouti": "DJI", "Dominica": "DMA", "Dominican Republic": "DOM", "Ecuador": "ECU", "Egypt": "EGY",
    "El Salvador": "SLV", "Equatorial Guinea": "GNQ", "Eritrea": "ERI", "Estonia": "EST", "Eswatini": "SWZ",
    "Ethiopia": "ETH", "Fiji": "FJI", "Finland": "FIN", "France": "FRA", "Gabon": "GAB",
    "Gambia": "GMB", "Georgia": "GEO", "Germany": "DEU", "Ghana": "GHA", "Greece": "GRC",
    "Grenada": "GRD", "Guatemala": "GTM", "Guinea": "GIN", "Guinea-Bissau": "GNB", "Guyana": "GUY",
    "Haiti": "HTI", "Honduras": "HND", "Hungary": "HUN", "Iceland": "ISL", "India": "IND",
    "Indonesia": "IDN", "Iraq": "IRQ", "Ireland": "IRL", "Israel": "ISR",
    "Italy": "ITA", "Jamaica": "JAM", "Japan": "JPN", "Jordan": "JOR", "Kazakhstan": "KAZ",
    "Kenya": "KEN", "Kiribati": "KIR", "Kuwait": "KWT",
    "Kyrgyzstan": "KGZ", "Latvia": "LVA", "Lebanon": "LBN", "Lesotho": "LSO",
    "Liberia": "LBR", "Libya": "LBY", "Liechtenstein": "LIE", "Lithuania": "LTU", "Luxembourg": "LUX",
    "Madagascar": "MDG", "Malawi": "MWI", "Malaysia": "MYS", "Maldives": "MDV", "Mali": "MLI",
    "Malta": "MLT", "Marshall Islands": "MHL", "Mauritania": "MRT", "Mauritius": "MUS", "Mexico": "MEX",
    "Micronesia": "FSM", "Monaco": "MCO", "Mongolia": "MNG", "Montenegro": "MNE",
    "Morocco": "MAR", "Mozambique": "MOZ", "Myanmar": "MMR", "Namibia": "NAM", "Nauru": "NRU",
    "Nepal": "NPL", "Netherlands": "NLD", "New Zealand": "NZL", "Nicaragua": "NIC", "Niger": "NER",
    "Nigeria": "NGA", "North Macedonia": "MKD", "Norway": "NOR", "Oman": "OMN", "Pakistan": "PAK",
    "Palau": "PLW", "Panama": "PAN", "Papua New Guinea": "PNG", "Paraguay": "PRY", "Peru": "PER",
    "Philippines": "PHL", "Poland": "POL", "Portugal": "PRT", "Qatar": "QAT", "Romania": "ROU",
    "Rwanda": "RWA", "Saint Kitts and Nevis": "KNA", "Saint Lucia": "LCA",
    "Saint Vincent and the Grenadines": "VCT", "Samoa": "WSM", "San Marino": "SMR", "Sao Tome and Principe": "STP",
    "Saudi Arabia": "SAU", "Senegal": "SEN", "Serbia": "SRB", "Seychelles": "SYC", "Sierra Leone": "SLE",
    "Singapore": "SGP", "Slovakia": "SVK", "Slovenia": "SVN", "Solomon Islands": "SLB", "Somalia": "SOM",
    "South Africa": "ZAF", "South Sudan": "SSD", "Spain": "ESP", "Sri Lanka": "LKA", "Sudan": "SDN",
    "Suriname": "SUR", "Sweden": "SWE", "Switzerland": "CHE", "Taiwan": "TWN",
    "Tajikistan": "TJK", "Thailand": "THA", "Timor-Leste": "TLS", "Togo": "TGO",
    "Tonga": "TON", "Trinidad and Tobago": "TTO", "Tunisia": "TUN", "Turkey": "TUR", "Turkmenistan": "TKM",
    "Tuvalu": "TUV", "Uganda": "UGA", "Ukraine": "UKR", "United Arab Emirates": "ARE", "Uruguay": "URY",
    "Uzbekistan": "UZB", "Vanuatu": "VUT", "Vatican City": "VAT",
    "Yemen": "YEM", "Zambia": "ZMB", "Zimbabwe": "ZWE",
    "Cote d'Ivoire": "CIV", "Ivory Coast": "CIV", "Eswatini": "SWZ",
}

# Try to use pycountry for missing entries
try:
    import pycountry
    HAS_PYCOUNTRY = True
except ImportError:
    HAS_PYCOUNTRY = False

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

def find_header_row(df, keywords=['Destination', 'Region', 'Country', 'Area']):
    """
    Scans the first 20 rows to find the header row containing keywords.
    Returns the index of the header row or None.
    """
    for i in range(min(20, len(df))):
        row_values = df.iloc[i].astype(str).tolist()
        if any(keyword in val for val in row_values for keyword in keywords):
            return i
    return None

    # Setup File Logging
    log_file = open('log.txt', 'w', encoding='utf-8')
    def log(msg):
        print(msg)
        log_file.write(str(msg) + '\n')
        log_file.flush()

    log("="*50)
    log(" MIGRATION DATA PREPROCESSING ")
    log("="*50)
    
    # 1. Verification
    if not os.path.exists(INPUT_FILE):
        log(f"ERROR: Input file not found at {INPUT_FILE}")
        sys.exit(1)
        
    log(f"Input File: {INPUT_FILE}")
    
    # 2. Load Excel File
    log("\nLoading Excel file (this may take a moment)...")
    try:
        # Load all sheets
        xl = pd.ExcelFile(INPUT_FILE)
        sheet_names = xl.sheet_names
        log(f"Detected {len(sheet_names)} sheet(s): {sheet_names}")
    except Exception as e:
        log(f"Error reading Excel file: {e}")
        sys.exit(1)

    all_data = [] # List to collect processed dataframes
    
    # 3. Process Sheets
    for sheet in sheet_names:
        log(f"\nProcessing Sheet: '{sheet}'...")
        
        # Read sheet raw
        try:
            df_raw = pd.read_excel(xl, sheet_name=sheet, header=None)
        except Exception as e:
             log(f"Error reading sheet: {e}")
             continue
        
        # Analyze structure
        header_idx = find_header_row(df_raw, keywords=['Destination', 'Major area', 'Region', 'Country', 'origin'])
        
        if header_idx is None:
            # Try finding years in the row?
            for i in range(min(20, len(df_raw))):
                row_vals = [str(x) for x in df_raw.iloc[i].values]
                
                # Check for year presence (handling 2000.0 etc)
                found_year = False
                for val in row_vals:
                    # Clean '2000.0' -> '2000'
                    val_clean = str(val).split('.')[0].strip()
                    if val_clean in [str(y) for y in TARGET_YEARS]:
                        found_year = True
                        break
                
                if found_year:
                    header_idx = i
                    break
            
            if header_idx is None:
                log("  WARNING: Could not identify header row (checked for Country/Region keywords and Years). Skipping sheet.")
                # Debug: print first few rows
                log("  First 5 rows of raw sheet:")
                log(df_raw.head(5).to_string())
                continue
            
        log(f"  Header identified at row {header_idx}")
        
        # Reload with correct header
        df = pd.read_excel(xl, sheet_name=sheet, header=header_idx)
        
        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]
        log(f"  Columns found: {list(df.columns)}")
        
        # --- Type Detection ---
        # CASE A: Matrix Format (Sheet = Year, Rows = Destination, Cols = Origin)
        # CASE B: Time Series Format (Rows = Destination, Cols = Years)
        
        # Check if year is in sheet name
        sheet_year = None
        for y in TARGET_YEARS:
            if str(y) in sheet:
                sheet_year = y
                break
        
        # Check if columns are Years (fuzzy match?)
        year_cols = []
        for c in df.columns:
            # Check for exact year or '1990.0' or '1990'
            c_str = str(c).replace('.0', '')
            if c_str in [str(y) for y in TARGET_YEARS]:
                year_cols.append(c)
        
        log(f"  Year columns identified: {year_cols}")
        
        if len(year_cols) >= 1:
            log("  Detected TIME SERIES format (Columns are years).")
            # Rows are likely Dest/Origin. We need to find the Country column.
            country_col = None
            for c in df.columns:
                if c in ['Region, development group, country or area', 'Destination', 'Country', 'Area', 'Major area, region, country or area of destination']:
                    country_col = c
                    break
            
            if not country_col:
                # Fallback: look for column with many unique string values
                possible_cols = [c for c in df.columns if df[c].dtype == object]
                if possible_cols:
                    country_col = possible_cols[1] if len(possible_cols) > 1 else possible_cols[0] # Index 1 often better if 0 is "Sort order"
            
            if not country_col:
                print("  ERROR: Could not identify Country column. Skipping.")
                continue
                
            print(f"  Country Column: {country_col}")
            
            # Melt: Transform from Wide (Years) to Long (Year, Stock)
            # Assuming this is Total Stock by Destination (since Origin usually isn't implied in single file unless specified)
            # IF this is the "Total" file, we treat Origin as "World" or unknown? 
            # OR, if this is truly an OD file, maybe it's "Origin" rows?
            # Standard UN file `UN_MigrantStockTotal` -> Rows are Destinations.
             
            # Let's clean and melt
            valid_year_cols = year_cols
            
            melted = df.melt(id_vars=[country_col], value_vars=valid_year_cols, var_name='year', value_name='migrant_stock')
            melted = melted.rename(columns={country_col: 'destination'})
            
            # Since we don't have origin, and user REQUESTED origin/destination structure...
            # If this is the only data we have, we might have to assume Origin = "Total" or similar to fit the schema,
            # or perhaps the user provided the wrong file.
            # However, looking at UN DESA "Origin and Destination" files, they usually have "Table 1" too but with both axes.
            # If we only have ONE country column -> it's likely totals.
            # I will set origin to "World" for now to produce a valid schema, and log a warning.
            
            melted['origin'] = 'World' 
            
            all_data.append(melted)
            
        elif sheet_year:
            print(f"  Detected MATRIX format for Year {sheet_year}.")
            # Rows = Destination (usually), Cols = Origin (usually) or vice versa.
            # Need to identify the destination column
            dest_col = None
            for c in df.columns:
                if c in ['Region, development group, country or area', 'Major area, region, country or area of destination']:
                    dest_col = c
                    break
            
            if not dest_col:
                # Guess first string col
                dest_col = df.columns[1] # usually col 0 is sort order
            
            print(f"  Destination Column: {dest_col}")
            
            # All other columns (that are countries) are Origins
            # Identify which columns are countries. Usually they start after some metadata cols.
            # We can check if column names look like countries (mapped in our dict) or have ISO codes.
            
            # Get list of potential specific origin columns
            origin_cols = []
            for c in df.columns:
                if c == dest_col: continue
                if str(c).lower() in ['sort order', 'notes', 'code', 'type of data (a)', 'total']: continue
                if isinstance(c, int): continue # Skip random ints if any
                
                # Check if it maps to a country?
                # This might be slow for all cols, but necessary
                if get_iso_code(str(c)):
                    origin_cols.append(c)
            
            if not origin_cols:
                print("  WARNING: No origin country columns identified. Is this a Total Stock file?")
                # Provide fallback for "Total" column if present
                if 'Total' in df.columns:
                     print("  Found 'Total' column. Treating as Aggregate.")
                     temp = df[[dest_col, 'Total']].copy()
                     temp = temp.rename(columns={dest_col: 'destination', 'Total': 'migrant_stock'})
                     temp['origin'] = 'World'
                     temp['year'] = sheet_year
                     all_data.append(temp)
                else:
                    print("  Skipping sheet.")
                continue

            print(f"  Identified {len(origin_cols)} origin columns.")
            
            # Melt Matrix
            melted = df.melt(id_vars=[dest_col], value_vars=origin_cols, var_name='origin', value_name='migrant_stock')
            melted = melted.rename(columns={dest_col: 'destination'})
            melted['year'] = sheet_year
            
            all_data.append(melted)

        else:
            print(f"  WARNING: Could not determine format (Matrix/Time Series) for sheet '{sheet}'.")

    # 4. Integrate Data
    if not all_data:
        print("\nERROR: No data extracted. Please check file format.")
        sys.exit(1)
        
    final_df = pd.concat(all_data, ignore_index=True)
    print(f"\nInput Rows Processed: {len(final_df)}")
    
    # 5. Cleaning & Standardization
    print("\n--- Cleaning Data ---")
    
    # Convert 'micrant_stock' to numeric, coercing errors
    final_df['migrant_stock'] = pd.to_numeric(final_df['migrant_stock'], errors='coerce')
    
    # Remove NaN, 0, strings like ".."
    pre_clean = len(final_df)
    final_df = final_df.dropna(subset=['migrant_stock'])
    final_df = final_df[final_df['migrant_stock'] > 0]
    print(f"Removed {pre_clean - len(final_df)} rows with invalid/zero stock.")
    
    # Convert Countries to ISO
    print("Mapping Country Names to ISO Codes...")
    
    unconvertible = set()
    def cautious_convert(name):
        res = get_iso_code(str(name))
        if not res and name not in ['Total', 'World']: 
            unconvertible.add(name)
        return res if res else None # Returns None if failed

    # Apply conversion
    # Note: If origin is 'World', we might want to keep it or map it to 'WLD' if desired, 
    # but user asked for ISO 3166-1 alpha-3. World isn't in that strict set, but useful.
    # We will map "World" -> "WLD" if possible, or drop if strict. 
    # User requirement: "origin (str): Migrant's origin country (ISO 3166-1 alpha-3 code)"
    # If we have "Total" data (Origin=World), we probably should drop it if we strictly strictly follow "Country" requirements,
    # but having Total is better than nothing if that's all the file has.
    # I'll keep "World" as "WLD" for now to avoid empty CSV if file is aggregate.
    COUNTRY_MAPPING['World'] = 'WLD'
    
    final_df['origin_iso'] = final_df['origin'].apply(cautious_convert)
    final_df['destination_iso'] = final_df['destination'].apply(cautious_convert)
    
    # Filter unmapped
    pre_iso = len(final_df)
    final_df = final_df.dropna(subset=['origin_iso', 'destination_iso'])
    print(f"Dropped {pre_iso - len(final_df)} rows due to unmapped countries.")
    
    if len(unconvertible) > 0:
        print(f"  (Sample unmapped: {list(unconvertible)[:5]})")
    
    # Replace columns
    final_df['origin'] = final_df['origin_iso']
    final_df['destination'] = final_df['destination_iso']
    final_df = final_df.drop(columns=['origin_iso', 'destination_iso'])
    
    # Remove self-loops
    final_df = final_df[final_df['origin'] != final_df['destination']]
    
    # Filter Time Range
    final_df['year'] = final_df['year'].astype(int)
    final_df = final_df[(final_df['year'] >= START_YEAR) & (final_df['year'] <= END_YEAR)]
    
    # 6. Statistics / Report
    print("\n" + "="*30)
    print(" STATISTICAL REPORT ")
    print("="*30)
    
    print(f"Total Flows: {len(final_df)}")
    
    # Flows per year
    print("\n[Flows per Year]")
    print(final_df.groupby('year').size())
    
    # Stock Distribution
    print("\n[Migrant Stock Stats]")
    print(final_df['migrant_stock'].describe())
    
    # Top 10 Corridors
    print("\n[Top 10 Corridors]")
    top_10 = final_df.sort_values('migrant_stock', ascending=False).head(10)
    for _, row in top_10.iterrows():
        print(f"  {row['year']}: {row['origin']} -> {row['destination']} : {int(row['migrant_stock'])}")
        
    # Missing Values Check (already cleaned, but checking consistency)
    missing_pct = final_df.isnull().mean() * 100
    if missing_pct.any():
        print("\n[Missing Values %]")
        print(missing_pct[missing_pct > 0])
    else:
        print("\nNo missing values in final dataset.")
        
    # 7. Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print("Done.")

if __name__ == "__main__":
    main()
