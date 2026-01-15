import pandas as pd
import numpy as np
import os
import sys

# --- CONFIGURATION ---
ROUTES_FILE = r'data/raw/routes.dat' # Or openflights_routes.csv
AIRPORTS_FILE = r'data/raw/airports.dat' # Or ariports.dat
OUTPUT_FILE = 'aviation_network_raw.csv'
DEBUG_MAPPING_FILE = 'airport_country_mapping_debug.csv'
SNAPSHOT_YEAR = 2014

# --- COUNTRY MAPPING DICTIONARY (Reused & Expanded) ---
COUNTRY_MAPPING = {
    "United States": "USA", "United Kingdom": "GBR", "Russia": "RUS", "South Korea": "KOR",
    "North Korea": "PRK", "Vietnam": "VNM", "Laos": "LAO", "Syria": "SYR", "Iran": "IRN",
    "Tanzania": "TZA", "Bolivia": "BOL", "Venezuela": "VEN", "Moldova": "MDA",
    "Congo, Dem. Rep.": "COD", "Congo, Rep.": "COG", "Gambia, The": "GMB", "Bahamas, The": "BHS",
    "USSR": "RUS", "Soviet Union": "RUS", "Yugoslavia": "SRB", "Czechoslovakia": "CZE",
    "East Germany": "DEU", "West Germany": "DEU", "Burma": "MMR", "Ivory Coast": "CIV",
    "Cote d'Ivoire": "CIV", "Cape Verde": "CPV", "Swaziland": "SWZ", "Macedonia": "MKD",
    "Holy See": "VAT", "Macau": "MAC", "Hong Kong": "HKG", "Taiwan": "TWN"
}

# Try to use pycountry for missing entries
try:
    import pycountry
    HAS_PYCOUNTRY = True
    print("Pycountry library detected.")
except ImportError:
    HAS_PYCOUNTRY = False
    print("Pycountry not found. Using internal dictionary only.")

def get_iso_code(country_name):
    if not isinstance(country_name, str): return None
    clean_name = country_name.strip()
    
    if clean_name in COUNTRY_MAPPING:
        return COUNTRY_MAPPING[clean_name]
        
    if HAS_PYCOUNTRY:
        try:
            country = pycountry.countries.get(name=clean_name)
            if country: return country.alpha_3
            matches = pycountry.countries.search_fuzzy(clean_name)
            if matches: return matches[0].alpha_3
        except (LookupError, AttributeError):
            pass
    return None

def find_file(filepath):
    """Smart file finder to handle common issues like typos or specific paths"""
    if os.path.exists(filepath):
        return filepath
    
    # Check current directory
    basename = os.path.basename(filepath)
    if os.path.exists(basename):
        return basename
        
    # Check data/raw/
    raw_path = os.path.join('data', 'raw', basename)
    if os.path.exists(raw_path):
        return raw_path
        
    # Common typos/alternatives
    if 'airports' in basename:
        alts = ['ariports.dat', 'airports.csv', 'openflights_airports.dat']
        for alt in alts:
             # Check raw
             p = os.path.join('data', 'raw', alt)
             if os.path.exists(p): return p
             # Check local
             if os.path.exists(alt): return alt
             
    if 'routes' in basename:
        alts = ['openflights_routes.csv', 'routes.csv']
        for alt in alts:
             p = os.path.join('data', 'raw', alt)
             if os.path.exists(p): return p
             if os.path.exists(alt): return alt
             
    return None

def load_airports(filepath):
    print(f"Loading airports from {filepath}...")
    # OpenFlights airports.dat layout (variable length, usually 14 columns)
    # ID, Name, City, Country, IATA, ICAO, Lat, Lon, Alt, Timezone, DST, Tz, Type, Source
    try:
        # Load with minimal columns to avoid parsing errors
        # We need: ID (col 0), Country (col 3), IATA (col 4), ICAO (col 5)
        # Warning: Using header=None means columns are ints 0..N
        df = pd.read_csv(filepath, header=None, encoding='utf-8', on_bad_lines='skip')
        
        # Verify valid columns
        if len(df.columns) < 6:
            print("Error: Airports file seems to have too few columns.")
            return None
            
        mapping = {} # Key: Code (IATA or ICAO), Value: Country
        
        count = 0
        for idx, row in df.iterrows():
            country = str(row[3]).strip()
            iata = str(row[4]).strip()
            icao = str(row[5]).strip()
            
            # Map IATA (3 chars)
            if len(iata) == 3 and iata != '\\N':
                mapping[iata] = country
                
            # Map ICAO (4 chars)
            if len(icao) == 4 and icao != '\\N':
                mapping[icao] = country
                
            count += 1
            
        print(f"Parsed {count} airports. Mapped {len(mapping)} codes.")
        return mapping
        
    except Exception as e:
        print(f"Error loading airports: {e}")
        # Try latin-1 if utf-8 fails
        try:
             print("Retrying with latin-1 encoding...")
             df = pd.read_csv(filepath, header=None, encoding='latin-1', on_bad_lines='skip')
             # ... reused logic ...
             # For brevity, let's just fail if this doesn't work or return empty
             return None 
        except:
            return None

def load_routes(filepath):
    print(f"Loading routes from {filepath}...")
    # OpenFlights routes.dat columns:
    # 0: Airline, 1: ID, 2: Src, 3: SrcID, 4: Dst, 5: DstID, 6: Codeshare, 7: Stops, 8: Equip
    try:
        df = pd.read_csv(filepath, header=None, encoding='utf-8', on_bad_lines='skip')
        print(f"Loaded {len(df)} raw route rows.")
        return df
    except Exception as e:
        print(f"Error loading routes: {e}")
        return None

def main():
    # 1. Locate files
    airports_path = find_file(AIRPORTS_FILE)
    routes_path = find_file(ROUTES_FILE)
    
    if not airports_path or not routes_path:
        print(f"ERROR: Could not find data files.")
        print(f"Airports search: {AIRPORTS_FILE} -> Found: {airports_path}")
        print(f"Routes search: {ROUTES_FILE} -> Found: {routes_path}")
        sys.exit(1)
        
    # 2. Load Airports & Build Map
    airport_country_map = load_airports(airports_path)
    if not airport_country_map:
        print("Failed to build airport mapping.")
        sys.exit(1)
        
    # 3. Load Routes
    routes_df = load_routes(routes_path)
    if routes_df is None:
        sys.exit(1)
        
    # 4. Map Routes to Countries
    print("\nMapping routes to countries...")
    
    # Columns of interest: 2 (Source), 4 (Dest)
    # Check if we have enough columns
    if len(routes_df.columns) < 5:
        print("Error: Routes file has invalid format.")
        sys.exit(1)
        
    VALID_COLS = [2, 4] # Source, Dest
    
    processed_routes = []
    unmapped_airports = {}
    
    mapped_count = 0
    total_count = 0
    
    # Using simple iteration for clarity and control, or apply for speed
    # Let's use apply/map for speed if possible
    
    # Vectorized approach
    # Rename for clarity
    routes_work = routes_df[[2, 4]].copy()
    routes_work.columns = ['src', 'dst']
    
    # Filter \N
    routes_work = routes_work[ (routes_work['src'] != '\\N') & (routes_work['dst'] != '\\N') ]
    
    # Map country names
    routes_work['src_country'] = routes_work['src'].map(airport_country_map)
    routes_work['dst_country'] = routes_work['dst'].map(airport_country_map)
    
    # Track unmapped
    unmapped_src = routes_work[routes_work['src_country'].isna()]['src'].value_counts()
    unmapped_dst = routes_work[routes_work['dst_country'].isna()]['dst'].value_counts()
    
    # Filter success
    routes_mapped = routes_work.dropna(subset=['src_country', 'dst_country'])
    
    print(f"Routes mapped successfully: {len(routes_mapped)} / {len(routes_work)}")
    print(f"Success rate: {len(routes_mapped)/len(routes_df)*100:.2f}%")
    
    if len(unmapped_src) > 0:
        print(f"\nTop unmapped source airports:\n{unmapped_src.head(10)}")
        
    # 5. Convert Country Names to ISO
    print("\nConverting countries to ISO codes...")
    
    # Optimization: Get unique countries first
    unique_countries = pd.unique(routes_mapped[['src_country', 'dst_country']].values.ravel('K'))
    iso_map = {c: get_iso_code(c) for c in unique_countries}
    
    routes_mapped['origin'] = routes_mapped['src_country'].map(iso_map)
    routes_mapped['destination'] = routes_mapped['dst_country'].map(iso_map)
    
    # Filter invalid ISO
    final_df = routes_mapped.dropna(subset=['origin', 'destination']).copy()
    
    # 6. Cleaning
    print("\nCleaning data...")
    # Remove self-loops
    final_df = final_df[final_df['origin'] != final_df['destination']]
    print(f"Rows after removing self-loops: {len(final_df)}")
    
    # 7. Aggregation
    print("\nAggregating routes...")
    # Group by origin, destination
    agg_df = final_df.groupby(['origin', 'destination']).size().reset_index(name='route_count')
    
    # Add Year
    agg_df['year'] = SNAPSHOT_YEAR
    
    # Reorder columns
    agg_df = agg_df[['year', 'origin', 'destination', 'route_count']]
    
    # 8. Statistics & Output
    print("\n" + "="*30)
    print("AVIATION NETWORK STATISTICS")
    print("="*30)
    print(f"Total Unique Routes (Country Pairs): {len(agg_df)}")
    print(f"Total Flight Routes: {agg_df['route_count'].sum()}")
    
    print("\n--- Route Count Distribution ---")
    print(agg_df['route_count'].describe())
    
    print("\n--- Top 10 Country Pairs ---")
    print(agg_df.sort_values('route_count', ascending=False).head(10))
    
    print("\n--- Top 10 Origins ---")
    print(agg_df.groupby('origin')['route_count'].sum().sort_values(ascending=False).head(10))
    
    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    agg_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    
    # Save debug map
    with open(DEBUG_MAPPING_FILE, 'w', encoding='utf-8') as f:
        f.write("code,country\n")
        for k, v in list(airport_country_map.items())[:100]: # First 100
             f.write(f"{k},{v}\n")
             
    print("Done.")

if __name__ == "__main__":
    main()
