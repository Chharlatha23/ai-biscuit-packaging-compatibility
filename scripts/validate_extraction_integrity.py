import pandas as pd
import zipfile
import glob
import sys
import os

def parse_archive(zip_path, dataset_name):
    archive_name = os.path.basename(zip_path)
    expected = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file in z.namelist():
            if not file.endswith('.xlsx') or 'Infrared spectra' in file:
                continue
            
            fname = os.path.basename(file)
            with z.open(file) as f:
                df = pd.read_excel(f, sheet_name=0, header=None)
                
            if 'Initial characterization' in fname:
                header_idx = -1
                for i, r in df.iterrows():
                    if str(r[0]).strip() == 'Sample':
                        header_idx = i
                        break
                if header_idx != -1:
                    df_sub = df.iloc[header_idx + 1: , :4].copy()
                    df_sub.columns = ['Sample', 'Replicates', 'Moisture_wet', 'Moisture_dry']
                    
                    current_sample = None
                    for i, r in df_sub.iterrows():
                        is_completely_empty = r.isna().all()
                        rep_str = str(r['Replicates']).strip().lower()
                        is_aggregate = rep_str in ['mean', 'standard deviation', 'nan']
                        
                        if pd.notna(r['Sample']):
                            current_sample = r['Sample']
                        elif is_completely_empty or is_aggregate:
                            current_sample = None
                            continue
                            
                        if is_completely_empty or pd.isna(current_sample) or is_aggregate:
                            continue
                            
                        expected.append({
                            'source_archive': archive_name,
                            'source_file': fname,
                            'source_sheet': 'Sheet1',
                            'source_row': i + 1,
                            'dataset_type': 'Initial Characterization',
                            'expected_sample': current_sample,
                            'expected_replicate': r['Replicates']
                        })
                        
            elif 'WaterAdsorptionIsotherms' in fname or 'Water Adsorption Isotherms' in fname:
                header_idx = -1
                for i, r in df.iterrows():
                    if str(r[0]).strip() == 'Replicates' and str(r[1]).strip() == 'Process':
                        header_idx = i
                        break
                if header_idx != -1:
                    for i in range(header_idx + 1, len(df)):
                        r = df.iloc[i]
                        valid_replicate = pd.notna(r[0])
                        valid_water_activity = pd.notna(r[3])
                        is_valid_data = valid_replicate and valid_water_activity
                        if not is_valid_data: continue
                        
                        expected.append({
                            'source_archive': archive_name,
                            'source_file': fname,
                            'source_sheet': 'Sheet1',
                            'source_row': i + 1,
                            'dataset_type': 'Water Adsorption Isotherms',
                            'expected_sample': None,
                            'expected_replicate': r[0]
                        })

            elif 'StorageAssesmentAchiraBiscuits' in fname:
                header_idx = -1
                for i, r in df.iterrows():
                    if str(r[0]).strip() == 'Sample' and str(r[1]).strip() == 'Time (day)':
                        header_idx = i
                        break
                if header_idx != -1:
                    df_sub = df.iloc[header_idx + 1: , :9].copy()
                    df_sub.columns = ['Sample', 'Time', 'L', 'a', 'b', 'Hardness', 'Moisture', 'Aw', 'Sensory']
                    
                    current_time = None
                    for i, r in df_sub.iterrows():
                        is_completely_empty = r.isna().all()
                        
                        if pd.notna(r['Time']):
                            current_time = pd.to_numeric(r['Time'], errors='coerce')
                        elif is_completely_empty:
                            current_time = None
                            continue
                            
                        if pd.isna(r['Sample']) or pd.isna(current_time) or is_completely_empty:
                            continue
                            
                        expected.append({
                            'source_archive': archive_name,
                            'source_file': fname,
                            'source_sheet': 'Sheet1',
                            'source_row': i + 1,
                            'dataset_type': 'Storage Assessment',
                            'expected_sample': r['Sample'],
                            'expected_replicate': None
                        })
    return expected

def build_expected_registry():
    zips = glob.glob('raw_data/**/*.zip', recursive=True)
    all_expected = []
    for zpath in zips:
        ds_name = 'achira_storage' if 'achira_storage' in zpath else 'achira_water_adsorption'
        all_expected.extend(parse_archive(zpath, ds_name))
    return pd.DataFrame(all_expected)

def validate():
    print("Validating source-to-output reconciliation...")
    try:
        df_exp = build_expected_registry()
        
        df_init = pd.read_csv('processed_data/phase1_initial_characterization.csv')
        df_store = pd.read_csv('processed_data/phase1_storage_assessment.csv')
        df_water = pd.read_csv('processed_data/phase1_water_adsorption.csv')
        
        df_raw = pd.read_csv('processed_data/merged_raw_dataset.csv')
        df_dedup = pd.read_csv('processed_data/deduplicated_dataset.csv')
        df_audit = pd.read_csv('processed_data/duplicate_audit_registry.csv')
        
        # 1. Dataset specific mappings
        # For each expected record, verify exactly one corresponding row exists in the merged_raw_dataset
        df_exp['prov_key'] = df_exp['source_archive'] + df_exp['source_file'] + df_exp['source_sheet'] + df_exp['source_row'].astype(str)
        df_raw['prov_key'] = df_raw['source_archive'] + df_raw['source_file'] + df_raw['source_sheet'] + df_raw['source_row'].astype(str)
        
        exp_keys = set(df_exp['prov_key'])
        raw_keys = set(df_raw['prov_key'])
        
        if not exp_keys.issubset(raw_keys):
            missing = exp_keys - raw_keys
            print(f"ERROR: Missing expected valid data rows in raw output! Example: {list(missing)[0]}")
            sys.exit(1)
            
        if not raw_keys.issubset(exp_keys):
            extra = raw_keys - exp_keys
            print(f"ERROR: Extracted row exists that is NOT a valid source row! Example: {list(extra)[0]}")
            sys.exit(1)
            
        # 2. Check assignment accuracy
        for _, row in df_exp.iterrows():
            matches = df_raw[df_raw['prov_key'] == row['prov_key']]
            if len(matches) != 1:
                print(f"ERROR: Found {len(matches)} rows for unique source key {row['prov_key']}")
                sys.exit(1)
            
            m = matches.iloc[0]
            if pd.notna(row['expected_sample']):
                if m['original_sample'] != row['expected_sample']:
                    print(f"ERROR: Sample assignment mismatch for {row['prov_key']}")
                    sys.exit(1)
                    
        # 3. Duplicate accounting
        if len(df_raw) != len(df_dedup) + len(df_audit):
            print("ERROR: Duplicate accounting mismatch")
            sys.exit(1)
            
        # 4. Numeric assertions and valid constraints
        if df_store['storage_duration'].isna().any() or not pd.api.types.is_numeric_dtype(df_store['storage_duration']):
            print("ERROR: storage_duration invalid")
            sys.exit(1)
            
        if df_water['replicate_id'].isna().any():
            print("ERROR: Missing replicate_id in water adsorption")
            sys.exit(1)
            
        if not pd.api.types.is_numeric_dtype(df_water['adsorption_water_activity']):
            print("ERROR: adsorption_water_activity not numeric")
            sys.exit(1)
            
        print("Genuine source-to-output reconciliation PASSED.")
        sys.exit(0)
    except Exception as e:
        print(f"Validation failed with exception: {e}")
        sys.exit(1)

if __name__ == '__main__':
    validate()
