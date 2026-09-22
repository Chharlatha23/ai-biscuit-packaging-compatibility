import os
import glob
import hashlib
import zipfile
import pandas as pd
import numpy as np

def make_id(*args):
    s = "|".join([str(a) for a in args])
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def extract_archive(zip_path, dataset_name):
    archive_name = os.path.basename(zip_path)
    records_init = []
    records_water = []
    records_storage = []
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file in z.namelist():
            if not file.endswith('.xlsx'): continue
            if 'Infrared spectra' in file: continue
            
            fname = os.path.basename(file)
            
            with z.open(file) as f:
                df = pd.read_excel(f, sheet_name=0, header=None)
                
            if 'Initial characterization' in fname:
                # Header: ['Sample', 'Replicates', 'Moisture content (% wet basis)', 'Moisture content (% dry basis)']
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
                            # A new sample block begins
                            current_sample = r['Sample']
                        elif is_completely_empty or is_aggregate:
                            # An empty row or aggregate row breaks the block (separator)
                            current_sample = None
                            continue
                            
                        if is_completely_empty or pd.isna(current_sample) or is_aggregate:
                            continue
                            
                        rec = {
                            'source_archive': archive_name,
                            'source_archive_path': zip_path.replace('\\\\', '/'),
                            'source_dataset': dataset_name,
                            'source_file': fname,
                            'source_sheet': 'Sheet1',
                            'source_row': i + 1,
                            'record_id': make_id(archive_name, fname, i + 1, current_sample),
                            'dataset_type': 'Initial Characterization',
                            'original_sample': current_sample,
                            'replicate_id': r['Replicates'],
                            'initial_moisture_content': r['Moisture_wet']
                        }
                        records_init.append(rec)
                        
            elif 'WaterAdsorptionIsotherms' in fname or 'Water Adsorption Isotherms' in fname:
                # Header: ['Replicates', 'Process', 'Storage temperature (°C)', 'Water activity', 'Moisture content (% wet basis)', 'Moisture content (% dry basis)']
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
                        rec = {
                            'source_archive': archive_name,
                            'source_archive_path': zip_path.replace('\\\\', '/'),
                            'source_dataset': dataset_name,
                            'source_file': fname,
                            'source_sheet': 'Sheet1',
                            'source_row': i + 1,
                            'record_id': make_id(archive_name, fname, i + 1, r[0], r[3]),
                            'dataset_type': 'Water Adsorption Isotherms',
                            'replicate_id': r[0],
                            'storage_temperature': r[2],
                            'adsorption_water_activity': r[3],
                            'adsorption_moisture_content': r[4]
                        }
                        records_water.append(rec)
                        
            elif 'StorageAssesmentAchiraBiscuits' in fname:
                # Header: ['Sample', 'Time (day)', 'L*', 'a*', 'b*', 'Hardness (N)', 'Moisture content (% wet basis)', 'Water activity', 'Sensory quality', 'Moisture content (% dry basis)']
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
                            # A new time block begins
                            current_time = pd.to_numeric(r['Time'], errors='coerce')
                        elif is_completely_empty:
                            # A fully empty row breaks the block
                            current_time = None
                            continue
                            
                        # If row lacks a valid sample or we have no time, we skip it
                        if pd.isna(r['Sample']) or pd.isna(current_time) or is_completely_empty:
                            continue
                            
                        rec = {
                            'source_archive': archive_name,
                            'source_archive_path': zip_path.replace('\\\\', '/'),
                            'source_dataset': dataset_name,
                            'source_file': fname,
                            'source_sheet': 'Sheet1',
                            'source_row': i + 1,
                            'record_id': make_id(archive_name, fname, i + 1, r['Sample']),
                            'dataset_type': 'Storage Assessment',
                            'original_sample': r['Sample'],
                            'storage_duration': current_time,
                            'L_star': r['L'],
                            'a_star': r['a'],
                            'b_star': r['b'],
                            'storage_hardness': r['Hardness'],
                            'storage_moisture_content': r['Moisture'],
                            'storage_water_activity': r['Aw'],
                            'sensory_quality': r['Sensory']
                        }
                        records_storage.append(rec)
                        
    return records_init, records_water, records_storage

def run_pipeline():
    print("Starting extraction pipeline...")
    zips = glob.glob('raw_data/**/*.zip', recursive=True)
    
    all_init, all_water, all_storage = [], [], []
    
    for zpath in zips:
        if 'achira_storage' in zpath:
            ds_name = 'achira_storage'
        else:
            ds_name = 'achira_water_adsorption'
            
        i, w, s = extract_archive(zpath, ds_name)
        all_init.extend(i)
        all_water.extend(w)
        all_storage.extend(s)
        
    df_init = pd.DataFrame(all_init)
    df_water = pd.DataFrame(all_water)
    df_storage = pd.DataFrame(all_storage)
    
    os.makedirs('processed_data', exist_ok=True)
    
    if not df_init.empty: df_init.to_csv('processed_data/phase1_initial_characterization.csv', index=False)
    if not df_water.empty: df_water.to_csv('processed_data/phase1_water_adsorption.csv', index=False)
    if not df_storage.empty: df_storage.to_csv('processed_data/phase1_storage_assessment.csv', index=False)
    
    df_combined = pd.concat([df_init, df_water, df_storage], ignore_index=True)
    if df_combined.empty:
        return
        
    # Deduplication
    provenance_cols = ['source_archive', 'source_dataset', 'source_file', 'source_sheet', 'source_row', 'record_id', 'dataset_type']
    meas_cols = [c for c in df_combined.columns if c not in provenance_cols]
    
    df_combined['duplicate_status'] = 'unique'
    df_combined['duplicate_group_id'] = np.nan
    df_combined['duplicate_reason'] = ''
    
    # 1. Exact duplicates (same exact measurements)
    df_meas = df_combined[meas_cols].fillna('MISSING')
    df_combined['exact_group_id'] = df_meas.groupby(meas_cols).ngroup().astype(str)
    exact_counts = df_combined.groupby('exact_group_id').size()
    exact_groups = exact_counts[exact_counts > 1].index
    
    for g in exact_groups:
        idx = df_combined[df_combined['exact_group_id'] == g].index
        # Keep first, others are confirmed_duplicate
        df_combined.loc[idx[1:], 'duplicate_status'] = 'confirmed_duplicate'
        df_combined.loc[idx[1:], 'duplicate_reason'] = 'Exact duplicate measurements'
        df_combined.loc[idx, 'duplicate_group_id'] = 'exact_' + str(g)
        
    # 2. Repeated scientific observations across datasets (not exact, but same core context)
    # 3. Possible duplicates
    
    # Save merged RAW dataset
    df_combined.to_csv('processed_data/merged_raw_dataset.csv', index=False)
    
    df_dedup = df_combined[df_combined['duplicate_status'] == 'unique'].copy()
    df_dedup.to_csv('processed_data/deduplicated_dataset.csv', index=False)
    
    # Audit registry
    dup_records = []
    for g in df_combined['duplicate_group_id'].dropna().unique():
        group = df_combined[df_combined['duplicate_group_id'] == g]
        kept = group[group['duplicate_status'] == 'unique']
        removed = group[group['duplicate_status'] != 'unique']
        
        if kept.empty:
            kept_id = ''
        else:
            kept_id = kept.iloc[0]['record_id']
            
        for _, row in removed.iterrows():
            dup_records.append({
                'duplicate_group_id': g,
                'retained_record_id': kept_id,
                'removed_record_id': row['record_id'],
                'source_archive': row['source_archive'],
                'source_dataset': row['source_dataset'],
                'source_file': row['source_file'],
                'source_sheet': row['source_sheet'],
                'source_row': row['source_row'],
                'duplicate_reason': row['duplicate_reason']
            })
            
    df_audit = pd.DataFrame(dup_records)
    if not df_audit.empty:
        df_audit.to_csv('processed_data/duplicate_audit_registry.csv', index=False)
    else:
        pd.DataFrame(columns=['duplicate_group_id', 'retained_record_id', 'removed_record_id', 'source_archive', 'source_dataset', 'source_file', 'source_sheet', 'source_row', 'duplicate_reason']).to_csv('processed_data/duplicate_audit_registry.csv', index=False)
        
if __name__ == '__main__':
    run_pipeline()
