import os
import pandas as pd
import numpy as np

def test_extraction_files_generated():
    assert os.path.exists('processed_data/phase1_initial_characterization.csv')
    assert os.path.exists('processed_data/phase1_storage_assessment.csv')
    assert os.path.exists('processed_data/deduplicated_dataset.csv')

def test_no_missing_sample_ids():
    df1 = pd.read_csv('processed_data/phase1_initial_characterization.csv')
    assert not df1['original_sample'].isna().any(), "Found missing Sample IDs in Initial Char"
    
    df2 = pd.read_csv('processed_data/phase1_storage_assessment.csv')
    assert not df2['original_sample'].isna().any(), "Found missing Sample IDs in Storage Assess"

def test_time_leakage():
    df = pd.read_csv('processed_data/phase1_storage_assessment.csv')
    assert not df['storage_duration'].isna().any(), "Found missing Time values"
    assert df['storage_duration'].dtype == 'float64' or df['storage_duration'].dtype == 'int64'

def test_valid_continuation_rows_preserved():
    df = pd.read_csv('processed_data/phase1_initial_characterization.csv')
    # If the logic incorrectly dropped continuation rows, we would only have 1 row per sample.
    counts = df.groupby('original_sample').size()
    assert (counts > 1).any(), "Continuation rows were incorrectly discarded, each sample only has 1 row"

def test_numeric_validity():
    # Initial characterization
    df_init = pd.read_csv('processed_data/phase1_initial_characterization.csv')
    num_cols_init = ['initial_moisture_content']
    for col in num_cols_init:
        assert pd.api.types.is_numeric_dtype(df_init[col]), f"{col} must be numeric"
        assert not df_init[col].isna().any(), f"{col} has unexpected NaNs"
        assert not np.isinf(df_init[col]).any(), f"{col} has Inf"
        
    # Storage assessment
    df_store = pd.read_csv('processed_data/phase1_storage_assessment.csv')
    num_cols_store = ['storage_duration', 'L_star', 'a_star', 'b_star', 'storage_hardness', 'storage_moisture_content', 'storage_water_activity']
    for col in num_cols_store:
        assert pd.api.types.is_numeric_dtype(df_store[col]), f"{col} must be numeric"
        assert not df_store[col].isna().any(), f"{col} has unexpected NaNs"
        assert not np.isinf(df_store[col]).any(), f"{col} has Inf"
        
    # Water adsorption
    df_water = pd.read_csv('processed_data/phase1_water_adsorption.csv')
    num_cols_water = ['storage_temperature', 'adsorption_water_activity', 'adsorption_moisture_content']
    for col in num_cols_water:
        assert pd.api.types.is_numeric_dtype(df_water[col]), f"{col} must be numeric"
        assert not df_water[col].isna().any(), f"{col} has unexpected NaNs"
        assert not np.isinf(df_water[col]).any(), f"{col} has Inf"

def test_provenance_columns():
    df = pd.read_csv('processed_data/phase1_storage_assessment.csv')
    required_cols = ['source_archive', 'source_file', 'source_sheet', 'source_row', 'record_id']
    for col in required_cols:
        assert col in df.columns, f"Missing provenance column {col}"
    assert not df['source_row'].isna().any(), "Missing source row numbers"

def test_duplicate_accounting():
    df_raw = pd.read_csv('processed_data/merged_raw_dataset.csv')
    df_dedup = pd.read_csv('processed_data/deduplicated_dataset.csv')
    df_audit = pd.read_csv('processed_data/duplicate_audit_registry.csv')
    assert len(df_raw) == len(df_dedup) + len(df_audit), "Duplicate accounting mismatch"
