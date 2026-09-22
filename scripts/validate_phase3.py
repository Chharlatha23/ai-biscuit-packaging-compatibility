import os
import pandas as pd
import sys

def validate():
    print("Starting Phase 3 Independent Validation...")
    
    # Check required files exist
    assert os.path.exists('processed_data/deduplicated_dataset.csv'), "deduplicated_dataset.csv missing"
    assert os.path.exists('processed_data/final_biscuit_analysis_dataset.csv'), "final_biscuit_analysis_dataset.csv missing"
    assert os.path.exists('source_registry/phase3_pairing_registry.csv'), "phase3_pairing_registry.csv missing"
    
    # Ensure invalid files DO NOT exist
    assert not os.path.exists('processed_data/paired_biscuit_packaging.csv'), "paired_biscuit_packaging.csv illegally created"
    assert not os.path.exists('processed_data/final_packaging_analysis_dataset.csv'), "final_packaging_analysis_dataset.csv illegally created"
    
    df_dedup = pd.read_csv('processed_data/deduplicated_dataset.csv')
    df_final = pd.read_csv('processed_data/final_biscuit_analysis_dataset.csv')
    df_registry = pd.read_csv('source_registry/phase3_pairing_registry.csv')
    
    # 1. identical schemas
    assert list(df_dedup.columns) == list(df_final.columns), "Schemas mismatched"
    
    # 2. identical row counts
    assert len(df_dedup) == len(df_final), "Row counts mismatched"
    
    # 3. identical values
    pd.testing.assert_frame_equal(df_dedup, df_final, check_dtype=False)
    
    # 4. registry must be completely empty
    assert len(df_registry) == 0, "Pairing registry is not empty! Fabricated pairings detected!"
    
    print("Phase 3 Independent Validation PASSED: No fabricated data. Datasets identically preserved.")
    sys.exit(0)

if __name__ == '__main__':
    validate()
