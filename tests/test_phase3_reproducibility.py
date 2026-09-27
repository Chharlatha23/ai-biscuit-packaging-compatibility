import os
import sys
import pytest
import pandas as pd
import hashlib
import subprocess

def run_phase3_script():
    # Execute the actual packaging_integration script
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.abspath('.')
    result = subprocess.run([sys.executable, "src/packaging_integration.py"], capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print("Script failed:", result.stderr)
    return result.returncode

def get_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def test_phase3_reproducibility():
    # Make sure we're in the right directory
    assert os.path.exists('src/packaging_integration.py'), "Must run from project root"

    # Backup tracked files
    registry_path = 'processed_data/packaging_pairing_registry.csv'
    report_path = 'reports/phase3_validation_report.md'

    registry_backup = None
    report_backup = None

    if os.path.exists(registry_path):
        with open(registry_path, 'rb') as f:
            registry_backup = f.read()
    if os.path.exists(report_path):
        with open(report_path, 'rb') as f:
            report_backup = f.read()

    try:
        # Run 1
        code1 = run_phase3_script()
        assert code1 == 0, "Phase 3 script failed on first run"

        hash_biscuit1 = get_hash('processed_data/final_biscuit_analysis_dataset.csv')
        hash_registry1 = get_hash('processed_data/packaging_pairing_registry.csv')

        df1 = pd.read_csv('processed_data/packaging_pairing_registry.csv')

        # Run 2
        code2 = run_phase3_script()
        assert code2 == 0, "Phase 3 script failed on second run"

        hash_biscuit2 = get_hash('processed_data/final_biscuit_analysis_dataset.csv')
        hash_registry2 = get_hash('processed_data/packaging_pairing_registry.csv')

        df2 = pd.read_csv('processed_data/packaging_pairing_registry.csv')

        assert hash_biscuit1 == hash_biscuit2, "Biscuit dataset hash mismatch between runs"
        assert hash_registry1 == hash_registry2, "Registry dataset hash mismatch between runs"

        # Verify the structure matches expectations
        assert len(df1) > 0, "Registry must not be empty"
        expected_cols = ['product_id', 'initial_moisture', 'final_moisture', 'packaging_material',
                         'thickness_micron', 'wvtr', 'otr', 'evidence_quality', 'recommendation_status']
        for col in expected_cols:
            assert col in df1.columns, f"Missing column {col}"
    finally:
        # Restore tracked files to avoid side effects
        if registry_backup is not None:
            with open(registry_path, 'wb') as f:
                f.write(registry_backup)
        if report_backup is not None:
            with open(report_path, 'wb') as f:
                f.write(report_backup)
