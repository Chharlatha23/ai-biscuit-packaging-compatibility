import os
import pandas as pd

def test_phase2_reports_exist():
    assert os.path.exists('reports/phase2_data_profile.md'), "Missing data profile"
    assert os.path.exists('reports/phase2_data_quality_report.md'), "Missing data quality report"
    assert os.path.exists('reports/phase2_eda_summary.csv'), "Missing summary"
    assert os.path.exists('reports/phase2_relationships.csv'), "Missing relationships"
    assert os.path.exists('reports/phase2_variable_dictionary.csv'), "Missing dictionary"

def test_phase2_figures_exist():
    assert os.path.exists('reports/phase2_figures/initial_moisture.png'), "Missing initial moisture figure"
    assert os.path.exists('reports/phase2_figures/storage_duration.png'), "Missing storage duration figure"
    assert os.path.exists('reports/phase2_figures/water_adsorption.png'), "Missing water adsorption figure"

def test_no_fabricated_variables():
    df = pd.read_csv('reports/phase2_variable_dictionary.csv')
    invalid_vars = ['fat', 'protein', 'ash', 'fiber']
    for v in invalid_vars:
        assert not df['variable_name'].str.lower().str.contains(v).any(), f"Fabricated variable {v} found!"

def test_metadata_exclusion():
    assert os.path.exists('reports/phase2_relationships.csv'), "Phase 2 relationships CSV is missing"
    df = pd.read_csv('reports/phase2_relationships.csv')
    assert not df['variable_x'].str.contains('source_', case=False, na=False).any(), "Metadata found in correlations"
    assert not df['variable_y'].str.contains('source_', case=False, na=False).any(), "Metadata found in correlations"

def test_outlier_wording():
    with open('reports/phase2_data_quality_report.md', 'r') as f:
        content = f.read()
    assert "Statistical outlier candidate requiring domain validation" in content, "Missing or incorrect outlier wording"
