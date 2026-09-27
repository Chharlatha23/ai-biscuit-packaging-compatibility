import os
import pandas as pd
import pytest
from src.packaging_validator import validate_packaging_data
from src.packaging_integration import extract_biscuit_features, classify_moisture_sensitivity, score_material

def test_packaging_validator(tmp_path):
    df = pd.DataFrame({'material_id': ['M1']})
    dummy_file = tmp_path / 'dummy_pack.csv'
    df.to_csv(dummy_file, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_packaging_data(dummy_file)

def test_packaging_validator_negative(tmp_path):
    df = pd.DataFrame([{
        'material_id': 'M1', 'material_name': 'Test', 'material_category': 'PE', 'thickness_micron': 10,
        'wvtr_value': -1.0, 'wvtr_unit': 'g/m2/day', 'otr_value': 10, 'otr_unit': 'cm3/m2/day',
        'test_temperature_c': 25, 'relative_humidity_percent': 50,
        'source_reference': 'Src', 'source_url_or_doi': 'URL', 'data_quality_note': 'Note'
    }])
    dummy_file = tmp_path / 'dummy_pack.csv'
    df.to_csv(dummy_file, index=False)
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_packaging_data(dummy_file)

def test_packaging_validator_units(tmp_path):
    df = pd.DataFrame([{
        'material_id': 'M1', 'material_name': 'Test', 'material_category': 'PE', 'thickness_micron': 10,
        'wvtr_value': 1.0, 'wvtr_unit': 'invalid', 'otr_value': 10, 'otr_unit': 'cm3/m2/day',
        'test_temperature_c': 25, 'relative_humidity_percent': 50,
        'source_reference': 'Src', 'source_url_or_doi': 'URL', 'data_quality_note': 'Note'
    }])
    dummy_file = tmp_path / 'dummy_pack.csv'
    df.to_csv(dummy_file, index=False)
    with pytest.raises(ValueError, match="Unrecognized WVTR units"):
        validate_packaging_data(dummy_file)

def test_packaging_validator_duplicate(tmp_path):
    df = pd.DataFrame([
        {'material_id': 'M1', 'material_name': 'Test', 'material_category': 'PE', 'thickness_micron': 10, 'wvtr_value': 1.0, 'wvtr_unit': 'g/m2/day', 'otr_value': 10, 'otr_unit': 'cm3/m2/day', 'test_temperature_c': 25, 'relative_humidity_percent': 50, 'source_reference': 'Src', 'source_url_or_doi': 'URL', 'data_quality_note': 'Note'},
        {'material_id': 'M1', 'material_name': 'Test', 'material_category': 'PE', 'thickness_micron': 10, 'wvtr_value': 1.0, 'wvtr_unit': 'g/m2/day', 'otr_value': 10, 'otr_unit': 'cm3/m2/day', 'test_temperature_c': 25, 'relative_humidity_percent': 50, 'source_reference': 'Src', 'source_url_or_doi': 'URL', 'data_quality_note': 'Note'}
    ])
    dummy_file = tmp_path / 'dummy_pack.csv'
    df.to_csv(dummy_file, index=False)
    with pytest.raises(ValueError, match="Duplicate material_ids"):
        validate_packaging_data(dummy_file)

def test_moisture_gain():
    biscuit_df = pd.DataFrame({
        'source_dataset': ['A', 'A'],
        'original_sample': ['S1', 'S1'],
        'dataset_type': ['Initial Characterization', 'Storage Assessment'],
        'storage_duration': [0, 10],
        'initial_moisture_content': [5.0, 5.0],
        'storage_moisture_content': [5.0, 6.5],
        'storage_temperature': [25.0, 25.0]
    })
    features = extract_biscuit_features(biscuit_df)
    assert features['moisture_gain'].iloc[0] == 1.5

def test_moisture_sensitivity():
    sens, wvtr = classify_moisture_sensitivity(1.5)
    assert sens == 'High'
    assert wvtr == 5.0

    sens, wvtr = classify_moisture_sensitivity(0.8)
    assert sens == 'Medium'
    assert wvtr == 15.0

    sens, wvtr = classify_moisture_sensitivity(0.2)
    assert sens == 'Low'
    assert wvtr == 30.0

def test_score_material():
    material = {
        'wvtr_value': 4.0,
        'source_url_or_doi': '10.123/123',
        'test_temperature_c': 25.0
    }
    # Required WVTR = 5.0, Biscuit temp = 25.0
    # moisture score = 1.0
    # evidence score = 1.0 (DOI present)
    # condition score = 1.0 (temp diff = 0)
    # final = 0.7*1 + 0.2*1 + 0.1*1 = 1.0
    score, status, reason = score_material(5.0, material, 25.0)
    assert abs(score - 1.0) < 1e-9
    assert status == 'Recommended'

    # Material fails requirement
    material2 = material.copy()
    material2['wvtr_value'] = 20.0
    score, status, reason = score_material(5.0, material2, 25.0)
    assert status == 'Not Recommended'

    # Material conditional
    material3 = material.copy()
    material3['wvtr_value'] = 7.0
    score, status, reason = score_material(5.0, material3, 25.0)
    assert status == 'Conditionally Recommended'

def test_extract_biscuit_features_source_scoping():
    biscuit_df = pd.DataFrame({
        'source_dataset': ['A', 'A', 'B', 'B'],
        'original_sample': ['S1', 'S1', 'S1', 'S1'],
        'dataset_type': ['Initial Characterization', 'Storage Assessment', 'Initial Characterization', 'Storage Assessment'],
        'storage_duration': [0, 10, 0, 10],
        'initial_moisture_content': [5.0, 5.0, 99.0, 99.0],
        'storage_moisture_content': [5.0, 6.5, 99.0, 99.5],
        'storage_temperature': [25.0, 25.0, 25.0, 25.0]
    })

    features_AB = extract_biscuit_features(biscuit_df)
    assert len(features_AB) == 2

    biscuit_df_BA = pd.DataFrame({
        'source_dataset': ['B', 'B', 'A', 'A'],
        'original_sample': ['S1', 'S1', 'S1', 'S1'],
        'dataset_type': ['Initial Characterization', 'Storage Assessment', 'Initial Characterization', 'Storage Assessment'],
        'storage_duration': [0, 10, 0, 10],
        'initial_moisture_content': [99.0, 99.0, 5.0, 5.0],
        'storage_moisture_content': [99.0, 99.5, 5.0, 6.5],
        'storage_temperature': [25.0, 25.0, 25.0, 25.0]
    })
    features_BA = extract_biscuit_features(biscuit_df_BA)

    features_AB = features_AB.sort_values('initial_moisture').reset_index(drop=True)
    features_BA = features_BA.sort_values('initial_moisture').reset_index(drop=True)

    pd.testing.assert_frame_equal(features_AB, features_BA)

    assert features_AB['initial_moisture'].iloc[0] == 5.0
    assert features_AB['moisture_gain'].iloc[0] == 1.5

    assert features_AB['initial_moisture'].iloc[1] == 99.0
    assert features_AB['moisture_gain'].iloc[1] == 0.5
