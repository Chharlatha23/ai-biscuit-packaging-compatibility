import os
import pytest
from ml.predict import predict_packaging_compatibility
import subprocess
import pandas as pd
import hashlib

def test_missing_input():
    test_input = {
        'initial_moisture': 5.0,
        'final_moisture': 6.5,
        'thickness_micron': 30,
        'wvtr': 12.0,
        'otr': 500,
        'evidence_quality': 1.0
        # Missing packaging_material
    }
    with pytest.raises(ValueError):
        predict_packaging_compatibility(test_input)

def test_negative_thickness():
    test_input = {
        'initial_moisture': 5.0,
        'final_moisture': 6.5,
        'thickness_micron': -10,
        'wvtr': 12.0,
        'otr': 500,
        'packaging_material': 'LDPE',
        'evidence_quality': 1.0
    }
    with pytest.raises(ValueError):
        predict_packaging_compatibility(test_input)

def test_negative_wvtr():
    test_input = {
        'initial_moisture': 5.0,
        'final_moisture': 6.5,
        'thickness_micron': 30,
        'wvtr': -5.0,
        'otr': 500,
        'packaging_material': 'LDPE',
        'evidence_quality': 1.0
    }
    with pytest.raises(ValueError):
        predict_packaging_compatibility(test_input)

def test_negative_otr():
    test_input = {
        'initial_moisture': 5.0,
        'final_moisture': 6.5,
        'thickness_micron': 30,
        'wvtr': 12.0,
        'otr': -500,
        'packaging_material': 'LDPE',
        'evidence_quality': 1.0
    }
    with pytest.raises(ValueError):
        predict_packaging_compatibility(test_input)

def test_invalid_moisture():
    test_input = {
        'initial_moisture': 150.0,
        'final_moisture': 6.5,
        'thickness_micron': 30,
        'wvtr': 12.0,
        'otr': 500,
        'packaging_material': 'LDPE',
        'evidence_quality': 1.0
    }
    with pytest.raises(ValueError):
        predict_packaging_compatibility(test_input)

def test_final_less_than_initial_moisture():
    test_input = {
        'initial_moisture': 6.5,
        'final_moisture': 5.0,
        'thickness_micron': 30,
        'wvtr': 12.0,
        'otr': 500,
        'packaging_material': 'LDPE',
        'evidence_quality': 1.0
    }
    result = predict_packaging_compatibility(test_input)
    assert result['recommendation'] in ['Recommended', 'Conditionally Recommended', 'Not Recommended']
    assert 'class_probabilities' in result

def test_final_equal_initial_moisture():
    test_input = {
        'initial_moisture': 5.0,
        'final_moisture': 5.0,
        'thickness_micron': 30,
        'wvtr': 12.0,
        'otr': 500,
        'packaging_material': 'LDPE',
        'evidence_quality': 1.0
    }
    result = predict_packaging_compatibility(test_input)
    assert result['recommendation'] in ['Recommended', 'Conditionally Recommended', 'Not Recommended']
    assert 'class_probabilities' in result

def test_unknown_categorical_handling():
    test_input = {
        'initial_moisture': 5.0,
        'final_moisture': 6.5,
        'thickness_micron': 30,
        'wvtr': 12.0,
        'otr': 500,
        'packaging_material': 'UNKNOWN_MATERIAL_123',
        'evidence_quality': 1.0
    }
    # Should raise a ValueError now because handle_unknown='error'
    with pytest.raises(ValueError):
        predict_packaging_compatibility(test_input)

def test_probability_bounds():
    test_input = {
        'initial_moisture': 5.0,
        'final_moisture': 6.5,
        'thickness_micron': 30,
        'wvtr': 12.0,
        'otr': 500,
        'packaging_material': 'LDPE',
        'evidence_quality': 1.0
    }
    result = predict_packaging_compatibility(test_input)
    assert 0.0 <= result['probability'] <= 1.0

def test_reproducible_output():
    test_input = {
        'initial_moisture': 4.0,
        'final_moisture': 6.0,
        'thickness_micron': 25,
        'wvtr': 5.0,
        'otr': 100,
        'packaging_material': 'PET',
        'evidence_quality': 1.0
    }
    result1 = predict_packaging_compatibility(test_input)
    result2 = predict_packaging_compatibility(test_input)
    assert result1['recommendation'] == result2['recommendation']
    assert result1['probability'] == result2['probability']

def test_app_starts():
    # Simple check if app.py exists
    assert os.path.exists('app.py')
