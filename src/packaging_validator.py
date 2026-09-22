import pandas as pd

def validate_packaging_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Packaging dataset not found at {filepath}")

    required_columns = [
        'material_id', 'material_name', 'material_category', 'thickness_micron',
        'wvtr_value', 'wvtr_unit', 'otr_value', 'otr_unit',
        'test_temperature_c', 'relative_humidity_percent',
        'source_reference', 'source_url_or_doi', 'data_quality_note'
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in packaging dataset: {missing_cols}")

    if df['material_id'].duplicated().any():
        raise ValueError("Duplicate material_ids found.")

    if df['material_name'].isna().any() or (df['material_name'].str.strip() == '').any():
        raise ValueError("Empty material_names found.")

    if not (df['thickness_micron'] > 0).all():
        raise ValueError("All thickness_micron values must be positive.")

    if not (df['wvtr_value'] >= 0).all():
        raise ValueError("All wvtr_value values must be non-negative.")

    if not (df['otr_value'] >= 0).all():
        raise ValueError("All otr_value values must be non-negative.")

    if not df['test_temperature_c'].between(-50, 150).all():
        raise ValueError("test_temperature_c contains unrealistic values.")

    if not df['relative_humidity_percent'].between(0, 100).all():
        raise ValueError("relative_humidity_percent must be between 0 and 100.")

    valid_wvtr_units = ['g/m2/day']
    if not df['wvtr_unit'].isin(valid_wvtr_units).all():
        raise ValueError(f"Unrecognized WVTR units. Expected {valid_wvtr_units}")

    valid_otr_units = ['cm3/m2/day']
    if not df['otr_unit'].isin(valid_otr_units).all():
        raise ValueError(f"Unrecognized OTR units. Expected {valid_otr_units}")

    if df['source_reference'].isna().any() or (df['source_reference'].str.strip() == '').any():
        raise ValueError("source_reference cannot be empty. All values must have a verifiable source.")

    return True

if __name__ == "__main__":
    try:
        validate_packaging_data('raw_data/packaging/literature_packaging_matrix.csv')
        print("Packaging data validation passed.")
    except Exception as e:
        print(f"Validation Error: {e}")
        exit(1)
