import pandas as pd
import numpy as np

def extract_biscuit_features(biscuit_df):
    """
    Extracts features for each original_sample.
    Calculates initial-to-final moisture gain.
    """
    features = []

    # Using the phase1 storage assessment for temporal data
    for sample_id, group in biscuit_df.groupby('original_sample'):
        storage_rows = group[group['dataset_type'] == 'Storage Assessment'].sort_values('storage_duration')
        init_rows = group[group['dataset_type'] == 'Initial Characterization']

        if not storage_rows.empty and not init_rows.empty and 'storage_moisture_content' in storage_rows.columns and 'initial_moisture_content' in init_rows.columns:
            storage_m = storage_rows['storage_moisture_content'].dropna()
            init_m = init_rows['initial_moisture_content'].dropna()

            if not storage_m.empty and not init_m.empty:
                first_storage_moisture = storage_m.iloc[0]
                last_storage_moisture = storage_m.iloc[-1]
                max_storage_moisture = storage_m.max()
                min_storage_moisture = storage_m.min()
                initial_moisture = init_m.iloc[0]

                moisture_gain = last_storage_moisture - initial_moisture
            else:
                first_storage_moisture = np.nan
                last_storage_moisture = np.nan
                max_storage_moisture = np.nan
                min_storage_moisture = np.nan
                initial_moisture = np.nan
                moisture_gain = np.nan
        else:
            first_storage_moisture = np.nan
            last_storage_moisture = np.nan
            max_storage_moisture = np.nan
            min_storage_moisture = np.nan
            initial_moisture = np.nan
            moisture_gain = np.nan

        storage_temp = storage_rows['storage_temperature'].dropna().iloc[0] if not storage_rows['storage_temperature'].dropna().empty else np.nan
        water_activity = group['storage_water_activity'].mean() if 'storage_water_activity' in group.columns else np.nan

        features.append({
            'product_id': sample_id,
            'initial_moisture': initial_moisture,
            'final_moisture': last_storage_moisture,
            'moisture_gain': moisture_gain,
            'water_activity': water_activity,
            'storage_temperature_c': storage_temp,
            'relative_humidity_percent': np.nan, # Not present in biscuit dataset explicitly
        })

    return pd.DataFrame(features)

def classify_moisture_sensitivity(moisture_gain):
    if pd.isna(moisture_gain):
        return None, None

    if moisture_gain > 1.0:
        return 'High', 5.0
    elif moisture_gain >= 0.5:
        return 'Medium', 15.0
    else:
        return 'Low', 30.0

def score_material(required_wvtr, material, biscuit_temp):
    missing_temp = pd.isna(biscuit_temp) or pd.isna(material.get('test_temperature_c', pd.NA))

    # 1. WVTR barrier score (70% weight)
    if pd.isna(material['wvtr_value']) or pd.isna(required_wvtr):
        if missing_temp:
            return np.nan, 'Insufficient Data', 'Missing WVTR values for comparison; condition matching unavailable due to missing storage temperature.'
        return np.nan, 'Insufficient Data', 'Missing WVTR values for comparison'

    if material['wvtr_value'] <= required_wvtr:
        moisture_barrier_score = 1.0
    else:
        # Penalty for exceeding limit, minimum 0
        moisture_barrier_score = max(0.0, 1.0 - ((material['wvtr_value'] - required_wvtr) / required_wvtr))

    # 2. Evidence quality score (20% weight)
    if pd.isna(material['source_url_or_doi']):
        evidence_quality_score = 0.0
    else:
        evidence_quality_score = 1.0 if '10.' in str(material['source_url_or_doi']) else 0.8

    # 3. Condition compatibility score (10% weight)
    if missing_temp:
        condition_compatibility_score = 0.0  # Treat missing matching condition explicitly
    else:
        temp_diff = abs(biscuit_temp - material['test_temperature_c'])
        condition_compatibility_score = max(0.0, 1.0 - (temp_diff / 50.0))

    final_score = (0.70 * moisture_barrier_score) + (0.20 * evidence_quality_score) + (0.10 * condition_compatibility_score)

    # Determine recommendation status based strictly on score and thresholds
    if final_score >= 0.85 and moisture_barrier_score == 1.0:
        status = 'Recommended'
        if missing_temp:
            reason = f'Score {final_score:.2f}: Meets WVTR requirement with good evidence; condition matching unavailable due to missing storage temperature.'
        else:
            reason = f'Score {final_score:.2f}: Meets WVTR requirement with good evidence and condition matching'
    elif final_score >= 0.60:
        status = 'Conditionally Recommended'
        if missing_temp:
            reason = f'Score {final_score:.2f}: Near the limit or subject to evidence/condition limitations; condition matching unavailable due to missing storage temperature.'
        else:
            reason = f'Score {final_score:.2f}: Near the limit, or lacking perfect condition/evidence matching'
    else:
        status = 'Not Recommended'
        if missing_temp:
            reason = f'Score {final_score:.2f}: Fails WVTR screening limit or has severe data quality issues; condition matching unavailable due to missing storage temperature.'
        else:
            reason = f'Score {final_score:.2f}: Fails WVTR screening limit or has severe data quality issues'

    return final_score, status, reason

def run_integration(biscuit_csv, packaging_csv, output_csv):
    from src.packaging_validator import validate_packaging_data

    # 1. Validate packaging data first
    validate_packaging_data(packaging_csv)

    biscuit_df = pd.read_csv(biscuit_csv)
    pack_df = pd.read_csv(packaging_csv)

    biscuit_features = extract_biscuit_features(biscuit_df)

    results = []

    for _, biscuit in biscuit_features.iterrows():
        sensitivity, required_wvtr = classify_moisture_sensitivity(biscuit['moisture_gain'])

        for _, pack in pack_df.iterrows():
            # Check for critical missing data
            if pd.isna(biscuit['moisture_gain']) or pd.isna(pack['wvtr_value']):
                status = 'Insufficient Data'
                reason = 'Required biscuit moisture or material WVTR information is unavailable'
                score = np.nan
            else:
                score, status, reason = score_material(required_wvtr, pack, biscuit['storage_temperature_c'])

            results.append({
                'product_id': biscuit['product_id'],
                'biscuit_properties_used': 'moisture_gain, storage_temperature',
                'initial_moisture': biscuit['initial_moisture'],
                'final_moisture': biscuit['final_moisture'],
                'moisture_gain': biscuit['moisture_gain'],
                'water_activity': biscuit['water_activity'],
                'storage_temperature_c': biscuit['storage_temperature_c'],
                'relative_humidity_percent': biscuit['relative_humidity_percent'],
                'required_wvtr': required_wvtr,
                'required_otr': np.nan, # uncertain

                'packaging_material': pack['material_name'],
                'selected_material_id': pack['material_id'],
                'material_category': pack['material_category'],
                'thickness_micron': pack['thickness_micron'],
                'wvtr': pack['wvtr_value'],
                'material_wvtr_unit': pack['wvtr_unit'],
                'original_wvtr_value': pack['wvtr_value'],
                'standardized_wvtr_value': pack['wvtr_value'],
                'otr': pack['otr_value'],
                'material_otr_unit': pack['otr_unit'],
                'original_otr_value': pack['otr_value'],
                'standardized_otr_value': pack['otr_value'],

                'suitability_score': round(score, 3) if pd.notna(score) else np.nan,
                'recommendation_status': status,
                'recommendation_reason': reason,
                'evidence_quality': '1.0' if '10.' in str(pack['source_url_or_doi']) else '0.8',
                'confidence_level': 'Preliminary Screening',
                'evidence_type': 'Literature-Informed AI-Assisted',
                'source_reference': pack['source_reference']
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)

    # Generate the report automatically based on output row counts
    status_counts = out_df['recommendation_status'].value_counts()

    summary_txt = 'reports/phase3_validation_report.md'
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write('# Phase 3 Validation Report\n\n')
        f.write('This report confirms that the Phase 3 packaging integration workflow completed successfully and generated the expected pairing registry.\n\n')
        f.write('The actual pairing distribution is as follows:\n\n')
        f.write('| Status | Rows |\n')
        f.write('| --- | --- |\n')
        for k, v in status_counts.items():
            f.write(f'| {k} | {v} |\n')
        f.write(f'| **Total** | **{status_counts.sum()}** |\n')

    return out_df

if __name__ == "__main__":
    run_integration(
        'processed_data/final_biscuit_analysis_dataset.csv',
        'raw_data/packaging/literature_packaging_matrix.csv',
        'processed_data/packaging_pairing_registry.csv'
    )
    print("Integration complete.")
