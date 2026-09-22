# Phase 1: Dataset and Target Variable Analysis

## Dataset Selected
processed_data/packaging_pairing_registry.csv

## Features Identified
**Valid Independent Input Features:**
1. initial_moisture (Numerical)
2. final_moisture (Numerical)
3. moisture_gain (Numerical, derived from initial and final moisture)
4. packaging_material (Categorical)
5. thickness_micron (Numerical)
6. wvtr (Numerical)
7. otr (Numerical)
8. evidence_quality (Numerical/Categorical)

**Excluded Features:**
- storage_temperature_c: 100% missing. Excluded from training and UI.
- relative_humidity_percent: 100% missing. Excluded from training and UI.
- required_wvtr: Excluded to prevent direct target leakage, as this is an intermediate deterministic calculation step.
- suitability_score: This is effectively the target/rule output. Excluded.

## Target Variable
- **Target Column**: recommendation_status
- **Classes Supported**: Recommended, Conditionally Recommended, Not Recommended.
- **Note on Insufficient Data**: The 48 records with Insufficient Data will be excluded from ML training. Insufficient Data is a data availability state, not a scientific compatibility class. The model should only predict real compatibility.

## ML Approach
This will be modeled as a **Multi-class Classification** problem.

## Limitations
The model is trained on rule-generated labels (recommendation_status), so it is learning the decision boundaries of the existing deterministic logic rather than independent empirical ground truth.
