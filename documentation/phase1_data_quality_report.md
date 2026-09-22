# Phase 1 Data Quality Report

## Extracted Data Sources
- **Mendeley_Achira_Storage_1**: Dataset on storage stability and quality changes of achira biscuits
  - Files used: Initial characterization_Achira biscuits.xlsx, Water Adsorption Isotherms_Achira biscuits.xlsx, StorageAssesmentAchiraBiscuits.xlsx
- **Mendeley_Achira_WaterAds_2**: Experimental water adsorption isotherms and ATR-FTIR analysis in Achira biscuits
  - Files used: Initial characterization_Achira biscuits.xlsx, Water Adsorption Isotherms_Achira biscuits.xlsx

## Summary Metrics
- **Total records extracted:** 4498
- **Final deduplicated records:** 1343

## Data Handling and Corrections Applied
- **Separate Datasets**: Raw data is strictly categorized into Initial Characterization, Water Adsorption Isotherms, and Storage Assessment before any harmonized merging.
- **Reproducibility**: The pipeline generates output deterministically from the raw data.
- **Stable Identifiers**: Deterministic hashing is used to create stable `record_id` strings based on provenance data (source file, dataset, sheet, row, and sample).
- **Forward Filling**: Storage duration is strictly forward-filled within the bounds of a specific dataset file to prevent cross-experiment leakage.
- **Duplicate Detection**: The duplicate status logic strictly relies on scientific fields and segregates records into unique, confirmed_duplicate, and possible_duplicate statuses. Provenance fields are preserved and ignored in deduplication.
- **Missing Variables**: Synthetic parameters like fat_content and protein_content have been intentionally stripped as they are not present in the base source data. Summary statistics in source sheets have been effectively excluded.
