import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob

def load_data():
    files = glob.glob('processed_data/*.csv')
    dfs = {}
    for path in files:
        fname = os.path.basename(path)
        dfs[fname] = pd.read_csv(path)
    return dfs

def audit_datasets(dfs):
    with open('reports/phase2_data_profile.md', 'w') as f:
        f.write("# Phase 2 Data Profile\\n\\n")
        
        # Verify source counts before deduplication (Requirement 3)
        if 'merged_raw_dataset.csv' in dfs:
            df_merged = dfs['merged_raw_dataset.csv']
            f.write("## Source Counts Before Deduplication\\n")
            f.write("| source_archive | source_dataset | source_file | dataset_type | raw_row_count | retained_row_count | duplicate_row_count |\\n")
            f.write("|---|---|---|---|---|---|---|\\n")
            groups = df_merged.groupby(['source_archive', 'source_dataset', 'source_file', 'dataset_type'])
            for (arc, ds, sf, dt), grp in groups:
                raw = len(grp)
                retained = len(grp[grp['duplicate_status'] == 'unique'])
                dups = len(grp[grp['duplicate_status'] != 'unique'])
                f.write(f"| {arc} | {ds} | {sf} | {dt} | {raw} | {retained} | {dups} |\\n")
            f.write("\\n")
        
        for name, df in dfs.items():
            if name not in ['phase1_initial_characterization.csv', 'phase1_water_adsorption.csv', 'phase1_storage_assessment.csv']:
                continue
            f.write(f"## Dataset: {name}\\n")
            f.write(f"- Rows: {len(df)}\\n")
            f.write(f"- Columns: {len(df.columns)}\\n\\n")
            f.write("### Columns\\n")
            for col in df.columns:
                n_miss = df[col].isna().sum()
                p_miss = (n_miss / len(df)) * 100
                n_uniq = df[col].nunique()
                dtype = df[col].dtype
                f.write(f"#### {col}\\n")
                f.write(f"- Type: {dtype}\\n")
                f.write(f"- Missing: {n_miss} ({p_miss:.2f}%)\\n")
                f.write(f"- Unique: {n_uniq}\\n")
                if pd.api.types.is_numeric_dtype(df[col]):
                    f.write(f"- Min: {df[col].min()}\\n")
                    f.write(f"- Max: {df[col].max()}\\n")
                    f.write(f"- Mean: {df[col].mean()}\\n")
                    f.write(f"- Median: {df[col].median()}\\n")
                    f.write(f"- StdDev: {df[col].std()}\\n")
            f.write("\\n")

def variable_dictionary(dfs):
    records = []
    
    source_mappings = {
        'initial_moisture_content': 'Moisture content (% wet basis)',
        'adsorption_water_activity': 'Water activity',
        'adsorption_moisture_content': 'Moisture content (% wet basis)',
        'storage_duration': 'Time (day)',
        'L_star': 'L*',
        'a_star': 'a*',
        'b_star': 'b*',
        'storage_hardness': 'Hardness (N)',
        'storage_moisture_content': 'Moisture content (% wet basis)',
        'storage_water_activity': 'Water activity',
        'sensory_quality': 'Sensory quality'
    }
    
    for name, df in dfs.items():
        if name not in ['phase1_initial_characterization.csv', 'phase1_water_adsorption.csv', 'phase1_storage_assessment.csv']:
            continue
        for col in df.columns:
            n_miss = df[col].isna().sum()
            n_uniq = df[col].nunique()
            min_val = df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else ''
            max_val = df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else ''
            
            src_col = source_mappings.get(col, col)
            
            records.append({
                'dataset_type': name,
                'variable_name': col,
                'data_type': str(df[col].dtype),
                'unit_if_known': 'Refer to source column mapping',
                'description': f"Extracted from {src_col}",
                'source_column_or_mapping': src_col,
                'missing_count': n_miss,
                'unique_count': n_uniq,
                'min_value': min_val,
                'max_value': max_val
            })
    pd.DataFrame(records).to_csv('reports/phase2_variable_dictionary.csv', index=False)

def eda_summary(dfs):
    records = []
    for name, df in dfs.items():
        if name not in ['phase1_initial_characterization.csv', 'phase1_water_adsorption.csv', 'phase1_storage_assessment.csv']:
            continue
        for col in df.columns:
            n_miss = df[col].isna().sum()
            p_miss = (n_miss / len(df)) * 100
            n_uniq = df[col].nunique()
            
            rec = {
                'dataset_type': name,
                'variable': col,
                'observation_count': len(df),
                'missing_count': n_miss,
                'missing_percentage': p_miss,
                'unique_count': n_uniq
            }
            if pd.api.types.is_numeric_dtype(df[col]):
                rec['mean'] = df[col].mean()
                rec['median'] = df[col].median()
                rec['standard_deviation'] = df[col].std()
                rec['minimum'] = df[col].min()
                rec['maximum'] = df[col].max()
                
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
                rec['outlier_candidate_count'] = len(outliers)
            else:
                rec['mean'] = ''
                rec['median'] = ''
                rec['standard_deviation'] = ''
                rec['minimum'] = ''
                rec['maximum'] = ''
                rec['outlier_candidate_count'] = 0
            records.append(rec)
    pd.DataFrame(records).to_csv('reports/phase2_eda_summary.csv', index=False)

def relationship_analysis(dfs):
    records = []
    exclude_cols = ['source_row', 'record_id', 'replicate_id', 'duplicate_group_id', 'duplicate_status']
    for name, df in dfs.items():
        if name not in ['phase1_initial_characterization.csv', 'phase1_water_adsorption.csv', 'phase1_storage_assessment.csv']:
            continue
        
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
        for i in range(len(num_cols)):
            for j in range(i+1, len(num_cols)):
                col_x = num_cols[i]
                col_y = num_cols[j]
                
                valid_df = df[[col_x, col_y]].dropna()
                valid_count = len(valid_df)
                
                if valid_count > 2:
                    corr = valid_df[col_x].corr(valid_df[col_y], method='pearson')
                    records.append({
                        'dataset_type': name,
                        'variable_x': col_x,
                        'variable_y': col_y,
                        'valid_pair_count': valid_count,
                        'correlation_method': 'pearson',
                        'correlation_value': corr,
                        'interpretation_status': 'calculated'
                    })
                else:
                    records.append({
                        'dataset_type': name,
                        'variable_x': col_x,
                        'variable_y': col_y,
                        'valid_pair_count': valid_count,
                        'correlation_method': '',
                        'correlation_value': '',
                        'interpretation_status': 'Insufficient observations'
                    })
    pd.DataFrame(records).to_csv('reports/phase2_relationships.csv', index=False)

def plot_and_report(dfs):
    os.makedirs('reports/phase2_figures', exist_ok=True)
    
    if 'phase1_initial_characterization.csv' in dfs:
        df = dfs['phase1_initial_characterization.csv']
        with open('reports/phase2_initial_characterization_eda.md', 'w') as f:
            f.write("# Initial Characterization EDA\\n")
            f.write(f"Total Rows: {len(df)}\\n")
            f.write("Data contains baseline moisture content without making any causation assumptions.\\n")
        if 'initial_moisture_content' in df.columns:
            valid_data = pd.to_numeric(df['initial_moisture_content'], errors='coerce').dropna()
            if not valid_data.empty:
                plt.figure()
                sns.histplot(valid_data, color='blue')
                plt.title('Initial Moisture Content')
                plt.savefig('reports/phase2_figures/initial_moisture.png')
                plt.close()
            
    if 'phase1_water_adsorption.csv' in dfs:
        df = dfs['phase1_water_adsorption.csv']
        with open('reports/phase2_water_adsorption_eda.md', 'w') as f:
            f.write("# Water Adsorption EDA\\n")
            f.write(f"Total Rows: {len(df)}\\n")
            f.write("Scatter plot represents empirical relation. No mathematical isotherms have been explicitly fitted.\\n")
        if 'adsorption_water_activity' in df.columns and 'adsorption_moisture_content' in df.columns:
            df['adsorption_water_activity'] = pd.to_numeric(df['adsorption_water_activity'], errors='coerce')
            df['adsorption_moisture_content'] = pd.to_numeric(df['adsorption_moisture_content'], errors='coerce')
            valid_df = df[['adsorption_water_activity', 'adsorption_moisture_content']].dropna()
            if not valid_df.empty:
                plt.figure()
                sns.scatterplot(x='adsorption_water_activity', y='adsorption_moisture_content', data=valid_df, color='blue')
                plt.title('Water Adsorption Curve')
                plt.savefig('reports/phase2_figures/water_adsorption.png')
                plt.close()
            
    if 'phase1_storage_assessment.csv' in dfs:
        df = dfs['phase1_storage_assessment.csv']
        with open('reports/phase2_storage_assessment_eda.md', 'w') as f:
            f.write("# Storage Assessment EDA\\n")
            f.write(f"Total Rows: {len(df)}\\n")
            f.write("No causative claims made regarding storage duration and quality degradation.\\n")
        if 'storage_duration' in df.columns:
            valid_data = pd.to_numeric(df['storage_duration'], errors='coerce').dropna()
            if not valid_data.empty:
                plt.figure()
                sns.histplot(valid_data, color='blue')
                plt.title('Storage Duration')
                plt.savefig('reports/phase2_figures/storage_duration.png')
                plt.close()

def data_quality_report():
    content = """# Phase 2 Data Quality Report

## Missing data risks
Nulls natively reflect experimental structure limitations. No automatic or silent imputation has been applied.

## Possible outliers
Statistical outlier candidate requiring domain validation. 

## Limitations
No inference is made regarding packaging material performance, barrier properties, oxygen permeability, WVTR, or overall material superiority.

## Reproducibility
Data generated completely deterministically from ZIP extraction payloads.
"""
    with open('reports/phase2_data_quality_report.md', 'w') as f:
        f.write(content)

def main():
    os.makedirs('reports', exist_ok=True)
    dfs = load_data()
    audit_datasets(dfs)
    variable_dictionary(dfs)
    eda_summary(dfs)
    relationship_analysis(dfs)
    plot_and_report(dfs)
    data_quality_report()

if __name__ == '__main__':
    main()
