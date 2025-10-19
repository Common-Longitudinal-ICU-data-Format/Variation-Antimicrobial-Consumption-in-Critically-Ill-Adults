"""
Create Validation CSV with Sample Data

This script creates a validation CSV file with sample data from PHI_DATA files:
- cohort_icu_first_stay.parquet
- dot_hospital_level.parquet
- afd_patient_level.parquet

Samples 5 random hospitalizations per admission year and combines all data.
"""

import pandas as pd
import polars as pl
from pathlib import Path
import sys


def create_validation_csv(
    phi_data_dir: Path = Path('PHI_DATA'),
    n_samples_per_year: int = 5
):
    """
    Create validation CSV with sample data from cohort, DOT, and AFD files.

    Parameters
    ----------
    phi_data_dir : Path
        Path to PHI_DATA directory (default: 'PHI_DATA')
    n_samples_per_year : int
        Number of random samples per admission year (default: 5)
    """

    print("\n=== Creating Validation CSV ===")
    print(f"PHI_DATA directory: {phi_data_dir}")
    print(f"Samples per year: {n_samples_per_year}\n")

    # Check if directory exists
    if not phi_data_dir.exists():
        print(f"❌ Error: {phi_data_dir} directory does not exist")
        sys.exit(1)

    # ============================================================
    # LOAD DATA FILES
    # ============================================================

    print("Loading data files...")

    # Load cohort
    cohort_path = phi_data_dir / 'cohort_icu_first_stay.parquet'
    if not cohort_path.exists():
        print(f"❌ Error: {cohort_path} not found")
        sys.exit(1)
    cohort_df = pd.read_parquet(cohort_path)
    print(f"✓ Cohort loaded: {len(cohort_df):,} rows, {len(cohort_df.columns)} columns")
    print(f"  Columns: {', '.join(list(cohort_df.columns)[:10])}{'...' if len(cohort_df.columns) > 10 else ''}")

    # Load DOT hospital level
    dot_path = phi_data_dir / 'dot_hospital_level.parquet'
    if not dot_path.exists():
        print(f"❌ Error: {dot_path} not found")
        sys.exit(1)
    dot_df = pl.read_parquet(dot_path).to_pandas()
    print(f"✓ DOT hospital-level loaded: {len(dot_df):,} rows, {len(dot_df.columns)} columns")
    print(f"  Columns: {', '.join(list(dot_df.columns)[:10])}{'...' if len(dot_df.columns) > 10 else ''}")

    # Load AFD
    afd_path = phi_data_dir / 'afd_patient_level.parquet'
    if not afd_path.exists():
        print(f"❌ Error: {afd_path} not found")
        sys.exit(1)
    afd_df = pl.read_parquet(afd_path).to_pandas()
    print(f"✓ AFD loaded: {len(afd_df):,} rows, {len(afd_df.columns)} columns")
    print(f"  Columns: {', '.join(list(afd_df.columns))}")

    # ============================================================
    # DETECT AND STANDARDIZE HOSPITALIZATION ID COLUMN
    # ============================================================

    print("\nDetecting hospitalization ID column...")

    def find_hosp_id_column(df, df_name):
        """Find the hospitalization identifier column in a dataframe."""
        possible_names = ['hospitalization_id', 'hospitalizations', 'hosp_id', 'hospitalization']
        for col in possible_names:
            if col in df.columns:
                return col
        print(f"❌ Error: No hospitalization ID column found in {df_name}")
        print(f"   Available columns: {', '.join(df.columns)}")
        sys.exit(1)

    # Find hospitalization ID column in each dataset
    cohort_hosp_col = find_hosp_id_column(cohort_df, 'cohort')
    dot_hosp_col = find_hosp_id_column(dot_df, 'DOT')
    afd_hosp_col = find_hosp_id_column(afd_df, 'AFD')

    print(f"✓ Hospitalization ID columns detected:")
    print(f"  Cohort: '{cohort_hosp_col}'")
    print(f"  DOT: '{dot_hosp_col}'")
    print(f"  AFD: '{afd_hosp_col}'")

    # Standardize column names to 'hospitalization_id' for easier merging
    HOSP_ID = 'hospitalization_id'

    if cohort_hosp_col != HOSP_ID:
        cohort_df = cohort_df.rename(columns={cohort_hosp_col: HOSP_ID})
        print(f"  Renamed cohort '{cohort_hosp_col}' → '{HOSP_ID}'")

    if dot_hosp_col != HOSP_ID:
        dot_df = dot_df.rename(columns={dot_hosp_col: HOSP_ID})
        print(f"  Renamed DOT '{dot_hosp_col}' → '{HOSP_ID}'")

    if afd_hosp_col != HOSP_ID:
        afd_df = afd_df.rename(columns={afd_hosp_col: HOSP_ID})
        print(f"  Renamed AFD '{afd_hosp_col}' → '{HOSP_ID}'")

    # ============================================================
    # SAMPLE DATA
    # ============================================================

    print("\nSampling data...")

    # Extract admission year
    cohort_df['admission_year'] = pd.to_datetime(cohort_df['admission_dttm']).dt.year

    # Sample n_samples_per_year from each year
    sampled_cohort = (
        cohort_df
        .groupby('admission_year', group_keys=False)
        .apply(lambda x: x.sample(n=min(n_samples_per_year, len(x)), random_state=42))
        .reset_index(drop=True)
    )

    print(f"✓ Sampled {len(sampled_cohort):,} hospitalizations:")
    print(sampled_cohort['admission_year'].value_counts().sort_index())

    # Get sampled hospitalization IDs
    sampled_hosp_ids = sampled_cohort[HOSP_ID].unique()

    # ============================================================
    # FILTER OTHER DATASETS
    # ============================================================

    print("\nFiltering other datasets to sampled hospitalizations...")

    # Filter DOT hospital-level
    dot_sampled = dot_df[dot_df[HOSP_ID].isin(sampled_hosp_ids)].copy()
    print(f"✓ DOT hospital-level filtered: {len(dot_sampled):,} rows")

    # Filter AFD
    afd_sampled = afd_df[afd_df[HOSP_ID].isin(sampled_hosp_ids)].copy()
    print(f"✓ AFD filtered: {len(afd_sampled):,} rows")

    # ============================================================
    # MERGE DATASETS
    # ============================================================

    print("\nMerging datasets...")

    # Start with sampled cohort
    validation_df = sampled_cohort.copy()

    # Merge DOT hospital-level (1:1 merge)
    # Remove duplicate columns (keep from cohort)
    dot_cols_to_merge = [col for col in dot_sampled.columns if col not in validation_df.columns or col == HOSP_ID]
    validation_df = pd.merge(
        validation_df,
        dot_sampled[dot_cols_to_merge],
        on=HOSP_ID,
        how='left'
    )
    print(f"✓ Merged DOT: {len(validation_df):,} rows, {len(validation_df.columns)} columns")

    # Merge AFD (1:1 merge)
    afd_cols_to_merge = [col for col in afd_sampled.columns if col not in validation_df.columns or col == HOSP_ID]
    validation_df = pd.merge(
        validation_df,
        afd_sampled[afd_cols_to_merge],
        on=HOSP_ID,
        how='left'
    )
    print(f"✓ Merged AFD: {len(validation_df):,} rows, {len(validation_df.columns)} columns")

    # Final validation dataframe (no daily ASC)
    validation_final = validation_df.copy()

    # ============================================================
    # REORDER COLUMNS FOR READABILITY
    # ============================================================

    print("\nReordering columns for readability...")

    # Define column order: key identifiers first, then cohort, then DOT/AFD
    key_cols = ['admission_year', HOSP_ID, 'patient_id']

    # Cohort columns (demographics and clinical)
    cohort_cols = [
        col for col in validation_final.columns
        if col in cohort_df.columns and col not in key_cols
    ]

    # DOT columns (PD and antibiotics)
    dot_cols = [
        col for col in validation_final.columns
        if col in dot_sampled.columns and col not in key_cols + cohort_cols
    ]

    # AFD columns
    afd_cols = [
        col for col in validation_final.columns
        if col in afd_sampled.columns and col not in key_cols + cohort_cols + dot_cols
    ]

    # Combine in logical order
    ordered_cols = key_cols + cohort_cols + dot_cols + afd_cols

    # Make sure we didn't miss any columns
    missing_cols = [col for col in validation_final.columns if col not in ordered_cols]
    if missing_cols:
        ordered_cols.extend(missing_cols)

    validation_final = validation_final[ordered_cols].copy()

    # ============================================================
    # SAVE VALIDATION CSV
    # ============================================================

    output_path = phi_data_dir / 'validation_sample.csv'
    validation_final.to_csv(output_path, index=False)

    print(f"\n=== Validation CSV Created ===")
    print(f"Location: {output_path}")
    print(f"Rows: {len(validation_final):,}")
    print(f"Columns: {len(validation_final.columns)}")
    print(f"File size: {output_path.stat().st_size / (1024**2):.2f} MB")
    print(f"\nColumn breakdown:")
    print(f"  Key columns: {len(key_cols)}")
    print(f"  Cohort columns: {len(cohort_cols)}")
    print(f"  DOT columns: {len(dot_cols)}")
    print(f"  AFD columns: {len(afd_cols)}")

    # Show sample years and counts
    print(f"\nSample distribution by year:")
    year_counts = validation_final.groupby('admission_year')[HOSP_ID].nunique()
    for year, count in year_counts.items():
        print(f"  {year}: {count} unique hospitalizations")


if __name__ == "__main__":
    create_validation_csv()
