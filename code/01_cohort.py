import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # FLAME-ICU: Cohort Generation

    This notebook generates the ICU cohort for antimicrobial consumption analysis.

    ## Objective
    Generate a cohort table containing:
    - `patient_id`, `hospitalization_id`
    - `start_dttm`, `end_dttm`: First ICU stay timestamps
    - `location_type`: ICU type (general_icu, cardiac_icu, etc.)
    - Demographics: `sex_category`, `ethnicity_category`, `race_category`, `race_ethnicity`, `language_category`
    - Clinical outcomes: `hospital_los_days`, `icu_los_days`, `inpatient_mortality`, `icu_mortality`
    - Vital signs: `highest_temperature`, `lowest_temperature`, `lowest_map` (during ICU stay)
    - Laboratory values: `highest_wbc`, `highest_creatinine` (during ICU stay)
    - Respiratory support: `NIPPV_ever`, `HFNO_ever` (device usage during ICU stay)
    - Medications: `vasopressor_ever`, `no_of_vasopressor` (vasopressor usage during ICU stay)

    ## Cohort Criteria
    - Adults (≥18 years)
    - First ICU admission (`location_category == 'icu'`)
    - Years: 2018-2024
    - First ICU stay only
    - ICU LOS > 6 hours (0.25 days)

    ## Clinical Features Processing
    All features are filtered to ICU stay window (start_dttm to end_dttm):

    **Vital Signs:**
    - Loaded using clifpy Vitals table
    - Outlier handling applied using clifpy built-in functions
    - Categories: temp_c, map
    - Aggregations: min, max

    **Laboratory Values:**
    - Loaded using clifpy Labs table
    - Outlier handling applied using clifpy built-in functions
    - Categories: wbc, creatinine
    - Aggregations: max (worst values)
    - Units: WBC in 10^3/μL, Creatinine in mg/dL

    **Respiratory Support:**
    - Loaded using clifpy RespiratorySupport table
    - All device categories loaded (no category filtering)
    - Devices tracked: NIPPV, High Flow NC (HFNO)
    - Metrics: binary flags (ever used during ICU stay)

    **Medications (Vasopressors):**
    - Loaded using clifpy MedicationAdminContinuous table
    - Categories: norepinephrine, epinephrine, phenylephrine, angiotensin, vasopressin, dopamine, dobutamine, milrinone, isoproterenol
    - Metrics: binary flag (ever used) + count of unique vasopressor categories
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Setup and Configuration""")
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from clifpy.tables import Adt, Hospitalization, Patient, MedicationAdminContinuous, Labs, Vitals, RespiratorySupport
    from clifpy.clif_orchestrator import ClifOrchestrator
    from clifpy.utils.outlier_handler import apply_outlier_handling
    import warnings
    warnings.filterwarnings('ignore')

    print("=== FLAME-ICU: Cohort Generation ===")
    print("Setting up environment...")
    return (
        Adt,
        ClifOrchestrator,
        Hospitalization,
        Labs,
        MedicationAdminContinuous,
        Patient,
        RespiratorySupport,
        Vitals,
        apply_outlier_handling,
        pd,
    )


@app.cell
def _(mo):
    mo.md(r"""## Load Data""")
    return


@app.cell
def _(Adt, Hospitalization, Patient):
    # Load required tables using clifpy config file
    print("Loading required tables...")

    # Load ADT data
    adt_table = Adt.from_file(config_path='clif_config.json')
    adt_df = adt_table.df.copy()
    print(f"✓ ADT data loaded: {len(adt_df):,} records")

    # Load hospitalization data
    hosp_table = Hospitalization.from_file(config_path='clif_config.json')
    hosp_df = hosp_table.df.copy()
    print(f"✓ Hospitalization data loaded: {len(hosp_df):,} records")

    # Load patient data
    patient_table = Patient.from_file(config_path='clif_config.json')
    patient_df = patient_table.df.copy()
    print(f"✓ Patient data loaded: {len(patient_df):,} records")
    return adt_df, hosp_df, patient_df


@app.cell
def _(mo):
    mo.md(r"""## Filter First ICU Stays""")
    return


@app.cell
def _(adt_df, hosp_df, pd):
    # Merge ADT with hospitalization data
    print("Merging ADT with hospitalization data...")

    icu_data = pd.merge(
        adt_df[['hospitalization_id', 'location_category', 'location_type', 'in_dttm', 'out_dttm']],
        hosp_df[['patient_id', 'hospitalization_id', 'age_at_admission', 'admission_dttm', 'discharge_dttm','discharge_category']],
        on='hospitalization_id',
        how='inner'
    )

    # Normalize location_category to lowercase for consistent matching
    icu_data['location_category'] = icu_data['location_category'].str.lower()

    print(f"✓ Merged data: {len(icu_data):,} records")
    return (icu_data,)


@app.cell
def _(icu_data, pd):
    # Apply cohort filters
    print("Applying cohort filters...")

    # Convert datetime columns
    datetime_cols = ['in_dttm', 'out_dttm', 'admission_dttm', 'discharge_dttm']
    for col in datetime_cols:
        icu_data[col] = pd.to_datetime(icu_data[col])

    # Filter for ICU admissions (2018-2024, adults ≥18)
    # Exclude records where ICU discharge is in 2025
    icu_filtered = icu_data[
        (icu_data['location_category'] == 'icu') &
        (icu_data['admission_dttm'].dt.year >= 2018) &
        (icu_data['admission_dttm'].dt.year <= 2024) &
        (icu_data['out_dttm'].dt.year <= 2024) &
        (icu_data['age_at_admission'] >= 18) &
        (icu_data['age_at_admission'].notna())
    ].copy()

    print(f"✓ ICU admissions (2018-2024, adults, ICU discharge ≤2024): {len(icu_filtered):,} records")

    # Get first ICU stay per hospitalization
    first_icu = icu_filtered.sort_values('in_dttm').groupby('hospitalization_id').first().reset_index()

    print(f"✓ First ICU stays: {len(first_icu):,} hospitalizations")
    return (first_icu,)


@app.cell
def _(mo):
    mo.md(r"""## Add Demographics""")
    return


@app.cell
def _(first_icu, patient_df, pd):
    # Merge with patient demographics
    print("Adding patient demographics...")

    cohort_df = pd.merge(
        first_icu[['patient_id', 'hospitalization_id', 'age_at_admission', 'in_dttm', 'out_dttm',     'location_type','admission_dttm', 'discharge_dttm','discharge_category']],
        patient_df[['patient_id', 'sex_category', 'ethnicity_category', 'race_category', 'language_category', 'death_dttm']],
        on='patient_id',
        how='left'
    )

    # Rename columns for clarity
    cohort_df = cohort_df.rename(columns={
        'in_dttm': 'start_dttm',
        'out_dttm': 'end_dttm'
    })

    # Recode language_category to English/Non-English
    cohort_df['language_category'] = cohort_df['language_category'].str.lower()
    cohort_df['language_category'] = cohort_df['language_category'].apply(
        lambda x: 'english' if x == 'english' else 'non-english'
    )

    # Create race_ethnicity column
    def categorize_race_ethnicity(row):
        ethnicity = str(row['ethnicity_category']).lower() if pd.notna(row['ethnicity_category']) else 'unknown'
        race = str(row['race_category']).lower() if pd.notna(row['race_category']) else 'unknown'

        # Check Non-Hispanic FIRST (more specific than "hispanic")
        if 'non-hispanic' in ethnicity or 'not hispanic' in ethnicity:
            if 'white' in race:
                return 'Non-Hispanic White'
            elif 'black' in race or 'african american' in race:
                return 'Non-Hispanic Black'
            elif 'asian' in race:
                return 'Non-Hispanic Asian'
            else:
                return 'Other'

        # Now check for Hispanic (less specific)
        if 'hispanic' in ethnicity:
            return 'Hispanic'

        # If ethnicity is Other
        if ethnicity == 'other':
            return 'Other'

        # If ethnicity is Unknown or not reported
        if ethnicity in ['unknown', 'not reported', 'nan']:
            return 'Not Reported'

        # Default to Other
        return 'Other'

    cohort_df['race_ethnicity'] = cohort_df.apply(categorize_race_ethnicity, axis=1)
    print(f"\n=== Race/Ethnicity Distribution ===")
    print(cohort_df['race_ethnicity'].value_counts())

    # Calculate Hospital Length of Stay (LOS) in days
    cohort_df['hospital_los_days'] = (cohort_df['discharge_dttm'] - cohort_df['admission_dttm']).dt.total_seconds() / (24 * 3600)

    # Calculate Inpatient Mortality (binary: 1 = died, 0 = survived)
    cohort_df['inpatient_mortality'] = cohort_df['discharge_category'].fillna('').str.lower().apply(
        lambda x: 1 if any(term in x for term in ['expired', 'dead', 'death', 'deceased']) else 0
    )

    # Calculate ICU Length of Stay (LOS) in days
    cohort_df['icu_los_days'] = (cohort_df['end_dttm'] - cohort_df['start_dttm']).dt.total_seconds() / (24 * 3600)

    # Calculate ICU Mortality (binary: 1 = died during ICU stay, 0 = did not die during ICU stay)
    # Convert death_dttm to datetime if not already
    cohort_df['death_dttm'] = pd.to_datetime(cohort_df['death_dttm'])

    # Check if death_dttm falls within ICU stay window (start_dttm to end_dttm)
    cohort_df['icu_mortality'] = (
        (cohort_df['death_dttm'].notna()) &  # death_dttm is not null
        (cohort_df['death_dttm'] >= cohort_df['start_dttm']) &  # death occurred after/at ICU start
        (cohort_df['death_dttm'] <= cohort_df['end_dttm'])  # death occurred before/at ICU end
    ).astype(int)

    print(f"\n=== ICU Mortality ===")
    print(f"Deaths during ICU stay: {cohort_df['icu_mortality'].sum():,} ({cohort_df['icu_mortality'].mean()*100:.2f}%)")
    print(f"Survived ICU stay: {(cohort_df['icu_mortality'] == 0).sum():,}")

    # Remove encounters with short ICU LOS (≤6 hours / 0.25 days)
    cohort_before_los_filter = len(cohort_df)
    cohort_df = cohort_df[cohort_df['icu_los_days'] > 0.25].copy()
    removed_short_los = cohort_before_los_filter - len(cohort_df)
    if removed_short_los > 0:
        print(f"⚠ Removed {removed_short_los:,} encounters with short ICU LOS (≤6 hours / 0.25 days)")

    # Reorder columns (vital columns will be added after vitals processing)
    base_columns = [
        'patient_id',
        'hospitalization_id',
        'age_at_admission',
        'admission_dttm', 'discharge_dttm','discharge_category',
        'start_dttm',
        'end_dttm',
        'hospital_los_days',
        'icu_los_days',
        'inpatient_mortality',
        'icu_mortality',
        'location_type',
        'sex_category',
        'ethnicity_category',
        'race_category',
        'race_ethnicity',
        'language_category'
    ]

    # Only select columns that exist in cohort_df at this point
    cohort_df = cohort_df[[col for col in base_columns if col in cohort_df.columns]]

    print(f"✓ Final cohort: {len(cohort_df):,} hospitalizations")
    return (cohort_df,)


@app.cell
def _(mo):
    mo.md(r"""## Load and Process Vitals""")
    return


@app.cell
def _(Vitals, apply_outlier_handling, cohort_df):
    # Load vitals data using clifpy Vitals table (after cohort is created to reduce memory usage)
    print("Loading vitals data for cohort using clifpy...")

    # Define vital categories
    vital_categories = ['temp_c', 'map']

    # Extract cohort hospitalization IDs
    cohort_hosp_ids = cohort_df['hospitalization_id'].astype(str).unique().tolist()
    print(f"Loading vitals for {len(cohort_hosp_ids):,} hospitalizations")
    print(f"Vital categories: {vital_categories}")

    # Load vitals table with filters
    vitals_table = Vitals.from_file(
        config_path='clif_config.json',
        filters={
            'hospitalization_id': cohort_hosp_ids,
            'vital_category': vital_categories
        },
        columns=['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value']
    )

    print(f"✓ Vitals loaded: {len(vitals_table.df):,} records")

    # Apply outlier handling using clifpy
    print("Applying outlier handling to vitals...")
    apply_outlier_handling(vitals_table)
    print(f"✓ Outlier handling applied")
    print(f"  Records after outlier removal: {len(vitals_table.df):,}")

    # Get vitals dataframe
    vitals_df = vitals_table.df.copy()
    return (vitals_df,)


@app.cell
def _(cohort_df, pd, vitals_df):
    # Filter vitals to ICU stay windows and calculate aggregates
    print("Filtering vitals to ICU stay windows...")

    # Merge vitals with cohort to get ICU stay windows
    vitals_with_windows = pd.merge(
        vitals_df,
        cohort_df[['hospitalization_id', 'start_dttm', 'end_dttm']],
        on='hospitalization_id',
        how='inner'
    )

    # Filter vitals to ICU stay window (start_dttm <= recorded_dttm <= end_dttm)
    vitals_icu_window = vitals_with_windows[
        (vitals_with_windows['recorded_dttm'] >= vitals_with_windows['start_dttm']) &
        (vitals_with_windows['recorded_dttm'] <= vitals_with_windows['end_dttm'])
    ].copy()

    print(f"✓ Vitals filtered to ICU windows: {len(vitals_icu_window):,} records")

    # Calculate aggregates by hospitalization_id and vital_category
    print("Calculating vital sign aggregates...")

    # Pivot to get temp_c and map as separate columns
    vitals_pivot = vitals_icu_window.pivot_table(
        index='hospitalization_id',
        columns='vital_category',
        values='vital_value',
        aggfunc={'vital_value': ['min', 'max']}
    )

    # Flatten column names
    vitals_pivot.columns = ['_'.join(col).strip() for col in vitals_pivot.columns.values]
    vitals_pivot = vitals_pivot.reset_index()

    # Rename columns to match requirements
    vitals_column_mapping = {
        'max_temp_c': 'highest_temperature',
        'min_temp_c': 'lowest_temperature',
        'min_map': 'lowest_map'
    }

    # Only rename columns that exist
    vitals_existing_mappings = {k: v for k, v in vitals_column_mapping.items() if k in vitals_pivot.columns}
    vitals_pivot = vitals_pivot.rename(columns=vitals_existing_mappings)

    print(f"✓ Vital aggregates calculated for {len(vitals_pivot):,} hospitalizations")
    print(f"  Columns: {[col for col in vitals_pivot.columns if col != 'hospitalization_id']}")

    # Merge vitals back to cohort_df
    cohort_with_vitals = pd.merge(
        cohort_df,
        vitals_pivot,
        on='hospitalization_id',
        how='left'
    )

    print(f"✓ Vitals merged to cohort: {len(cohort_with_vitals):,} hospitalizations")

    # Reorder columns with vitals included
    final_column_order = [
        'patient_id',
        'hospitalization_id',
        'age_at_admission',
        'admission_dttm', 'discharge_dttm','discharge_category',
        'start_dttm',
        'end_dttm',
        'hospital_los_days',
        'icu_los_days',
        'inpatient_mortality',
        'icu_mortality',
        'highest_temperature',
        'lowest_temperature',
        'lowest_map',
        'location_type',
        'sex_category',
        'ethnicity_category',
        'race_category',
        'race_ethnicity',
        'language_category'
    ]

    # Only select columns that exist
    cohort_with_vitals_df = cohort_with_vitals[[col for col in final_column_order if col in cohort_with_vitals.columns]]

    # Return final cohort with vitals (renamed to avoid circular dependency)
    return (cohort_with_vitals_df,)


@app.cell
def _(mo):
    mo.md(r"""## Load and Process Medications (Vasopressors)""")
    return


@app.cell
def _(MedicationAdminContinuous, cohort_df):
    # Load medications data for vasopressors
    print("Loading medication data for vasopressors using clifpy...")

    # Define vasopressor categories
    vasopressor_categories = [
        'norepinephrine', 'epinephrine', 'phenylephrine', 'angiotensin',
        'vasopressin', 'dopamine', 'dobutamine', 'milrinone', 'isoproterenol'
    ]

    # Extract cohort hospitalization IDs
    cohort_hosp_ids_meds = cohort_df['hospitalization_id'].astype(str).unique().tolist()
    print(f"Loading medications for {len(cohort_hosp_ids_meds):,} hospitalizations")
    print(f"Vasopressor categories: {len(vasopressor_categories)}")

    # Load medications table with filters
    med_table = MedicationAdminContinuous.from_file(
        config_path='clif_config.json',
        filters={
            'hospitalization_id': cohort_hosp_ids_meds,
            'med_category': vasopressor_categories
        },
        columns=['hospitalization_id', 'admin_dttm', 'med_category']
    )

    meds_df = med_table.df.copy()
    print(f"✓ Medications loaded: {len(meds_df):,} records")
    return (meds_df,)


@app.cell
def _(cohort_df, meds_df, pd):
    # Filter medications to ICU stay windows and calculate vasopressor metrics
    print("Filtering medications to ICU stay windows...")

    # Merge medications with cohort to get ICU stay windows
    meds_with_windows = pd.merge(
        meds_df,
        cohort_df[['hospitalization_id', 'start_dttm', 'end_dttm']],
        on='hospitalization_id',
        how='inner'
    )

    # Convert datetime column
    meds_with_windows['admin_dttm'] = pd.to_datetime(meds_with_windows['admin_dttm'])

    # Filter medications to ICU stay window (start_dttm <= admin_dttm <= end_dttm)
    meds_icu_window = meds_with_windows[
        (meds_with_windows['admin_dttm'] >= meds_with_windows['start_dttm']) &
        (meds_with_windows['admin_dttm'] <= meds_with_windows['end_dttm'])
    ].copy()

    print(f"✓ Medications filtered to ICU windows: {len(meds_icu_window):,} records")

    # Calculate vasopressor metrics per hospitalization
    print("Calculating vasopressor metrics...")

    # Count unique vasopressor categories per hospitalization
    vaso_summary = meds_icu_window.groupby('hospitalization_id').agg({
        'med_category': lambda x: x.nunique()  # count unique vasopressor categories
    }).reset_index()
    vaso_summary.columns = ['hospitalization_id', 'no_of_vasopressor']

    # Add binary flag: 1 if any vasopressor used, 0 otherwise
    vaso_summary['vasopressor_ever'] = 1

    print(f"✓ Vasopressor metrics calculated for {len(vaso_summary):,} hospitalizations")
    print(f"  Vasopressor usage distribution:")
    print(vaso_summary['no_of_vasopressor'].value_counts().sort_index().to_dict())
    return (vaso_summary,)


@app.cell
def _(cohort_with_vitals_df, pd, vaso_summary):
    # Merge vasopressor metrics back to cohort
    print("Merging vasopressor metrics to cohort...")

    cohort_with_meds = pd.merge(
        cohort_with_vitals_df,
        vaso_summary,
        on='hospitalization_id',
        how='left'
    )

    # Fill NaN (no vasopressors) with 0
    cohort_with_meds['vasopressor_ever'] = cohort_with_meds['vasopressor_ever'].fillna(0).astype(int)
    cohort_with_meds['no_of_vasopressor'] = cohort_with_meds['no_of_vasopressor'].fillna(0).astype(int)

    print(f"✓ Vasopressor metrics merged to cohort: {len(cohort_with_meds):,} hospitalizations")
    print(f"  Hospitalizations with vasopressors: {(cohort_with_meds['vasopressor_ever'] == 1).sum():,} ({(cohort_with_meds['vasopressor_ever'] == 1).mean()*100:.1f}%)")
    print(f"  Hospitalizations without vasopressors: {(cohort_with_meds['vasopressor_ever'] == 0).sum():,} ({(cohort_with_meds['vasopressor_ever'] == 0).mean()*100:.1f}%)")

    # Reorder columns with vasopressor columns included
    final_column_order_with_meds = [
        'patient_id',
        'hospitalization_id',
        'age_at_admission',
        'admission_dttm', 'discharge_dttm','discharge_category',
        'start_dttm',
        'end_dttm',
        'hospital_los_days',
        'icu_los_days',
        'inpatient_mortality',
        'icu_mortality',
        'highest_temperature',
        'lowest_temperature',
        'lowest_map',
        'vasopressor_ever',
        'no_of_vasopressor',
        'location_type',
        'sex_category',
        'ethnicity_category',
        'race_category',
        'race_ethnicity',
        'language_category'
    ]

    # Only select columns that exist
    cohort_final = cohort_with_meds[[col for col in final_column_order_with_meds if col in cohort_with_meds.columns]]

    # Return final cohort with all features
    return (cohort_final,)


@app.cell
def _(mo):
    mo.md(r"""## Load and Process Labs (WBC, Creatinine)""")
    return


@app.cell
def _(Labs, apply_outlier_handling, cohort_df):
    # Load labs data for WBC and creatinine
    print("Loading labs data (WBC, creatinine) using clifpy...")

    # Define lab categories
    lab_categories = ['wbc', 'creatinine']

    # Extract cohort hospitalization IDs
    cohort_hosp_ids_labs = cohort_df['hospitalization_id'].astype(str).unique().tolist()
    print(f"Loading labs for {len(cohort_hosp_ids_labs):,} hospitalizations")
    print(f"Lab categories: {lab_categories}")

    # Load labs table with filters
    labs_table = Labs.from_file(
        config_path='clif_config.json',
        filters={
            'hospitalization_id': cohort_hosp_ids_labs,
            'lab_category': lab_categories
        },
        columns=['hospitalization_id', 'lab_result_dttm', 'lab_category', 'lab_value_numeric']
    )

    print(f"✓ Labs loaded: {len(labs_table.df):,} records")

    # Apply outlier handling using clifpy
    print("Applying outlier handling to labs...")
    apply_outlier_handling(labs_table)
    print(f"✓ Outlier handling applied")
    print(f"  Records after outlier removal: {len(labs_table.df):,}")

    # Get labs dataframe
    labs_df = labs_table.df.copy()
    return (labs_df,)


@app.cell
def _(cohort_df, labs_df, pd):
    # Filter labs to ICU stay windows and calculate max values
    print("Filtering labs to ICU stay windows...")

    # Merge labs with cohort to get ICU stay windows
    labs_with_windows = pd.merge(
        labs_df,
        cohort_df[['hospitalization_id', 'start_dttm', 'end_dttm']],
        on='hospitalization_id',
        how='inner'
    )

    # Convert datetime column
    labs_with_windows['lab_result_dttm'] = pd.to_datetime(labs_with_windows['lab_result_dttm'])

    # Filter labs to ICU stay window (start_dttm <= lab_result_dttm <= end_dttm)
    labs_icu_window = labs_with_windows[
        (labs_with_windows['lab_result_dttm'] >= labs_with_windows['start_dttm']) &
        (labs_with_windows['lab_result_dttm'] <= labs_with_windows['end_dttm'])
    ].copy()

    print(f"✓ Labs filtered to ICU windows: {len(labs_icu_window):,} records")

    # Calculate max values per hospitalization and lab_category
    print("Calculating maximum lab values...")

    # Pivot to get wbc and creatinine as separate columns
    labs_pivot = labs_icu_window.pivot_table(
        index='hospitalization_id',
        columns='lab_category',
        values='lab_value_numeric',
        aggfunc='max'
    ).reset_index()

    # Rename columns to match requirements
    labs_column_mapping = {
        'wbc': 'highest_wbc',
        'creatinine': 'highest_creatinine'
    }

    # Only rename columns that exist
    labs_existing_mappings = {k: v for k, v in labs_column_mapping.items() if k in labs_pivot.columns}
    labs_pivot = labs_pivot.rename(columns=labs_existing_mappings)

    print(f"✓ Lab aggregates calculated for {len(labs_pivot):,} hospitalizations")
    print(f"  Columns: {[col for col in labs_pivot.columns if col != 'hospitalization_id']}")
    return (labs_pivot,)


@app.cell
def _(cohort_final, labs_pivot, pd):
    # Merge labs back to cohort
    print("Merging labs to cohort...")

    cohort_with_labs = pd.merge(
        cohort_final,
        labs_pivot,
        on='hospitalization_id',
        how='left'
    )

    print(f"✓ Labs merged to cohort: {len(cohort_with_labs):,} hospitalizations")

    # Reorder columns with labs included
    final_column_order_with_labs = [
        'patient_id',
        'hospitalization_id',
        'age_at_admission',
        'admission_dttm', 'discharge_dttm','discharge_category',
        'start_dttm',
        'end_dttm',
        'hospital_los_days',
        'icu_los_days',
        'inpatient_mortality',
        'icu_mortality',
        'highest_temperature',
        'lowest_temperature',
        'lowest_map',
        'highest_wbc',
        'highest_creatinine',
        'vasopressor_ever',
        'no_of_vasopressor',
        'location_type',
        'sex_category',
        'ethnicity_category',
        'race_category',
        'race_ethnicity',
        'language_category'
    ]

    # Only select columns that exist
    cohort_complete = cohort_with_labs[[col for col in final_column_order_with_labs if col in cohort_with_labs.columns]]

    # Return complete cohort with all features
    return (cohort_complete,)


@app.cell
def _(mo):
    mo.md(r"""## Load and Process Respiratory Support (NIPPV, HFNO)""")
    return


@app.cell
def _(RespiratorySupport, cohort_df):
    # Load respiratory support data for NIPPV and HFNO
    print("Loading respiratory support data using clifpy...")

    # Extract cohort hospitalization IDs
    cohort_hosp_ids_resp = cohort_df['hospitalization_id'].astype(str).unique().tolist()
    print(f"Loading respiratory support for {len(cohort_hosp_ids_resp):,} hospitalizations")
    print("Loading all device categories (no category filter)")

    # Load respiratory support table with hospitalization_id filter only (NO category filter)
    resp_table = RespiratorySupport.from_file(
        config_path='clif_config.json',
        filters={
            'hospitalization_id': cohort_hosp_ids_resp
        },
        columns=['hospitalization_id', 'recorded_dttm', 'device_category']
    )

    resp_df = resp_table.df.copy()
    print(f"✓ Respiratory support loaded: {len(resp_df):,} records")

    # Show device category distribution
    if 'device_category' in resp_df.columns:
        print(f"\n=== Device Category Distribution ===")
        print(resp_df['device_category'].value_counts())
    return (resp_df,)


@app.cell
def _(cohort_df, pd, resp_df):
    # Filter respiratory support to ICU stay windows and create NIPPV/HFNO flags
    print("Filtering respiratory support to ICU stay windows...")

    # Merge respiratory support with cohort to get ICU stay windows
    resp_with_windows = pd.merge(
        resp_df,
        cohort_df[['hospitalization_id', 'start_dttm', 'end_dttm']],
        on='hospitalization_id',
        how='inner'
    )

    # Convert datetime column
    resp_with_windows['recorded_dttm'] = pd.to_datetime(resp_with_windows['recorded_dttm'])

    # Filter respiratory support to ICU stay window (start_dttm <= recorded_dttm <= end_dttm)
    resp_icu_window = resp_with_windows[
        (resp_with_windows['recorded_dttm'] >= resp_with_windows['start_dttm']) &
        (resp_with_windows['recorded_dttm'] <= resp_with_windows['end_dttm'])
    ].copy()

    print(f"✓ Respiratory support filtered to ICU windows: {len(resp_icu_window):,} records")

    # Create NIPPV_ever and HFNO_ever flags
    print("Creating NIPPV_ever and HFNO_ever flags...")

    # Normalize device_category to lowercase for case-insensitive matching
    resp_icu_window['device_category_lower'] = resp_icu_window['device_category'].str.lower()

    # Group by hospitalization_id and check if devices were used
    resp_summary = resp_icu_window.groupby('hospitalization_id').agg(
        NIPPV_ever=('device_category_lower', lambda x: 1 if any('nippv' in str(d) for d in x) else 0),
        HFNO_ever=('device_category_lower', lambda x: 1 if any('high flow nc' in str(d) for d in x) else 0),
        IMV_ever=('device_category_lower', lambda x: 1 if any('imv' in str(d) for d in x) else 0)
    ).reset_index()

    print(f"✓ Respiratory support metrics calculated for {len(resp_summary):,} hospitalizations")
    print(f"\n=== Respiratory Support Usage ===")
    print(f"NIPPV usage:")
    print(f"  Hospitalizations with NIPPV: {(resp_summary['NIPPV_ever'] == 1).sum():,} ({(resp_summary['NIPPV_ever'] == 1).mean()*100:.1f}%)")
    print(f"  Hospitalizations without NIPPV: {(resp_summary['NIPPV_ever'] == 0).sum():,} ({(resp_summary['NIPPV_ever'] == 0).mean()*100:.1f}%)")

    print(f"\nHFNO usage:")
    print(f"  Hospitalizations with HFNO: {(resp_summary['HFNO_ever'] == 1).sum():,} ({(resp_summary['HFNO_ever'] == 1).mean()*100:.1f}%)")
    print(f"  Hospitalizations without HFNO: {(resp_summary['HFNO_ever'] == 0).sum():,} ({(resp_summary['HFNO_ever'] == 0).mean()*100:.1f}%)")

    print(f"\nIMV usage:")
    print(f"  Hospitalizations with IMV: {(resp_summary['IMV_ever'] == 1).sum():,} ({(resp_summary['IMV_ever'] == 1).mean()*100:.1f}%)")
    print(f"  Hospitalizations without IMV: {(resp_summary['IMV_ever'] == 0).sum():,} ({(resp_summary['IMV_ever'] == 0).mean()*100:.1f}%)")
    return (resp_summary,)


@app.cell
def _(cohort_complete, pd, resp_summary):
    # Merge respiratory support metrics to cohort
    print("Merging respiratory support metrics to cohort...")

    cohort_with_resp = pd.merge(
        cohort_complete,
        resp_summary,
        on='hospitalization_id',
        how='left'
    )

    # Fill NaN (no NIPPV/HFNO/IMV) with 0
    cohort_with_resp['NIPPV_ever'] = cohort_with_resp['NIPPV_ever'].fillna(0).astype(int)
    cohort_with_resp['HFNO_ever'] = cohort_with_resp['HFNO_ever'].fillna(0).astype(int)
    cohort_with_resp['IMV_ever'] = cohort_with_resp['IMV_ever'].fillna(0).astype(int)

    print(f"✓ Respiratory support metrics merged to cohort: {len(cohort_with_resp):,} hospitalizations")
    print(f"  Hospitalizations with NIPPV: {(cohort_with_resp['NIPPV_ever'] == 1).sum():,} ({(cohort_with_resp['NIPPV_ever'] == 1).mean()*100:.1f}%)")
    print(f"  Hospitalizations without NIPPV: {(cohort_with_resp['NIPPV_ever'] == 0).sum():,} ({(cohort_with_resp['NIPPV_ever'] == 0).mean()*100:.1f}%)")
    print(f"  Hospitalizations with HFNO: {(cohort_with_resp['HFNO_ever'] == 1).sum():,} ({(cohort_with_resp['HFNO_ever'] == 1).mean()*100:.1f}%)")
    print(f"  Hospitalizations without HFNO: {(cohort_with_resp['HFNO_ever'] == 0).sum():,} ({(cohort_with_resp['HFNO_ever'] == 0).mean()*100:.1f}%)")
    print(f"  Hospitalizations with IMV: {(cohort_with_resp['IMV_ever'] == 1).sum():,} ({(cohort_with_resp['IMV_ever'] == 1).mean()*100:.1f}%)")
    print(f"  Hospitalizations without IMV: {(cohort_with_resp['IMV_ever'] == 0).sum():,} ({(cohort_with_resp['IMV_ever'] == 0).mean()*100:.1f}%)")

    # Reorder columns with respiratory support columns included
    final_column_order_with_resp = [
        'patient_id',
        'hospitalization_id',
        'age_at_admission',
        'admission_dttm', 'discharge_dttm','discharge_category',
        'start_dttm',
        'end_dttm',
        'hospital_los_days',
        'icu_los_days',
        'inpatient_mortality',
        'icu_mortality',
        'highest_temperature',
        'lowest_temperature',
        'lowest_map',
        'highest_wbc',
        'highest_creatinine',
        'NIPPV_ever',
        'HFNO_ever',
        'IMV_ever',
        'vasopressor_ever',
        'no_of_vasopressor',
        'location_type',
        'sex_category',
        'ethnicity_category',
        'race_category',
        'race_ethnicity',
        'language_category'
    ]

    # Only select columns that exist
    cohort_final_with_resp = cohort_with_resp[[col for col in final_column_order_with_resp if col in cohort_with_resp.columns]]

    # Return complete cohort with all features including respiratory support
    return (cohort_final_with_resp,)


@app.cell
def _(mo):
    mo.md(r"""## Compute SOFA Scores""")
    return


@app.cell
def _(ClifOrchestrator):
    # Initialize NEW ClifOrchestrator for SOFA computation
    print("\n=== SOFA Score Computation ===")
    print("Initializing ClifOrchestrator for SOFA...")
    co_sofa = ClifOrchestrator(config_path='clif_config.json')
    print("✓ ClifOrchestrator initialized for SOFA")
    return (co_sofa,)


@app.cell
def _(cohort_final_with_resp, pd):
    # Prepare cohort for SOFA computation (first 24 hours of ICU stay only)
    print("Preparing cohort for SOFA score computation (first 24 hours)...")

    # Calculate end_time as start + 24 hours, but cap at actual ICU discharge
    sofa_end_time = pd.Series([
        min(start + pd.Timedelta(hours=24), end)
        for start, end in zip(
            cohort_final_with_resp['start_dttm'],
            cohort_final_with_resp['end_dttm']
        )
    ])

    sofa_cohort_df = pd.DataFrame({
        'hospitalization_id': cohort_final_with_resp['hospitalization_id'],
        'start_time': cohort_final_with_resp['start_dttm'],
        'end_time': sofa_end_time
    })

    print(f"✓ SOFA cohort prepared: {len(sofa_cohort_df):,} hospitalizations (first 24h window)")
    return (sofa_cohort_df,)


@app.cell
def _(cohort_final_with_resp):
    # Extract hospitalization IDs for SOFA table filtering
    print("Extracting hospitalization IDs for SOFA data filtering...")

    sofa_cohort_ids = cohort_final_with_resp['hospitalization_id'].astype(str).unique().tolist()

    print(f"✓ Extracted {len(sofa_cohort_ids):,} hospitalization IDs")
    return (sofa_cohort_ids,)


@app.cell
def _(co_sofa, sofa_cohort_ids):
    # Load required tables for SOFA computation
    print("Loading tables for SOFA computation...")
    print("  Tables: Labs, Vitals, PatientAssessments, MedicationAdminContinuous, RespiratorySupport")

    # Define columns AND category filters for each table (memory optimization)
    sofa_config = {
        'labs': {
            'columns': ['hospitalization_id', 'lab_result_dttm', 'lab_category', 'lab_value', 'lab_value_numeric'],
            'categories': ['creatinine', 'platelet_count', 'po2_arterial', 'bilirubin_total']
        },
        'vitals': {
            'columns': ['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value'],
            'categories': ['map', 'spo2', 'weight_kg', 'height_cm']
        },
        'patient_assessments': {
            'columns': ['hospitalization_id', 'recorded_dttm', 'assessment_category', 'numerical_value','categorical_value'],
            'categories': ['gcs_total']
        },
        'medication_admin_continuous': {
            'columns': None,  # Load all columns
            'categories': ['norepinephrine', 'epinephrine', 'dopamine', 'dobutamine']
        },
        'respiratory_support': {
            'columns': None,  # Load all columns
            'categories': None  # Load all device categories (need device_category + fio2_set)
        }
    }

    for table_name, config in sofa_config.items():
        table_cols = config['columns']
        table_cats = config['categories']

        # Build filters dictionary
        filters = {'hospitalization_id': sofa_cohort_ids}

        # Add category filter if specified
        if table_cats is not None:
            category_col = {
                'labs': 'lab_category',
                'vitals': 'vital_category',
                'patient_assessments': 'assessment_category',
                'medication_admin_continuous': 'med_category',
                'respiratory_support': 'device_category'
            }[table_name]
            filters[category_col] = table_cats
            print(f"  Loading {table_name} ({len(table_cats)} categories)...")
        else:
            print(f"  Loading {table_name} (all categories)...")

        co_sofa.load_table(
            table_name,
            filters=filters,
            columns=table_cols
        )

    print("✓ All SOFA tables loaded with category filters")
    return


@app.cell
def _(co_sofa):
    # Clean medication data (remove null/NaN med_dose and med_dose_unit)
    print("Cleaning medication data...")

    med_df = co_sofa.medication_admin_continuous.df.copy()
    initial_count = len(med_df)

    # Remove null med_dose
    med_df = med_df[med_df['med_dose'].notna()]
    # Remove null med_dose_unit
    med_df = med_df[med_df['med_dose_unit'].notna()]
    # Remove 'nan' string values
    med_df = med_df[~med_df['med_dose_unit'].astype(str).str.lower().isin(['nan', 'none', ''])]

    final_count = len(med_df)
    co_sofa.medication_admin_continuous.df = med_df

    print(f"✓ Medication data cleaned: {initial_count:,} → {final_count:,} records ({initial_count - final_count:,} removed)")
    return


@app.cell
def _(co_sofa):
    # Convert medication units for SOFA
    print("Converting medication units to mcg/kg/min...")

    preferred_units = {
        'norepinephrine': 'mcg/kg/min',
        'epinephrine': 'mcg/kg/min',
        'dopamine': 'mcg/kg/min',
        'dobutamine': 'mcg/kg/min'
    }

    co_sofa.convert_dose_units_for_continuous_meds(
        preferred_units=preferred_units,
        override=True,
        save_to_table=True
    )

    # Check conversion results
    conversion_counts = co_sofa.medication_admin_continuous.conversion_counts
    success_count = conversion_counts[conversion_counts['_convert_status'] == 'success']['count'].sum()
    total_count = conversion_counts['count'].sum()

    print(f"✓ Unit conversion complete: {success_count:,} / {total_count:,} successful ({100*success_count/total_count:.1f}%)")
    return


@app.cell
def _(co_sofa, sofa_cohort_df):
    # Compute SOFA scores
    print("Computing SOFA scores...")

    sofa_scores = co_sofa.compute_sofa_scores(
        cohort_df=sofa_cohort_df,
        id_name='hospitalization_id'
    )

    # Show SOFA columns (inline to avoid variable conflicts)
    print(f"✓ SOFA scores computed: {sofa_scores.shape}")
    print(f"  SOFA columns: {[col for col in sofa_scores.columns if 'sofa' in col.lower()]}")
    return (sofa_scores,)


@app.cell
def _(co_sofa, pd):
    # Extract height and weight from vitals and calculate BMI
    print("\n=== BMI Calculation ===")
    print("Extracting height and weight from vitals...")

    # Get vitals dataframe from co_sofa
    vitals_for_bmi = co_sofa.vitals.df.copy()

    # Filter for height_cm and weight_kg only
    bmi_vitals = vitals_for_bmi[
        vitals_for_bmi['vital_category'].isin(['height_cm', 'weight_kg'])
    ].copy()

    print(f"  Vitals for BMI: {len(bmi_vitals):,} records")

    # Get first recorded height and weight for each hospitalization
    # Sort by recorded_dttm to get earliest values
    bmi_vitals = bmi_vitals.sort_values('recorded_dttm')

    # Pivot to get height_cm and weight_kg as separate columns (take first value)
    bmi_pivot = bmi_vitals.groupby(['hospitalization_id', 'vital_category'])['vital_value'].first().unstack()

    # Calculate BMI: weight_kg / (height_cm/100)^2
    # If either height or weight is missing, BMI will be null
    bmi_df = pd.DataFrame({
        'hospitalization_id': bmi_pivot.index,
        'height_cm': bmi_pivot.get('height_cm', pd.Series(dtype=float)),
        'weight_kg': bmi_pivot.get('weight_kg', pd.Series(dtype=float))
    }).reset_index(drop=True)

    # Calculate BMI (null if either height or weight is missing)
    bmi_df['bmi'] = bmi_df.apply(
        lambda row: row['weight_kg'] / ((row['height_cm'] / 100) ** 2)
        if pd.notna(row['height_cm']) and pd.notna(row['weight_kg']) and row['height_cm'] > 0
        else None,
        axis=1
    )

    # Keep only hospitalization_id and bmi for merging
    bmi_final = bmi_df[['hospitalization_id', 'bmi']].copy()

    print(f"✓ BMI calculated for {bmi_final['bmi'].notna().sum():,} hospitalizations")
    print(f"  Missing BMI: {bmi_final['bmi'].isna().sum():,}")
    if bmi_final['bmi'].notna().any():
        print(f"  Mean BMI: {bmi_final['bmi'].mean():.2f}")
        print(f"  Median BMI: {bmi_final['bmi'].median():.2f}")

    return (bmi_final,)


@app.cell
def _(bmi_final, cohort_final_with_resp, pd, sofa_scores):
    # Merge SOFA scores and BMI with cohort
    print("Merging SOFA scores with cohort...")

    cohort_with_sofa_temp = pd.merge(
        cohort_final_with_resp,
        sofa_scores,
        on='hospitalization_id',
        how='left'
    )

    # Print SOFA merge summary (inline to avoid variable conflicts)
    print(f"✓ SOFA scores merged: {cohort_with_sofa_temp.shape}")
    print(f"  Total columns: {len(cohort_with_sofa_temp.columns)}")
    print(f"  SOFA columns added: {len([col for col in cohort_with_sofa_temp.columns if 'sofa' in col.lower()])}")

    # Merge BMI
    print("Merging BMI with cohort...")
    cohort_with_sofa = pd.merge(
        cohort_with_sofa_temp,
        bmi_final,
        on='hospitalization_id',
        how='left'
    )

    print(f"✓ BMI merged to cohort: {len(cohort_with_sofa):,} hospitalizations")
    print(f"  BMI available: {cohort_with_sofa['bmi'].notna().sum():,} ({cohort_with_sofa['bmi'].notna().mean()*100:.1f}%)")
    print(f"  BMI missing: {cohort_with_sofa['bmi'].isna().sum():,} ({cohort_with_sofa['bmi'].isna().mean()*100:.1f}%)")

    return (cohort_with_sofa,)


@app.cell
def _(mo):
    mo.md(r"""## Cohort Summary""")
    return


@app.cell
def _(cohort_with_sofa):
    # Display cohort summary
    print("=== ICU Cohort Summary ===")
    print(f"Total hospitalizations: {len(cohort_with_sofa):,}")
    print(f"Unique patients: {cohort_with_sofa['patient_id'].nunique():,}")
    print(f"\nDate range:")
    print(f"  Start: {cohort_with_sofa['start_dttm'].min()}")
    print(f"  End: {cohort_with_sofa['start_dttm'].max()}")

    print(f"\n=== Hospital Length of Stay (Days) ===")
    print(f"Mean: {cohort_with_sofa['hospital_los_days'].mean():.2f}")
    print(f"Median: {cohort_with_sofa['hospital_los_days'].median():.2f}")
    print(f"Std: {cohort_with_sofa['hospital_los_days'].std():.2f}")
    print(f"Min: {cohort_with_sofa['hospital_los_days'].min():.2f}")
    print(f"Max: {cohort_with_sofa['hospital_los_days'].max():.2f}")
    print(f"25th percentile: {cohort_with_sofa['hospital_los_days'].quantile(0.25):.2f}")
    print(f"75th percentile: {cohort_with_sofa['hospital_los_days'].quantile(0.75):.2f}")

    print(f"\n=== ICU Length of Stay (Days) ===")
    print(f"Mean: {cohort_with_sofa['icu_los_days'].mean():.2f}")
    print(f"Median: {cohort_with_sofa['icu_los_days'].median():.2f}")
    print(f"Std: {cohort_with_sofa['icu_los_days'].std():.2f}")
    print(f"Min: {cohort_with_sofa['icu_los_days'].min():.2f}")
    print(f"Max: {cohort_with_sofa['icu_los_days'].max():.2f}")
    print(f"25th percentile: {cohort_with_sofa['icu_los_days'].quantile(0.25):.2f}")
    print(f"75th percentile: {cohort_with_sofa['icu_los_days'].quantile(0.75):.2f}")

    print(f"\n=== Inpatient Mortality ===")
    print(f"Deaths: {cohort_with_sofa['inpatient_mortality'].sum():,} ({cohort_with_sofa['inpatient_mortality'].mean()*100:.2f}%)")
    print(f"Survived: {(cohort_with_sofa['inpatient_mortality'] == 0).sum():,}")

    print(f"\n=== ICU Mortality ===")
    print(f"Deaths during ICU stay: {cohort_with_sofa['icu_mortality'].sum():,} ({cohort_with_sofa['icu_mortality'].mean()*100:.2f}%)")
    print(f"Survived ICU stay: {(cohort_with_sofa['icu_mortality'] == 0).sum():,}")

    print(f"\n=== BMI (Body Mass Index) ===")
    if 'bmi' in cohort_with_sofa.columns:
        print(f"Mean BMI: {cohort_with_sofa['bmi'].mean():.2f}")
        print(f"Median BMI: {cohort_with_sofa['bmi'].median():.2f}")
        print(f"Std: {cohort_with_sofa['bmi'].std():.2f}")
        print(f"Min: {cohort_with_sofa['bmi'].min():.2f}")
        print(f"Max: {cohort_with_sofa['bmi'].max():.2f}")
        print(f"25th percentile: {cohort_with_sofa['bmi'].quantile(0.25):.2f}")
        print(f"75th percentile: {cohort_with_sofa['bmi'].quantile(0.75):.2f}")
        print(f"Missing: {cohort_with_sofa['bmi'].isna().sum():,} ({cohort_with_sofa['bmi'].isna().mean()*100:.1f}%)")

    print(f"\n=== Vital Signs (ICU Stay Window) ===")
    if 'highest_temperature' in cohort_with_sofa.columns:
        print(f"Highest Temperature (°C):")
        print(f"  Mean: {cohort_with_sofa['highest_temperature'].mean():.2f}")
        print(f"  Median: {cohort_with_sofa['highest_temperature'].median():.2f}")
        print(f"  Missing: {cohort_with_sofa['highest_temperature'].isna().sum():,} ({cohort_with_sofa['highest_temperature'].isna().mean()*100:.1f}%)")

    if 'lowest_temperature' in cohort_with_sofa.columns:
        print(f"Lowest Temperature (°C):")
        print(f"  Mean: {cohort_with_sofa['lowest_temperature'].mean():.2f}")
        print(f"  Median: {cohort_with_sofa['lowest_temperature'].median():.2f}")
        print(f"  Missing: {cohort_with_sofa['lowest_temperature'].isna().sum():,} ({cohort_with_sofa['lowest_temperature'].isna().mean()*100:.1f}%)")

    if 'lowest_map' in cohort_with_sofa.columns:
        print(f"Lowest MAP (mmHg):")
        print(f"  Mean: {cohort_with_sofa['lowest_map'].mean():.2f}")
        print(f"  Median: {cohort_with_sofa['lowest_map'].median():.2f}")
        print(f"  Missing: {cohort_with_sofa['lowest_map'].isna().sum():,} ({cohort_with_sofa['lowest_map'].isna().mean()*100:.1f}%)")

    print(f"\n=== Laboratory Values (ICU Stay Window) ===")
    if 'highest_wbc' in cohort_with_sofa.columns:
        print(f"Highest WBC (10^3/μL):")
        print(f"  Mean: {cohort_with_sofa['highest_wbc'].mean():.2f}")
        print(f"  Median: {cohort_with_sofa['highest_wbc'].median():.2f}")
        print(f"  Missing: {cohort_with_sofa['highest_wbc'].isna().sum():,} ({cohort_with_sofa['highest_wbc'].isna().mean()*100:.1f}%)")

    if 'highest_creatinine' in cohort_with_sofa.columns:
        print(f"Highest Creatinine (mg/dL):")
        print(f"  Mean: {cohort_with_sofa['highest_creatinine'].mean():.2f}")
        print(f"  Median: {cohort_with_sofa['highest_creatinine'].median():.2f}")
        print(f"  Missing: {cohort_with_sofa['highest_creatinine'].isna().sum():,} ({cohort_with_sofa['highest_creatinine'].isna().mean()*100:.1f}%)")

    print(f"\n=== Respiratory Support Usage (ICU Stay Window) ===")
    if 'NIPPV_ever' in cohort_with_sofa.columns:
        print(f"Hospitalizations with NIPPV: {(cohort_with_sofa['NIPPV_ever'] == 1).sum():,} ({(cohort_with_sofa['NIPPV_ever'] == 1).mean()*100:.1f}%)")
        print(f"Hospitalizations without NIPPV: {(cohort_with_sofa['NIPPV_ever'] == 0).sum():,} ({(cohort_with_sofa['NIPPV_ever'] == 0).mean()*100:.1f}%)")

    if 'HFNO_ever' in cohort_with_sofa.columns:
        print(f"\nHospitalizations with HFNO: {(cohort_with_sofa['HFNO_ever'] == 1).sum():,} ({(cohort_with_sofa['HFNO_ever'] == 1).mean()*100:.1f}%)")
        print(f"Hospitalizations without HFNO: {(cohort_with_sofa['HFNO_ever'] == 0).sum():,} ({(cohort_with_sofa['HFNO_ever'] == 0).mean()*100:.1f}%)")

    if 'IMV_ever' in cohort_with_sofa.columns:
        print(f"\nHospitalizations with IMV: {(cohort_with_sofa['IMV_ever'] == 1).sum():,} ({(cohort_with_sofa['IMV_ever'] == 1).mean()*100:.1f}%)")
        print(f"Hospitalizations without IMV: {(cohort_with_sofa['IMV_ever'] == 0).sum():,} ({(cohort_with_sofa['IMV_ever'] == 0).mean()*100:.1f}%)")

    print(f"\n=== Vasopressor Usage (ICU Stay Window) ===")
    if 'vasopressor_ever' in cohort_with_sofa.columns:
        print(f"Hospitalizations with vasopressors: {(cohort_with_sofa['vasopressor_ever'] == 1).sum():,} ({(cohort_with_sofa['vasopressor_ever'] == 1).mean()*100:.1f}%)")
        print(f"Hospitalizations without vasopressors: {(cohort_with_sofa['vasopressor_ever'] == 0).sum():,} ({(cohort_with_sofa['vasopressor_ever'] == 0).mean()*100:.1f}%)")

    if 'no_of_vasopressor' in cohort_with_sofa.columns:
        print(f"\nNumber of Vasopressor Categories:")
        print(f"  Mean: {cohort_with_sofa['no_of_vasopressor'].mean():.2f}")
        print(f"  Median: {cohort_with_sofa['no_of_vasopressor'].median():.2f}")
        print(f"  Distribution: {cohort_with_sofa['no_of_vasopressor'].value_counts().sort_index().to_dict()}")

    print(f"\n=== ICU Location Types ===")
    print(cohort_with_sofa['location_type'].value_counts())

    print(f"\n=== SOFA Score Summary ===")
    if 'sofa_total' in cohort_with_sofa.columns:
        print(f"Mean SOFA: {cohort_with_sofa['sofa_total'].mean():.2f}")
        print(f"Median SOFA: {cohort_with_sofa['sofa_total'].median():.2f}")
        print(f"Missing SOFA: {cohort_with_sofa['sofa_total'].isna().sum():,} ({cohort_with_sofa['sofa_total'].isna().mean()*100:.1f}%)")
    return


@app.cell
def _():
    # Display cohort table
    #cohort_final_with_resp
    return


@app.cell
def _(mo):
    mo.md(r"""## Save Cohort to PHI_DATA""")
    return


@app.cell
def _(cohort_with_sofa):
    import os
    from pathlib import Path

    # Create PHI_DATA directory using Python
    phi_data_dir = Path('PHI_DATA')
    phi_data_dir.mkdir(exist_ok=True)

    # Save final cohort to parquet (all columns including SOFA)
    output_path = phi_data_dir / 'cohort_icu_first_stay.parquet'
    cohort_with_sofa.to_parquet(output_path, index=False)

    print(f"\n=== Cohort Saved ===")
    print(f"Location: {output_path}")
    print(f"Rows: {len(cohort_with_sofa):,}")
    print(f"Columns: {len(cohort_with_sofa.columns)}")
    print(f"File size: {output_path.stat().st_size / (1024**2):.2f} MB")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
