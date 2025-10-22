import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Days of Therapy (DOT) Analysis

    This notebook calculates **Days of Therapy (DOT)** metrics for antimicrobial consumption analysis.

    ## Objective
    Calculate DOT metrics using:
    - ICU cohort from `01_cohort.py`
    - Intermittent antibiotic administrations
    - Antibiotic spectrum scoring reference

    ## DOT Metrics Calculated
    **Patient-level:**
    - Total DOT (unique calendar days with any antibiotic)
    - Total antibiotic doses administered
    - Number of unique antibiotics used
    - First/last antibiotic day
    - Therapy duration (calendar days)
    - Spectrum-weighted DOT
    - Mean/max spectrum score

    **Antibiotic-level:**
    - Total DOT by antibiotic category
    - Number of patients receiving each antibiotic
    - Total doses by antibiotic
    - Mean spectrum score by antibiotic

    **Cohort-level:**
    - DOT per 1000 patient-days
    - Mean/median DOT per admission
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Setup and Configuration""")
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from clifpy.tables import MedicationAdminIntermittent, HospitalDiagnosis
    from clifpy.utils.comorbidity import calculate_elix, calculate_cci
    from pathlib import Path
    import warnings
    import json
    warnings.filterwarnings('ignore')

    print("=== Days of Therapy (DOT) Analysis ===")
    print("Setting up environment...")

    # Load site name from config
    with open('clif_config.json', 'r') as f:
        config = json.load(f)
    site_name = config.get('site', 'UNKNOWN_SITE')
    print(f"Site: {site_name}")
    return (
        HospitalDiagnosis,
        MedicationAdminIntermittent,
        Path,
        calculate_cci,
        calculate_elix,
        json,
        np,
        pd,
        site_name,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Cohort""")
    return


@app.cell
def _(Path, pd):
    # Load cohort from 01_cohort.py output
    print("Loading ICU cohort...")

    cohort_path = Path('PHI_DATA/cohort_icu_first_stay.parquet')
    cohort_df = pd.read_parquet(cohort_path)

    # Strip timezone info from datetime columns (keep local time, remove timezone awareness)
    cohort_df['start_dttm'] = cohort_df['start_dttm'].dt.tz_localize(None)
    cohort_df['end_dttm'] = cohort_df['end_dttm'].dt.tz_localize(None)

    print(f"✓ Cohort loaded: {len(cohort_df):,} hospitalizations")
    print(f"  Unique patients: {cohort_df['patient_id'].nunique():,}")
    print(f"  Date range: {cohort_df['start_dttm'].min()} to {cohort_df['start_dttm'].max()}")
    print(f"  Columns: {len(cohort_df.columns)}")

    # Extract key columns for filtering
    cohort_ids = cohort_df['hospitalization_id'].astype(str).unique().tolist()
    print(f"  Cohort IDs extracted: {len(cohort_ids):,}")
    return cohort_df, cohort_ids


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Calculate Comorbidity Scores (Elixhauser & CCI)""")
    return


@app.cell
def _(HospitalDiagnosis, cohort_df):
    # Load hospital diagnosis data for comorbidity calculation
    print("\n=== Comorbidity Score Calculation ===")
    print("Loading hospital diagnosis data...")

    cohort_hosp_ids_dx = cohort_df['hospitalization_id'].astype(str).unique().tolist()

    hosp_dx_table = HospitalDiagnosis.from_file(
        config_path='clif_config.json',
        filters={
            'hospitalization_id': cohort_hosp_ids_dx
        },
        columns=['hospitalization_id', 'diagnosis_code', 'diagnosis_code_format']
    )

    print(f"✓ Hospital diagnosis data loaded: {len(hosp_dx_table.df):,} records")
    print(f"  Unique hospitalizations: {hosp_dx_table.df['hospitalization_id'].nunique():,}")
    print(f"  ICD10CM codes: {(hosp_dx_table.df['diagnosis_code_format'] == 'ICD10CM').sum():,}")
    return (hosp_dx_table,)


@app.cell
def _(calculate_elix, hosp_dx_table):
    # Calculate Elixhauser Comorbidity Index
    print("\nCalculating Elixhauser Comorbidity Index...")

    elix_results = calculate_elix(hosp_dx_table, hierarchy=True)

    print(f"\n✓ Elixhauser scores calculated: {len(elix_results):,} hospitalizations")
    print(f"  Mean Elixhauser score: {elix_results['elix_score'].mean():.2f}")
    print(f"  Median Elixhauser score: {elix_results['elix_score'].median():.2f}")
    print(f"  Range: {elix_results['elix_score'].min():.0f} to {elix_results['elix_score'].max():.0f}")
    return (elix_results,)


@app.cell
def _(calculate_cci, hosp_dx_table):
    # Calculate Charlson Comorbidity Index
    print("\nCalculating Charlson Comorbidity Index (CCI)...")

    cci_results = calculate_cci(hosp_dx_table, hierarchy=True)

    print(f"\n✓ CCI scores calculated: {len(cci_results):,} hospitalizations")
    print(f"  Mean CCI score: {cci_results['cci_score'].mean():.2f}")
    print(f"  Median CCI score: {cci_results['cci_score'].median():.2f}")
    print(f"  Range: {cci_results['cci_score'].min():.0f} to {cci_results['cci_score'].max():.0f}")
    return (cci_results,)


@app.cell
def _(cci_results, cohort_df, elix_results, pd):
    # Merge comorbidity scores to cohort
    print("\nMerging comorbidity scores to cohort...")

    # Merge Elixhauser scores (select score and chronic_pulmonary column)
    elix_cols_to_merge = ['hospitalization_id', 'elix_score']
    # Check if chronic_pulmonary column exists (may be named differently)
    elix_chronic_col = None
    for col in elix_results.columns:
        if 'chronic' in col.lower() and 'pulmonary' in col.lower():
            elix_chronic_col = col
            elix_cols_to_merge.append(col)
            break

    cohort_with_elix = pd.merge(
        cohort_df,
        elix_results[elix_cols_to_merge],
        on='hospitalization_id',
        how='left'
    )

    # Merge CCI scores (select score and chronic_pulmonary_disease column)
    cci_cols_to_merge = ['hospitalization_id', 'cci_score']
    # Check if chronic_pulmonary_disease column exists
    cci_chronic_col = None
    for col in cci_results.columns:
        if 'chronic' in col.lower() and 'pulmonary' in col.lower():
            cci_chronic_col = col
            cci_cols_to_merge.append(col)
            break

    cohort_with_comorbidity = pd.merge(
        cohort_with_elix,
        cci_results[cci_cols_to_merge],
        on='hospitalization_id',
        how='left',
        suffixes=('', '_cci')
    )

    # Create unified chronic_pulmonary_disease flag (1 if present in either index)
    if elix_chronic_col or cci_chronic_col:
        chronic_cols = []
        if elix_chronic_col:
            chronic_cols.append(cohort_with_comorbidity[elix_chronic_col])
        if cci_chronic_col:
            chronic_cols.append(cohort_with_comorbidity[cci_chronic_col])

        # Take max across available columns (1 if present in any)
        cohort_with_comorbidity['chronic_pulmonary_disease'] = pd.concat(chronic_cols, axis=1).max(axis=1).fillna(0).astype(int)
    else:
        # If no chronic pulmonary column found, set to 0
        cohort_with_comorbidity['chronic_pulmonary_disease'] = 0
        print("  Warning: No chronic pulmonary disease column found in comorbidity results")

    print(f"✓ Comorbidity scores merged: {len(cohort_with_comorbidity):,} hospitalizations")
    print(f"  Elixhauser available: {cohort_with_comorbidity['elix_score'].notna().sum():,}")
    print(f"  CCI available: {cohort_with_comorbidity['cci_score'].notna().sum():,}")
    print(f"  Chronic pulmonary disease: {(cohort_with_comorbidity['chronic_pulmonary_disease'] == 1).sum():,} ({(cohort_with_comorbidity['chronic_pulmonary_disease'] == 1).mean()*100:.1f}%)")
    return (cohort_with_comorbidity,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Antibiotic Spectrum Scoring""")
    return


@app.cell
def _(pd):
    # Load antibiotic spectrum scoring (pre-cleaned CSV)
    print("Loading antibiotic spectrum scoring...")

    abx_spectrum = pd.read_csv('antibiotic_spectrum_scoring.csv')

    # Clean column names (strip whitespace)
    abx_spectrum.columns = abx_spectrum.columns.str.strip()

    # Remove any rows with null values
    abx_spectrum = abx_spectrum.dropna()

    # Rename columns for consistency
    abx_spectrum = abx_spectrum.rename(columns={
        'Antibiotic': 'med_category',
        'Score': 'spectrum_score'
    })

    print(f"✓ Antibiotic spectrum scoring loaded: {len(abx_spectrum)} antibiotics")
    print(f"  Spectrum score range: {abx_spectrum['spectrum_score'].min()} - {abx_spectrum['spectrum_score'].max()}")

    # Extract antibiotic list for filtering medications
    antibiotic_filter_list = abx_spectrum['med_category'].str.strip().tolist()

    print(f"  Antibiotic categories for filtering: {len(antibiotic_filter_list)}")

    # Show sample
    print("\n=== Sample Antibiotic Spectrum Scores ===")
    print(abx_spectrum.sort_values('spectrum_score', ascending=False).head(10).to_string(index=False))
    return abx_spectrum, antibiotic_filter_list


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Medication Administration Intermittent""")
    return


@app.cell
def _(MedicationAdminIntermittent, antibiotic_filter_list, cohort_ids):
    # Load medication_admin_intermittent table with category filters
    print("Loading medication_admin_intermittent table...")
    print(f"  Filters:")
    print(f"    - Hospitalization IDs: {len(cohort_ids):,}")
    print(f"    - Antibiotic categories: {len(antibiotic_filter_list)}")

    meds_intermittent_table = MedicationAdminIntermittent.from_file(
        config_path='clif_config.json',
        filters={
            'hospitalization_id': cohort_ids,
            'med_category': antibiotic_filter_list
        },
        columns=[
            'hospitalization_id',
            'admin_dttm',
            'med_category',
            'med_route_category',
            'med_dose',
            'med_dose_unit'
        ]
    )

    meds_intermittent_df = meds_intermittent_table.df.copy()

    # Strip timezone info from datetime column (keep local time, remove timezone awareness)
    meds_intermittent_df['admin_dttm'] = meds_intermittent_df['admin_dttm'].dt.tz_localize(None)

    print(f"✓ Medication administrations loaded: {len(meds_intermittent_df):,} records")
    print(f"  Unique patients: {meds_intermittent_df['hospitalization_id'].nunique():,}")
    print(f"  Unique antibiotics: {meds_intermittent_df['med_category'].nunique()}")
    print(f"  Date range: {meds_intermittent_df['admin_dttm'].min()} to {meds_intermittent_df['admin_dttm'].max()}")
    return (meds_intermittent_df,)


@app.cell
def _(antibiotic_filter_list, meds_intermittent_df):
    # Compare antibiotics in CSV vs medication data
    print("\n=== Antibiotic Reference Table Coverage ===")

    # Get unique antibiotics found in medication data
    antibiotics_found_in_meds = set(meds_intermittent_df['med_category'].unique())
    antibiotics_in_csv = set(antibiotic_filter_list)

    # Find antibiotics in CSV but NOT in medication data
    antibiotics_not_found = antibiotics_in_csv - antibiotics_found_in_meds

    # Statistics
    total_csv_antibiotics = len(antibiotics_in_csv)
    total_found_antibiotics = len(antibiotics_found_in_meds)
    total_not_found = len(antibiotics_not_found)

    print(f"  Total antibiotics in CSV reference table: {total_csv_antibiotics}")
    print(f"  Antibiotics found in medication data: {total_found_antibiotics}")
    print(f"  Antibiotics in CSV but NOT found in data: {total_not_found} ({100*total_not_found/total_csv_antibiotics:.1f}%)")

    if total_not_found > 0:
        print(f"\n=== Antibiotics from CSV Not Found in Medication Data ===")
        for abx_name in sorted(antibiotics_not_found):
            print(f"  - {abx_name}")
    else:
        print("\n✓ All antibiotics in CSV were found in medication data")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Filter to ICU Stay Window""")
    return


@app.cell
def _(cohort_with_comorbidity, meds_intermittent_df, pd):
    # Filter medication administrations to ICU stay windows
    print("Filtering medications to ICU stay windows...")

    # Merge with cohort to get start_dttm and end_dttm
    meds_with_windows = pd.merge(
        meds_intermittent_df,
        cohort_with_comorbidity[['hospitalization_id', 'start_dttm', 'end_dttm']],
        on='hospitalization_id',
        how='inner'
    )

    # Filter to ICU window: start_dttm <= admin_dttm <= end_dttm
    meds_icu_window = meds_with_windows[
        (meds_with_windows['admin_dttm'] >= meds_with_windows['start_dttm']) &
        (meds_with_windows['admin_dttm'] <= meds_with_windows['end_dttm'])
    ].copy()

    print(f"✓ Medications filtered to ICU windows: {len(meds_icu_window):,} records")
    print(f"  Unique patients with antibiotics: {meds_icu_window['hospitalization_id'].nunique():,}")
    print(f"  Records removed (outside ICU window): {len(meds_with_windows) - len(meds_icu_window):,}")
    return (meds_icu_window,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Prepare Data for Analysis""")
    return


@app.cell
def _():
    import polars as pl
    from tqdm import tqdm
    from datetime import timedelta
    import matplotlib.pyplot as plt
    import scipy.interpolate

    print("Loading analysis libraries...")
    return pl, plt, scipy, timedelta, tqdm


@app.cell
def _(cohort_with_comorbidity, meds_icu_window, pl):
    # Prepare data structures
    print("Preparing data for analysis...")

    # Cohort
    cohort_pl = pl.DataFrame(cohort_with_comorbidity)

    # Medications
    meds_pl = pl.DataFrame(meds_icu_window)

    print(f"✓ Data prepared")
    print(f"  Cohort shape: {cohort_pl.shape}")
    print(f"  Medications shape: {meds_pl.shape}")
    return cohort_pl, meds_pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Create 24-Hour Windows""")
    return


@app.cell
def _(cohort_pl, pl, timedelta, tqdm):
    # Create 24-hour windows for each hospitalization
    print("Creating 24-hour windows for each hospitalization...")

    # Example: Patient admitted 2024-01-15 08:00, discharged 2024-01-17 14:00
    #   Window 0: 2024-01-15 08:00 → 2024-01-16 08:00 (24 hours)
    #   Window 1: 2024-01-16 08:00 → 2024-01-17 08:00 (24 hours)
    #   Window 2: 2024-01-17 08:00 → 2024-01-17 14:00 (6 hours, capped at end_dttm)
    #   Total: 3 windows = 3 patient-days

    window_data = []

    for row in tqdm(cohort_pl.iter_rows(named=True), total=len(cohort_pl), desc="Creating windows"):
        _hosp_id = row['hospitalization_id']
        _start = row['start_dttm']
        _end = row['end_dttm']

        # Generate 24-hour windows
        _current = _start
        _win_num = 0

        while _current < _end:
            _win_end = min(_current + timedelta(hours=24), _end)

            window_data.append({
                'hospitalization_id': _hosp_id,
                'window_num': _win_num,
                'window_start': _current,
                'window_end': _win_end,
                'window_hours': (_win_end - _current).total_seconds() / 3600
            })

            _current = _win_end
            _win_num += 1

    # Create Polars DataFrame
    windows_pl = pl.DataFrame(window_data)

    print(f"✓ Created {len(windows_pl):,} 24-hour windows")
    print(f"  Total hospitalizations: {windows_pl['hospitalization_id'].n_unique():,}")
    print(f"  Average windows per hospitalization: {len(windows_pl) / windows_pl['hospitalization_id'].n_unique():.1f}")
    print(f"  Total Patient-Days: {len(windows_pl):,}")
    return (windows_pl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Calculate DOT per Antibiotic

    ### Overview
    For each hospitalization, we calculate **Days of Therapy (DOT)** for each antibiotic by counting how many 24-hour windows contain at least one dose of that antibiotic.

    ### What is a "Day"?
    A "day" is defined as a 24-hour window starting from ICU admission. For example:
    - Window 0 = First 24 hours (ICU Day 0)
    - Window 1 = Second 24 hours (ICU Day 1)
    - And so on...

    If an antibiotic is given at any time during a window, that counts as 1 day of therapy for that antibiotic.

    ### Antibiotic-Free Days
    We also track days where no antibiotics were given. This helps measure antibiotic stewardship.

    ### Output Format
    Patient-level table showing DOT for each antibiotic:

    **Example:**
    ```
    hospitalization_id | antibiotic              | dot
    -------------------|-------------------------|----
    12345             | Vancomycin              | 5
    12345             | Piperacillin-Tazobactam | 3
    12345             | ANTIBIOTIC_FREE         | 2
    67890             | Ceftriaxone             | 1
    ```

    This means patient 12345 received Vancomycin for 5 days, Piperacillin-Tazobactam for 3 days, and had 2 days without any antibiotics during their ICU stay.
    """
    )
    return


@app.cell
def _(abx_spectrum, cohort_with_comorbidity, meds_pl, pl, tqdm, windows_pl):
    # ============================================================
    # INITIALIZE DATA STRUCTURES AND LOOKUP TABLES
    # ============================================================

    print("Calculating DOT and Daily ASC for each antibiotic...")

    # Get unique antibiotics from the spectrum scoring
    all_antibiotics = abx_spectrum['med_category'].str.strip().tolist()

    # Create spectrum score lookup dictionary for ASC calculation
    spectrum_lookup = dict(zip(
        abx_spectrum['med_category'].str.strip(),
        abx_spectrum['spectrum_score']
    ))

    print(f"  Antibiotics to track: {len(all_antibiotics)}")
    print(f"  Spectrum scores loaded: {len(spectrum_lookup)}")

    # Initialize result lists
    dot_results = []
    daily_asc_results = []

    # Group windows by hospitalization for faster processing
    hosp_windows = windows_pl.group_by('hospitalization_id').agg([
        pl.col('window_num'),
        pl.col('window_start'),
        pl.col('window_end')
    ])

    print(f"\nProcessing {len(hosp_windows):,} hospitalizations...")

    # ============================================================
    # MAIN LOOP: PROCESS EACH HOSPITALIZATION
    # ============================================================

    for hosp_row in tqdm(hosp_windows.iter_rows(named=True), total=len(hosp_windows), desc="Calculating DOT & ASC"):
        h_id = hosp_row['hospitalization_id']

        # Get all medications for this hospitalization
        h_meds = meds_pl.filter(pl.col('hospitalization_id') == h_id)

        # Note: We process ALL hospitalizations, even those with no antibiotics
        # If h_meds is empty, all windows will be counted as antibiotic-free days

        # Get windows for this hospitalization
        h_window_nums = hosp_row['window_num']
        h_window_starts = hosp_row['window_start']
        h_window_ends = hosp_row['window_end']

        # Initialize DOT counter for each antibiotic
        h_antibiotic_dot = {abx: 0 for abx in all_antibiotics}
        h_antibiotic_free_days = 0

        # --------------------------------------------------------
        # INNER LOOP: PROCESS EACH 24-HOUR WINDOW
        # --------------------------------------------------------

        for w_idx in range(len(h_window_nums)):
            w_num = h_window_nums[w_idx]
            w_start = h_window_starts[w_idx]
            w_end = h_window_ends[w_idx]

            # Filter medications in this window
            w_meds = h_meds.filter(
                (pl.col('admin_dttm') >= w_start) &
                (pl.col('admin_dttm') <= w_end)
            )

            # Track if any antibiotic was given in this window
            w_has_antibiotic = False

            # --------------------------------------------------------
            # CALCULATE DOT (Days of Therapy)
            # --------------------------------------------------------

            # Check each antibiotic for DOT counting
            for abx_cat in all_antibiotics:
                abx_in_win = w_meds.filter(
                    pl.col('med_category') == abx_cat
                )

                # Check if at least one non-null dose exists
                if len(abx_in_win) > 0:
                    has_dose = abx_in_win.filter(
                        pl.col('med_dose').is_not_null()
                    )

                    if len(has_dose) > 0:
                        h_antibiotic_dot[abx_cat] += 1
                        w_has_antibiotic = True

            # Count antibiotic-free days
            if not w_has_antibiotic:
                h_antibiotic_free_days += 1

            # --------------------------------------------------------
            # CALCULATE DAILY ASC (Antibiotic Spectrum Coverage)
            # --------------------------------------------------------

            # Get unique antibiotics used in this window (with non-null doses)
            w_meds_with_dose = w_meds.filter(pl.col('med_dose').is_not_null())
            antibiotics_in_window = w_meds_with_dose['med_category'].unique().to_list()

            # Sum spectrum scores for all antibiotics used in this window
            daily_asc = sum(
                spectrum_lookup.get(abx.strip(), 0)
                for abx in antibiotics_in_window
            )

            # Store daily ASC result
            daily_asc_results.append({
                'hospitalization_id': h_id,
                'window_num': w_num,
                'daily_asc': daily_asc
            })

        # --------------------------------------------------------
        # STORE RESULTS FOR THIS HOSPITALIZATION
        # --------------------------------------------------------

        # Store DOT results for this hospitalization
        for abx_cat, dot_count in h_antibiotic_dot.items():
            if dot_count > 0:  # Only store if DOT > 0 to save space
                dot_results.append({
                    'hospitalization_id': h_id,
                    'antibiotic': abx_cat,
                    'dot': dot_count
                })

        # Store antibiotic-free days
        dot_results.append({
            'hospitalization_id': h_id,
            'antibiotic': 'ANTIBIOTIC_FREE',
            'dot': h_antibiotic_free_days
        })

    # ============================================================
    # CREATE DOT DATAFRAMES
    # ============================================================

    # Create DOT DataFrame (long format)
    dot_long_pl = pl.DataFrame(dot_results)

    print(f"\n✓ DOT calculation complete")
    print(f"  Total DOT records: {len(dot_long_pl):,}")
    print(f"  Unique hospitalizations with antibiotics: {dot_long_pl.filter(pl.col('antibiotic') != 'ANTIBIOTIC_FREE')['hospitalization_id'].n_unique():,}")

    # Calculate Patient-Days (PD) per hospitalization
    print("\nCalculating Patient-Days per hospitalization...")
    pd_per_hosp = (
        windows_pl
        .group_by('hospitalization_id')
        .agg(pl.col('window_num').count().alias('PD'))
    )

    print(f"✓ Patient-Days calculated for {len(pd_per_hosp):,} hospitalizations")

    # Pivot DOT data from long to wide format
    print("\nPivoting DOT data to wide format...")
    dot_wide_pl = (
        dot_long_pl
        .pivot(
            index='hospitalization_id',
            columns='antibiotic',
            values='dot',
            aggregate_function='first'
        )
        .fill_null(0)  # Fill missing antibiotics with 0
    )

    # Join with Patient-Days
    dot_hospital_level = (
        dot_wide_pl
        .join(pd_per_hosp, on='hospitalization_id', how='left')
    )

    # Join with patient_id from cohort
    cohort_pl_for_join = pl.DataFrame(cohort_with_comorbidity[['patient_id', 'hospitalization_id']])
    dot_hospital_level = (
        dot_hospital_level
        .join(cohort_pl_for_join, on='hospitalization_id', how='left')
        # Reorder columns: patient_id, hospitalization_id, PD, then all antibiotics
        .select([
            'patient_id',
            'hospitalization_id',
            'PD'
        ] + [col for col in dot_hospital_level.columns if col not in ['patient_id', 'hospitalization_id', 'PD']])
    )

    print(f"✓ Hospital-level table created")
    print(f"  Shape: {dot_hospital_level.shape}")
    print(f"  Columns: {len(dot_hospital_level.columns)} (patient_id, hospitalization_id, PD, + {len(dot_hospital_level.columns)-3} antibiotics)")

    # ============================================================
    # CREATE DAILY ASC DATAFRAMES
    # ============================================================

    # Create Polars DataFrame for daily ASC
    daily_asc_patient_level = pl.DataFrame(daily_asc_results)

    print(f"\n✓ Daily ASC calculated")
    print(f"  Total records: {len(daily_asc_patient_level):,}")
    print(f"  Unique hospitalizations: {daily_asc_patient_level['hospitalization_id'].n_unique():,}")
    return daily_asc_patient_level, dot_hospital_level


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Calculate Antibiotic-Level DOT per 1000 Patient-Days

    ### Overview
    This section calculates **antibiotic-level metrics** to quantify consumption rates for each individual antibiotic across the entire cohort.

    ### Formula (Step 2 from README)
    ```
    DOT per 1000 PD = (Total DOT for antibiotic / Total PD) × 1000
    ```

    Where:
    - **Total DOT for antibiotic**: Sum of DOT values for that antibiotic across all hospitalizations
    - **Total PD**: Sum of all Patient-Days (number of 24-hour windows) across all hospitalizations

    ### Purpose
    - Standardizes antibiotic consumption rates across hospitals with different patient volumes
    - Enables fair comparison of antibiotic usage patterns
    - Identifies which antibiotics are most commonly used in the ICU

    ### Example
    If Vancomycin has:
    - Total DOT = 5,000 days (summed across all hospitalizations)
    - Total PD = 50,000 patient-days
    - DOT per 1000 PD = (5,000 / 50,000) × 1000 = 100

    This means: For every 1000 patient-days in the ICU, Vancomycin is administered on 100 days.
    """
    )
    return


@app.cell
def _(abx_spectrum, dot_hospital_level, pl):
    # Calculate antibiotic-level DOT per 1000 Patient-Days
    print("Calculating antibiotic-level DOT per 1000 Patient-Days...")

    # Calculate total Patient-Days across all hospitalizations
    total_pd = dot_hospital_level['PD'].sum()
    total_hospitalizations = len(dot_hospital_level)

    print(f"\nCohort totals:")
    print(f"  Total Patient-Days: {total_pd:,}")
    print(f"  Total Hospitalizations: {total_hospitalizations:,}")

    # Get antibiotic columns (exclude patient_id, hospitalization_id, PD, and ANTIBIOTIC_FREE)
    antibiotic_cols = [
        col for col in dot_hospital_level.columns
        if col not in ['patient_id', 'hospitalization_id', 'PD', 'ANTIBIOTIC_FREE']
    ]

    print(f"  Antibiotics to analyze: {len(antibiotic_cols)}")

    # Calculate metrics for each antibiotic
    antibiotic_metrics = []

    for abx_col in antibiotic_cols:
        # Sum DOT across all hospitalizations
        total_dot = dot_hospital_level[abx_col].sum()

        # Calculate DOT per 1000 PD
        dot_per_1000_pd = (total_dot / total_pd) * 1000

        # Count hospitalizations with this antibiotic (DOT > 0)
        hospitalizations_with_abx = (dot_hospital_level[abx_col] > 0).sum()

        # Calculate percentage of hospitalizations
        percent_hospitalizations = (hospitalizations_with_abx / total_hospitalizations) * 100

        antibiotic_metrics.append({
            'antibiotic': abx_col,
            'total_dot': total_dot,
            'total_pd': total_pd,
            'dot_per_1000_pd': dot_per_1000_pd,
            'n_any_antibiotic_during_hospitalization': hospitalizations_with_abx,
            'pct_any_antibiotic_during_hospitalization': percent_hospitalizations
        })

    # Create DataFrame
    dot_antibiotic_level = pl.DataFrame(antibiotic_metrics)

    # Merge with spectrum scores from abx_spectrum
    abx_spectrum_pl = pl.DataFrame(abx_spectrum).select([
        pl.col('med_category').alias('antibiotic'),
        'spectrum_score'
    ])

    dot_antibiotic_level = (
        dot_antibiotic_level
        .join(abx_spectrum_pl, on='antibiotic', how='left')
        .sort('dot_per_1000_pd', descending=True)
    )

    print(f"\n✓ Antibiotic-level metrics calculated")
    print(f"  Total antibiotics: {len(dot_antibiotic_level)}")
    print(f"\n=== Top 10 Antibiotics by DOT per 1000 PD ===")
    print(dot_antibiotic_level.head(10).to_pandas().to_string(index=False))
    return (dot_antibiotic_level,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Calculate Overall DOT per 1000 Patient-Days (Cohort-Level + Location Type Sub-Analysis)

    ### Overview
    This section calculates **overall antibiotic consumption** across the ENTIRE cohort, combining all antibiotics into a single aggregate metric. Additionally, we perform a **sub-analysis by ICU location type** to compare antibiotic usage across different ICU types.

    ### Formula (Step 4 from README)
    ```
    Overall DOT per 1000 PD = (Total DOT across all antibiotics / Total PD) × 1000
    ```

    Where:
    - **Total DOT across all antibiotics**: Sum of DOT values for ALL antibiotics for ALL hospitalizations (NOT per individual antibiotic)
    - **Total PD**: Sum of all Patient-Days across all hospitalizations

    ### Difference from Step 2
    - **Step 2 (Antibiotic-Level)**: DOT per 1000 PD calculated SEPARATELY for each individual antibiotic
    - **Step 4 (Cohort-Level)**: DOT per 1000 PD calculated for ALL antibiotics COMBINED into one aggregate metric

    ### Sub-Analysis: Location Type Stratification
    We stratify the cohort-level analysis by **ICU location type** (e.g., general ICU, cardiac ICU, surgical ICU, etc.) to evaluate:
    - Differences in antibiotic consumption patterns across ICU types
    - Whether certain ICU types have higher overall antibiotic burden
    - Comparative antibiotic stewardship across different critical care settings

    ### Example
    **Cohort-Level (Overall):**
    - Total DOT (all antibiotics, all hospitalizations) = 50,000 days
    - Total PD = 100,000 patient-days
    - Overall DOT per 1000 PD = (50,000 / 100,000) × 1000 = 500

    **Location-Type-Level (Cardiac ICU):**
    - Total DOT (all antibiotics, cardiac ICU only) = 8,000 days
    - Total PD (cardiac ICU only) = 12,000 patient-days
    - DOT per 1000 PD = (8,000 / 12,000) × 1000 = 667

    This means: For every 1000 patient-days, there are 500 days of antibiotic therapy overall, but 667 in the cardiac ICU specifically.
    """
    )
    return


@app.cell
def _(dot_hospital_level, pl):
    # Calculate cohort-level overall DOT metrics
    print("Calculating overall DOT per 1000 Patient-Days...")

    # Get antibiotic columns (exclude patient_id, hospitalization_id, PD, ANTIBIOTIC_FREE)
    antibiotic_cols_for_total = [
        col for col in dot_hospital_level.columns
        if col not in ['patient_id', 'hospitalization_id', 'PD', 'ANTIBIOTIC_FREE']
    ]

    print(f"  Antibiotics for total DOT calculation: {len(antibiotic_cols_for_total)}")

    # Calculate total DOT per hospitalization (sum across all antibiotics)
    print("\nCalculating total DOT per hospitalization...")
    dot_with_total = dot_hospital_level.with_columns(
        pl.sum_horizontal(antibiotic_cols_for_total).alias('total_dot')
    )

    # === COHORT-LEVEL OVERALL METRICS ===
    print("\n=== Cohort-Level Overall Metrics ===")

    total_dot_cohort = dot_with_total['total_dot'].sum()
    total_pd_cohort = dot_with_total['PD'].sum()
    overall_dot_per_1000_pd_cohort = round((total_dot_cohort / total_pd_cohort) * 1000, 3)

    print(f"Total DOT (all antibiotics, all hospitalizations): {total_dot_cohort:,}")
    print(f"Total PD (all hospitalizations): {total_pd_cohort:,}")
    print(f"Overall DOT per 1000 PD: {overall_dot_per_1000_pd_cohort:.3f}")

    # Create cohort-level DataFrame
    dot_cohort_level = pl.DataFrame({
        'metric': ['Total DOT', 'Total PD', 'Overall DOT per 1000 PD'],
        'value': [float(total_dot_cohort), float(total_pd_cohort), overall_dot_per_1000_pd_cohort]
    })

    print(f"\n✓ Cohort-level metrics calculated")
    return (dot_cohort_level,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Calculate Daily Antibiotic Spectrum Coverage (ASC) per Window

    ### Overview (Step 5 from README)
    This section calculates the **Antibiotic Spectrum Coverage (ASC)** for each 24-hour window (ICU day) since admission. ASC quantifies the breadth of antibiotic coverage by summing the spectrum scores of all antibiotics used on a given day.

    ### Key Concept: Windows = ICU Days
    - **Window 0** = ICU Day 0 (first 24 hours)
    - **Window 1** = ICU Day 1 (second 24 hours)
    - ...
    - **Window 10** = ICU Day 10

    We analyze the first 11 windows (Days 0-10) to capture early ICU antibiotic patterns.

    ### Formula
    ```
    Daily ASC = Σ (spectrum_score for each antibiotic used in window)
    ```

    ### Example
    **Window 0 (ICU Day 0):**
    - Vancomycin (spectrum score = 3.5)
    - Piperacillin-Tazobactam (spectrum score = 4.2)
    - Daily ASC = 3.5 + 4.2 = 7.7

    **Window 1 (ICU Day 1):**
    - Vancomycin only (spectrum score = 3.5)
    - Daily ASC = 3.5

    **Window 2 (ICU Day 2):**
    - No antibiotics
    - Daily ASC = 0

    ### Purpose
    - Track antibiotic spectrum intensity over the first 10 ICU days
    - Identify escalation or de-escalation patterns
    - Enable cross-site comparison of antibiotic stewardship practices

    ### Output Files (for sharing across sites)
    - **Patient-level**: Daily ASC for each hospitalization for each window (0-10)
    - **Summary**: Mean and SD of daily ASC per window (for multi-site comparison)
    """
    )
    return


@app.cell
def _(daily_asc_patient_level, pl):
    # Calculate summary statistics per window (for sharing across sites)
    # Filter to windows 0-10 for summary/plotting
    print("Calculating summary statistics per window (Days 0-10 for plotting)...")
    daily_asc_summary = (
        daily_asc_patient_level
        .filter(pl.col('window_num') <= 10)  # Only windows 0-10 for summary
        .group_by('window_num')
        .agg([
            pl.col('daily_asc').mean().alias('mean_asc'),
            pl.col('daily_asc').std().alias('sd_asc'),
            pl.col('daily_asc').median().alias('median_asc'),
            pl.col('daily_asc').quantile(0.25).alias('q25_asc'),
            pl.col('daily_asc').quantile(0.75).alias('q75_asc'),
            pl.col('daily_asc').min().alias('min_asc'),
            pl.col('daily_asc').max().alias('max_asc'),
            pl.col('hospitalization_id').count().alias('n_hospitalizations')
        ])
        .with_columns([
            (pl.col('sd_asc') / pl.col('n_hospitalizations').sqrt()).alias('se_asc'),
            (pl.col('mean_asc') - 1.96 * (pl.col('sd_asc') / pl.col('n_hospitalizations').sqrt())).alias('lower_ci_asc'),
            (pl.col('mean_asc') + 1.96 * (pl.col('sd_asc') / pl.col('n_hospitalizations').sqrt())).alias('upper_ci_asc')
        ])
        .sort('window_num')
    )

    print(f"\n=== Daily ASC Summary (ICU Days 0-10) ===")
    print(daily_asc_summary.to_pandas().to_string(index=False))

    print(f"\n✓ Daily ASC summary calculated")
    return (daily_asc_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Calculate DASC per 1000 Patient-Days

    ### Overview (Step 6 from README)
    This section calculates **Days of Antibiotic Spectrum Coverage (DASC)**, which is the cumulative spectrum coverage across a patient's entire ICU stay.

    ### Formula
    ```
    DASC = Σ (daily_asc across all ICU days)
    DASC per 1000 PD = (DASC / PD) × 1000
    ```

    Where:
    - **DASC**: Sum of daily ASC values across ALL windows during ICU stay
    - **PD**: Total Patient-Days (number of 24-hour windows)
    - **DASC per 1000 PD**: Standardized metric for comparing spectrum intensity

    ### Example
    **Hospitalization with 5 ICU days:**
    - Window 0: Daily ASC = 7.5
    - Window 1: Daily ASC = 7.5
    - Window 2: Daily ASC = 3.5
    - Window 3: Daily ASC = 3.5
    - Window 4: Daily ASC = 0

    **DASC** = 7.5 + 7.5 + 3.5 + 3.5 + 0 = 22.0

    **PD** = 5 windows = 5 patient-days

    **DASC per 1000 PD** = (22.0 / 5) × 1000 = 4,400

    ### Purpose
    - Quantifies cumulative antibiotic spectrum burden per patient
    - Enables comparison across hospitalizations with different lengths of stay
    - Identifies high vs. low spectrum antibiotic utilization patterns

    ### Outputs
    - **Overall**: DASC per 1000 PD for entire cohort (all years combined)
    - **By Year**: DASC per 1000 PD stratified by year (2018-2024)
    """
    )
    return


@app.cell
def _(
    cohort_with_comorbidity,
    daily_asc_patient_level,
    dot_hospital_level,
    pl,
):
    # Calculate DASC per 1000 Patient-Days
    print("Calculating DASC per 1000 Patient-Days...")

    # Sum daily ASC across ALL windows per hospitalization to get DASC
    print("\nCalculating total DASC per hospitalization...")
    dasc_per_hosp = (
        daily_asc_patient_level
        .group_by('hospitalization_id')
        .agg(pl.col('daily_asc').sum().alias('DASC'))
    )

    print(f"✓ DASC calculated for {len(dasc_per_hosp):,} hospitalizations")

    # Join with PD from dot_hospital_level
    dasc_with_pd = (
        dasc_per_hosp
        .join(
            dot_hospital_level.select(['hospitalization_id', 'PD']),
            on='hospitalization_id',
            how='left'
        )
        .with_columns(
            ((pl.col('DASC') / pl.col('PD')) * 1000).alias('dasc_per_1000_pd')
        )
    )

    # === OVERALL DASC METRICS (all years combined) ===
    print("\n=== Overall DASC Metrics (All Years) ===")

    total_dasc_overall = dasc_with_pd['DASC'].sum()
    total_pd_overall = dasc_with_pd['PD'].sum()
    dasc_per_1000_pd_overall = (total_dasc_overall / total_pd_overall) * 1000

    print(f"Total DASC (all hospitalizations): {total_dasc_overall:,.2f}")
    print(f"Total PD (all hospitalizations): {total_pd_overall:,}")
    print(f"DASC per 1000 PD: {dasc_per_1000_pd_overall:.2f}")

    # Create overall metrics DataFrame
    dasc_overall = pl.DataFrame({
        'metric': ['Total DASC', 'Total PD', 'DASC per 1000 PD'],
        'value': [float(total_dasc_overall), float(total_pd_overall), float(dasc_per_1000_pd_overall)]
    })

    # === DASC METRICS BY YEAR ===
    print("\n=== DASC Metrics by Year ===")

    # Extract year from cohort start_dttm
    cohort_with_year = pl.DataFrame(cohort_with_comorbidity).select([
        'hospitalization_id',
        pl.col('start_dttm').dt.year().alias('year')
    ])

    # Join year with DASC data
    dasc_with_year = (
        dasc_with_pd
        .join(cohort_with_year, on='hospitalization_id', how='left')
    )

    # Calculate DASC per 1000 PD by year
    dasc_by_year = (
        dasc_with_year
        .group_by('year')
        .agg([
            pl.col('DASC').sum().alias('total_dasc'),
            pl.col('PD').sum().alias('total_pd'),
            pl.col('hospitalization_id').count().alias('hospitalizations')
        ])
        .with_columns(
            ((pl.col('total_dasc') / pl.col('total_pd')) * 1000).round(3).alias('dasc_per_1000_pd')
        )
        .sort('year')
    )

    print(dasc_by_year.to_pandas().to_string(index=False))

    print(f"\n✓ DASC per 1000 PD calculated")
    return dasc_by_year, dasc_overall


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Calculate Antibiotic-Free Days (AFD)

    ### Overview (Step 7 from README)
    This section calculates **Antibiotic-Free Days (AFD)**, which is the ratio of days without any antibiotic administration to total ICU days for each hospitalization.

    ### Key Concept
    **Antibiotic-Free Days** were already counted during the DOT calculation and stored in the `ANTIBIOTIC_FREE` column.

    ### Formula
    ```
    AFD Rate = (Days without antibiotics / Days in ICU) per hospitalization
    ```

    Where:
    - **Days without antibiotics**: Number of 24-hour windows with zero antibiotic administrations (ANTIBIOTIC_FREE column)
    - **Days in ICU**: Total Patient-Days (PD column = number of 24-hour windows)
    - **AFD Rate**: Proportion of ICU stay without antibiotics (0.0 to 1.0)

    ### Example
    **Hospitalization with 5 ICU days:**
    - Day 0: Vancomycin + Piperacillin-Tazobactam (antibiotics given)
    - Day 1: Vancomycin (antibiotics given)
    - Day 2: No antibiotics (antibiotic-free)
    - Day 3: No antibiotics (antibiotic-free)
    - Day 4: Vancomycin (antibiotics given)

    **Antibiotic-Free Days** = 2 (Days 2, 3)

    **Days in ICU** = 5

    **AFD Rate** = 2 / 5 = 0.40 (40%)

    ### Purpose
    - Quantifies antibiotic stewardship effectiveness
    - Identifies opportunities for antibiotic de-escalation
    - Enables benchmarking across sites
    - Higher AFD rate may indicate better antibiotic stewardship

    ### Outputs
    - **Patient-level**: AFD rate for each hospitalization
    - **Summary**: Mean, SD, median, min, max of AFD rates (for sharing)
    """
    )
    return


@app.cell
def _(dot_hospital_level, pl):
    # Calculate Antibiotic-Free Days (AFD) Rate
    print("Calculating Antibiotic-Free Days (AFD) Rate...")

    # Extract relevant columns from dot_hospital_level
    afd_patient_level = (
        dot_hospital_level
        .select([
            'hospitalization_id',
            pl.col('ANTIBIOTIC_FREE').alias('antibiotic_free_days'),
            pl.col('PD').alias('days_in_icu')
        ])
        .with_columns(
            (pl.col('antibiotic_free_days') / pl.col('days_in_icu')).alias('afd_rate')
        )
    )

    print(f"✓ AFD calculated for {len(afd_patient_level):,} hospitalizations")

    # Calculate summary statistics
    mean_afd_rate = afd_patient_level['afd_rate'].mean()
    std_afd_rate = afd_patient_level['afd_rate'].std()
    median_afd_rate = afd_patient_level['afd_rate'].median()
    min_afd_rate = afd_patient_level['afd_rate'].min()
    max_afd_rate = afd_patient_level['afd_rate'].max()
    q25_afd_rate = afd_patient_level['afd_rate'].quantile(0.25)
    q75_afd_rate = afd_patient_level['afd_rate'].quantile(0.75)
    total_hospitalizations_afd = len(afd_patient_level)

    # Calculate standard error
    import math
    se_afd_rate = std_afd_rate / math.sqrt(total_hospitalizations_afd)

    print(f"\n=== Antibiotic-Free Days (AFD) Summary ===")
    print(f"  Mean AFD Rate: {mean_afd_rate:.4f} ({mean_afd_rate*100:.2f}%)")
    print(f"  SD AFD Rate: {std_afd_rate:.4f}")
    print(f"  SE AFD Rate: {se_afd_rate:.4f}")
    print(f"  Median AFD Rate: {median_afd_rate:.4f} ({median_afd_rate*100:.2f}%)")
    print(f"  25th Percentile: {q25_afd_rate:.4f} ({q25_afd_rate*100:.2f}%)")
    print(f"  75th Percentile: {q75_afd_rate:.4f} ({q75_afd_rate*100:.2f}%)")
    print(f"  Min AFD Rate: {min_afd_rate:.4f} ({min_afd_rate*100:.2f}%)")
    print(f"  Max AFD Rate: {max_afd_rate:.4f} ({max_afd_rate*100:.2f}%)")
    print(f"  Total Hospitalizations: {total_hospitalizations_afd:,}")

    # Create summary DataFrame
    afd_summary = pl.DataFrame({
        'metric': [
            'mean_afd_rate',
            'std_afd_rate',
            'se_afd_rate',
            'median_afd_rate',
            'q25_afd_rate',
            'q75_afd_rate',
            'min_afd_rate',
            'max_afd_rate',
            'total_hospitalizations'
        ],
        'value': [
            float(mean_afd_rate),
            float(std_afd_rate),
            float(se_afd_rate),
            float(median_afd_rate),
            float(q25_afd_rate),
            float(q75_afd_rate),
            float(min_afd_rate),
            float(max_afd_rate),
            float(total_hospitalizations_afd)
        ]
    })

    print(f"\n✓ AFD summary calculated")
    return afd_patient_level, afd_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Calculate Year-Based ASC Time Series Summary

    ### Overview
    This section calculates **year-based summary statistics for Antibiotic Spectrum Coverage (ASC)** to enable time series plotting and trend analysis across years 2018-2024.

    ### Purpose
    - Track ASC trends over time to identify changes in antibiotic prescribing patterns
    - Enable cross-site comparison of ASC evolution across healthcare centers
    - Support generation of time series plots showing ASC trajectories by year

    ### Method
    1. Join `daily_asc_patient_level` (window-level ASC data) with `windows_pl` (window timestamps)
    2. Extract year from `window_start` datetime
    3. Group by year and calculate summary statistics:
       - Mean ASC per year (average across all windows in that year)
       - Standard deviation of ASC
       - Median, min, max ASC
       - Number of window observations
       - Number of unique hospitalizations

    ### Formula
    ```
    For each year (2018-2024):
        Year-based Mean ASC = Mean(daily_asc) across all windows in that year
    ```

    ### Example Output
    ```
    year | mean_asc | sd_asc | n_windows | n_hospitalizations
    -----|----------|--------|-----------|-------------------
    2018 |   4.52   |  3.21  |   15,234  |       1,245
    2019 |   4.68   |  3.34  |   18,567  |       1,512
    2020 |   5.12   |  3.58  |   20,123  |       1,678
    ```
    """
    )
    return


@app.cell
def _(daily_asc_patient_level, pl, windows_pl):
    # Calculate year-based ASC summary for time series plotting
    print("Calculating year-based ASC summary for time series analysis...")

    # Join daily_asc_patient_level with windows_pl to get window timestamps
    print("\nJoining daily ASC data with window timestamps...")
    asc_with_timestamps = (
        daily_asc_patient_level
        .join(
            windows_pl.select(['hospitalization_id', 'window_num', 'window_start']),
            on=['hospitalization_id', 'window_num'],
            how='left'
        )
    )

    print(f"✓ Joined {len(asc_with_timestamps):,} records")

    # Extract year from window_start
    print("\nExtracting year from window_start timestamps...")
    asc_with_year = asc_with_timestamps.with_columns(
        pl.col('window_start').dt.year().alias('year')
    )

    print(f"✓ Year extracted")
    print(f"  Year range: {asc_with_year['year'].min()} - {asc_with_year['year'].max()}")

    # Group by year and calculate summary statistics
    print("\nCalculating summary statistics by year...")
    asc_by_year_summary = (
        asc_with_year
        .group_by('year')
        .agg([
            pl.col('daily_asc').mean().alias('mean_asc'),
            pl.col('daily_asc').std().alias('sd_asc'),
            pl.col('daily_asc').median().alias('median_asc'),
            pl.col('daily_asc').quantile(0.25).alias('q25_asc'),
            pl.col('daily_asc').quantile(0.75).alias('q75_asc'),
            pl.col('daily_asc').min().alias('min_asc'),
            pl.col('daily_asc').max().alias('max_asc'),
            pl.col('window_num').count().alias('n_windows'),
            pl.col('hospitalization_id').n_unique().alias('n_hospitalizations')
        ])
        .with_columns([
            (pl.col('sd_asc') / pl.col('n_windows').sqrt()).alias('se_asc'),
            (pl.col('mean_asc') - 1.96 * (pl.col('sd_asc') / pl.col('n_windows').sqrt())).alias('lower_ci_asc'),
            (pl.col('mean_asc') + 1.96 * (pl.col('sd_asc') / pl.col('n_windows').sqrt())).alias('upper_ci_asc')
        ])
        .sort('year')
    )

    print(f"\n=== Year-Based ASC Summary ===")
    print(asc_by_year_summary.to_pandas().to_string(index=False))

    print(f"\n✓ Year-based ASC summary calculated")
    print(f"  Years analyzed: {len(asc_by_year_summary)}")
    print(f"  Total windows: {asc_by_year_summary['n_windows'].sum():,}")
    print(f"\n📊 This data enables time series plotting of ASC trends over years 2018-2024")
    return (asc_by_year_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Save Results""")
    return


@app.cell
def _(
    Path,
    afd_patient_level,
    afd_summary,
    asc_by_year_summary,
    daily_asc_patient_level,
    daily_asc_summary,
    dasc_by_year,
    dasc_overall,
    dot_antibiotic_level,
    dot_cohort_level,
    dot_hospital_level,
    pl,
    site_name,
):
    # Save results
    print("Saving results...")

    # Ensure output directories exist
    Path('PHI_DATA').mkdir(exist_ok=True)
    Path('RESULTS_UPLOAD_ME').mkdir(exist_ok=True)

    print("\n=== PHI DATA (Patient-Level - Do Not Share) ===")

    # Save hospital-level DOT table (wide format with PD column)
    print("\n1. Hospital-level DOT table (wide format):")
    dot_hospital_level_with_site = dot_hospital_level.with_columns(
        pl.lit(site_name).alias('site')
    ).select(['site', 'hospitalization_id'] + [col for col in dot_hospital_level.columns if col != 'hospitalization_id'])
    dot_hospital_level_with_site.write_parquet(Path('PHI_DATA') / 'dot_hospital_level.parquet')
    print(f"   ✓ Saved: PHI_DATA/dot_hospital_level.parquet")
    print(f"   Shape: {dot_hospital_level_with_site.shape}")

    # Save daily ASC patient-level data
    print("\n2. Daily ASC patient-level (all windows):")
    daily_asc_patient_level_with_site = daily_asc_patient_level.with_columns(
        pl.lit(site_name).alias('site')
    ).select(['site'] + [col for col in daily_asc_patient_level.columns])
    daily_asc_patient_level_with_site.write_parquet(Path('PHI_DATA') / 'daily_asc_patient_level.parquet')
    print(f"   ✓ Saved: PHI_DATA/daily_asc_patient_level.parquet")
    print(f"   Shape: {daily_asc_patient_level_with_site.shape}")

    # Save AFD patient-level data
    print("\n3. AFD patient-level (all hospitalizations):")
    afd_patient_level_with_site = afd_patient_level.with_columns(
        pl.lit(site_name).alias('site')
    ).select(['site'] + [col for col in afd_patient_level.columns])
    afd_patient_level_with_site.write_parquet(Path('PHI_DATA') / 'afd_patient_level.parquet')
    print(f"   ✓ Saved: PHI_DATA/afd_patient_level.parquet")
    print(f"   Shape: {afd_patient_level_with_site.shape}")

    print("\n=== SUMMARY DATA (Safe to Share - Upload These Files) ===")

    # Save antibiotic-level metrics
    print("\n4. Antibiotic-level metrics:")
    dot_antibiotic_level_with_site = dot_antibiotic_level.with_columns(
        pl.lit(site_name).alias('site')
    ).select(['site'] + [col for col in dot_antibiotic_level.columns])
    dot_antibiotic_level_with_site.write_csv(Path('RESULTS_UPLOAD_ME') / 'dot_antibiotic_level.csv')
    print(f"   ✓ Saved: RESULTS_UPLOAD_ME/dot_antibiotic_level.csv")
    print(f"   Shape: {dot_antibiotic_level_with_site.shape}")

    # Save cohort-level overall metrics
    print("\n5. Cohort-level overall metrics:")
    dot_cohort_level_with_site = dot_cohort_level.with_columns(
        pl.lit(site_name).alias('site')
    ).select(['site'] + [col for col in dot_cohort_level.columns])
    dot_cohort_level_with_site.write_csv(Path('RESULTS_UPLOAD_ME') / 'dot_cohort_level.csv')
    print(f"   ✓ Saved: RESULTS_UPLOAD_ME/dot_cohort_level.csv")
    print(f"   Shape: {dot_cohort_level_with_site.shape}")

    # REMOVED: Location-type-level metrics not needed for MICU-only cohort
    # # Save location-type-level metrics
    # print("\n6. Location-type-level metrics:")
    # dot_location_type_level_with_site = dot_location_type_level.with_columns(
    #     pl.lit(site_name).alias('site')
    # ).select(['site'] + [col for col in dot_location_type_level.columns])
    # dot_location_type_level_with_site.write_csv(Path('RESULTS_UPLOAD_ME') / 'dot_location_type_level.csv')
    # print(f"   ✓ Saved: RESULTS_UPLOAD_ME/dot_location_type_level.csv")
    # print(f"   Shape: {dot_location_type_level_with_site.shape}")

    # Save daily ASC summary (for sharing)
    print("\n6. Daily ASC summary (windows 0-10, for sharing):")
    daily_asc_summary_with_site = daily_asc_summary.with_columns(
        pl.lit(site_name).alias('site')
    ).select(['site'] + [col for col in daily_asc_summary.columns])
    daily_asc_summary_with_site.write_csv(Path('RESULTS_UPLOAD_ME') / 'daily_asc_summary.csv')
    print(f"   ✓ Saved: RESULTS_UPLOAD_ME/daily_asc_summary.csv")
    print(f"   Shape: {daily_asc_summary_with_site.shape}")

    # Save DASC overall metrics
    print("\n7. DASC overall metrics:")
    dasc_overall_with_site = dasc_overall.with_columns(
        pl.lit(site_name).alias('site')
    ).select(['site'] + [col for col in dasc_overall.columns])
    dasc_overall_with_site.write_csv(Path('RESULTS_UPLOAD_ME') / 'dasc_overall.csv')
    print(f"   ✓ Saved: RESULTS_UPLOAD_ME/dasc_overall.csv")
    print(f"   Shape: {dasc_overall_with_site.shape}")

    # Save DASC by year metrics
    print("\n8. DASC by year metrics:")
    dasc_by_year_with_site = dasc_by_year.with_columns(
        pl.lit(site_name).alias('site')
    ).select(['site'] + [col for col in dasc_by_year.columns])
    dasc_by_year_with_site.write_csv(Path('RESULTS_UPLOAD_ME') / 'dasc_by_year.csv')
    print(f"   ✓ Saved: RESULTS_UPLOAD_ME/dasc_by_year.csv")
    print(f"   Shape: {dasc_by_year_with_site.shape}")

    # Save AFD summary (for sharing)
    print("\n9. AFD summary (for sharing):")
    afd_summary_with_site = afd_summary.with_columns(
        pl.lit(site_name).alias('site')
    ).select(['site'] + [col for col in afd_summary.columns])
    afd_summary_with_site.write_csv(Path('RESULTS_UPLOAD_ME') / 'afd_summary.csv')
    print(f"   ✓ Saved: RESULTS_UPLOAD_ME/afd_summary.csv")
    print(f"   Shape: {afd_summary_with_site.shape}")

    # Save year-based ASC summary (for time series plotting)
    print("\n10. Year-based ASC summary (for time series plotting):")
    asc_by_year_summary_with_site = asc_by_year_summary.with_columns(
        pl.lit(site_name).alias('site')
    ).select(['site'] + [col for col in asc_by_year_summary.columns])
    asc_by_year_summary_with_site.write_csv(Path('RESULTS_UPLOAD_ME') / 'asc_by_year_summary.csv')
    print(f"   ✓ Saved: RESULTS_UPLOAD_ME/asc_by_year_summary.csv")
    print(f"   Shape: {asc_by_year_summary_with_site.shape}")

    print(f"\n✓ All results saved successfully!")
    print(f"\n{'='*80}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Visualizations: ASC Time Series Plots""")
    return


@app.cell
def _(Path, asc_by_year_summary, np, plt, scipy):
    # Plot 1: ASC Trend by Year
    print("Creating ASC by Year plot...")

    # Convert to pandas for easier plotting
    df_year = asc_by_year_summary.to_pandas().sort_values('year')

    # Extract data
    years = df_year['year'].values
    mean_asc_year = df_year['mean_asc'].values
    sd_asc_year = df_year['sd_asc'].values
    n_windows_year = df_year['n_windows'].values

    # Calculate 95% CI
    se_asc_year = sd_asc_year / np.sqrt(n_windows_year)
    ci_lower_year = mean_asc_year - 1.96 * se_asc_year
    ci_upper_year = mean_asc_year + 1.96 * se_asc_year

    # Create spline interpolation (if enough points)
    if len(years) >= 4:
        # Cubic spline requires at least 4 points
        years_smooth = np.linspace(years.min(), years.max(), 500)
        spline = scipy.interpolate.make_interp_spline(years, mean_asc_year, k=3)
        mean_smooth = spline(years_smooth)

        # Interpolate CI bounds as well for smooth error bands
        spline_ci_lower = scipy.interpolate.make_interp_spline(years, ci_lower_year, k=3)
        ci_lower_smooth = spline_ci_lower(years_smooth)

        spline_ci_upper = scipy.interpolate.make_interp_spline(years, ci_upper_year, k=3)
        ci_upper_smooth = spline_ci_upper(years_smooth)
    else:
        # Fall back to original data if insufficient points
        years_smooth = years
        mean_smooth = mean_asc_year
        ci_lower_smooth = ci_lower_year
        ci_upper_smooth = ci_upper_year

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mean line
    ax.plot(years_smooth, mean_smooth, color='#2E86AB', linewidth=2.5, label='Mean ASC')

    # Plot error band (mean ± 95% CI)
    ax.fill_between(
        years_smooth,
        ci_lower_smooth,
        ci_upper_smooth,
        alpha=0.3,
        color='#2E86AB',
        label='±95% CI'
    )

    # Plot original data points
    ax.scatter(years, mean_asc_year, color='#A23B72', s=80, zorder=5, label='Observed Mean')

    # Styling
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Antibiotic Spectrum Coverage (ASC)', fontsize=12, fontweight='bold')
    ax.set_title('Antibiotic Spectrum Coverage Trend Over Time\n(Single-Site Analysis)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)

    # Format x-axis to show integer years
    ax.set_xticks(years)
    ax.set_xticklabels([int(y) for y in years])

    # Add sample size annotation
    total_windows = df_year['n_windows'].sum()
    total_patients = df_year['n_hospitalizations'].sum()
    ax.text(0.02, 0.98, f'Total: {total_patients:,} hospitalizations, {total_windows:,} patient-days',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save
    plt.savefig(Path('RESULTS_UPLOAD_ME') / 'asc_by_year_plot.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: RESULTS_UPLOAD_ME/asc_by_year_plot.png")

    plt.close()
    return


@app.cell
def _(Path, daily_asc_summary, np, plt, scipy):
    # Plot 2: ASC Trend by ICU Day (Window)
    print("Creating ASC by ICU Day plot...")

    # Convert to pandas for easier plotting
    df_window = daily_asc_summary.to_pandas().sort_values('window_num')

    # Extract data
    windows = df_window['window_num'].values
    mean_asc_window = df_window['mean_asc'].values
    sd_asc_window = df_window['sd_asc'].values
    n_hospitalizations_window = df_window['n_hospitalizations'].values

    # Calculate 95% CI
    se_asc_window = sd_asc_window / np.sqrt(n_hospitalizations_window)
    ci_lower_window = mean_asc_window - 1.96 * se_asc_window
    ci_upper_window = mean_asc_window + 1.96 * se_asc_window

    # Create spline interpolation
    if len(windows) >= 4:
        win_windows_smooth = np.linspace(windows.min(), windows.max(), 500)
        win_spline = scipy.interpolate.make_interp_spline(windows, mean_asc_window, k=3)
        win_mean_smooth = win_spline(win_windows_smooth)

        win_spline_ci_lower = scipy.interpolate.make_interp_spline(windows, ci_lower_window, k=3)
        win_ci_lower_smooth = win_spline_ci_lower(win_windows_smooth)

        win_spline_ci_upper = scipy.interpolate.make_interp_spline(windows, ci_upper_window, k=3)
        win_ci_upper_smooth = win_spline_ci_upper(win_windows_smooth)
    else:
        win_windows_smooth = windows
        win_mean_smooth = mean_asc_window
        win_ci_lower_smooth = ci_lower_window
        win_ci_upper_smooth = ci_upper_window

    # Create figure
    win_fig, win_ax = plt.subplots(figsize=(10, 6))

    # Plot mean line
    win_ax.plot(win_windows_smooth, win_mean_smooth, color='#06A77D', linewidth=2.5, label='Mean ASC')

    # Plot error band (mean ± 95% CI)
    win_ax.fill_between(
        win_windows_smooth,
        win_ci_lower_smooth,
        win_ci_upper_smooth,
        alpha=0.3,
        color='#06A77D',
        label='±95% CI'
    )

    # Plot original data points
    win_ax.scatter(windows, mean_asc_window, color='#D62828', s=80, zorder=5, label='Observed Mean')

    # Styling
    win_ax.set_xlabel('ICU Day (Window Number)', fontsize=12, fontweight='bold')
    win_ax.set_ylabel('Mean Antibiotic Spectrum Coverage (ASC)', fontsize=12, fontweight='bold')
    win_ax.set_title('Antibiotic Spectrum Coverage by ICU Day\n(Days 0-10 Post-Admission)',
                 fontsize=14, fontweight='bold', pad=20)
    win_ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    win_ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)

    # Format x-axis
    win_ax.set_xticks(windows)
    win_ax.set_xticklabels([f'Day {int(w)}' for w in windows])

    # Add sample size annotation
    win_total_windows = df_window['n_hospitalizations'].sum()
    win_ax.text(0.02, 0.98, f'Sample: {win_total_windows:,} patient-window observations',
            transform=win_ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save
    plt.savefig(Path('RESULTS_UPLOAD_ME') / 'asc_by_window_plot.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: RESULTS_UPLOAD_ME/asc_by_window_plot.png")

    plt.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Generate Table 1: Summary Statistics for Multi-Site Comparison""")
    return


@app.cell
def _(
    Path,
    afd_summary,
    cohort_with_comorbidity,
    daily_asc_summary,
    dasc_overall,
    dot_antibiotic_level,
    dot_cohort_level,
    json,
    pl,
    site_name,
):
    print("Generating Table 1 summary statistics...")

    # ============================================================
    # DEMOGRAPHICS & BASELINE
    # ============================================================

    cohort_pl_t1 = pl.DataFrame(cohort_with_comorbidity)

    t1_total_patients = len(cohort_pl_t1)

    # ============================================================
    # MISSINGNESS CALCULATIONS
    # ============================================================

    # Calculate null counts and percentages for all tracked variables
    # Demographics
    age_n_missing = cohort_pl_t1['age_at_admission'].is_null().sum() if 'age_at_admission' in cohort_pl_t1.columns else t1_total_patients
    age_pct_missing = (age_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    sex_n_missing = cohort_pl_t1['sex_category'].is_null().sum() if 'sex_category' in cohort_pl_t1.columns else t1_total_patients
    sex_pct_missing = (sex_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    race_n_missing = cohort_pl_t1['race_ethnicity'].is_null().sum() if 'race_ethnicity' in cohort_pl_t1.columns else t1_total_patients
    race_pct_missing = (race_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    bmi_n_missing = cohort_pl_t1['bmi'].is_null().sum() if 'bmi' in cohort_pl_t1.columns else t1_total_patients
    bmi_pct_missing = (bmi_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    # Vitals
    highest_temp_n_missing = cohort_pl_t1['highest_temperature'].is_null().sum() if 'highest_temperature' in cohort_pl_t1.columns else t1_total_patients
    highest_temp_pct_missing = (highest_temp_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    lowest_temp_n_missing = cohort_pl_t1['lowest_temperature'].is_null().sum() if 'lowest_temperature' in cohort_pl_t1.columns else t1_total_patients
    lowest_temp_pct_missing = (lowest_temp_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    lowest_map_n_missing = cohort_pl_t1['lowest_map'].is_null().sum() if 'lowest_map' in cohort_pl_t1.columns else t1_total_patients
    lowest_map_pct_missing = (lowest_map_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    # Severity
    sofa_n_missing = cohort_pl_t1['sofa_total'].is_null().sum() if 'sofa_total' in cohort_pl_t1.columns else t1_total_patients
    sofa_pct_missing = (sofa_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    # Comorbidity Scores
    elix_score_n_missing = cohort_pl_t1['elix_score'].is_null().sum() if 'elix_score' in cohort_pl_t1.columns else t1_total_patients
    elix_score_pct_missing = (elix_score_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    cci_score_n_missing = cohort_pl_t1['cci_score'].is_null().sum() if 'cci_score' in cohort_pl_t1.columns else t1_total_patients
    cci_score_pct_missing = (cci_score_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    chronic_pulm_n_missing = cohort_pl_t1['chronic_pulmonary_disease'].is_null().sum() if 'chronic_pulmonary_disease' in cohort_pl_t1.columns else t1_total_patients
    chronic_pulm_pct_missing = (chronic_pulm_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    # Interventions
    vasopressor_n_missing = cohort_pl_t1['vasopressor_ever'].is_null().sum() if 'vasopressor_ever' in cohort_pl_t1.columns else t1_total_patients
    vasopressor_pct_missing = (vasopressor_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    nippv_n_missing = cohort_pl_t1['NIPPV_ever'].is_null().sum() if 'NIPPV_ever' in cohort_pl_t1.columns else t1_total_patients
    nippv_pct_missing = (nippv_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    hfno_n_missing = cohort_pl_t1['HFNO_ever'].is_null().sum() if 'HFNO_ever' in cohort_pl_t1.columns else t1_total_patients
    hfno_pct_missing = (hfno_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    imv_n_missing = cohort_pl_t1['IMV_ever'].is_null().sum() if 'IMV_ever' in cohort_pl_t1.columns else t1_total_patients
    imv_pct_missing = (imv_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    cvvh_n_missing = t1_total_patients  # Not available yet
    cvvh_pct_missing = 100.0

    # Labs
    wbc_n_missing = cohort_pl_t1['highest_wbc'].is_null().sum() if 'highest_wbc' in cohort_pl_t1.columns else t1_total_patients
    wbc_pct_missing = (wbc_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    creatinine_n_missing = cohort_pl_t1['highest_creatinine'].is_null().sum() if 'highest_creatinine' in cohort_pl_t1.columns else t1_total_patients
    creatinine_pct_missing = (creatinine_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    # Outcomes
    hospital_los_n_missing = cohort_pl_t1['hospital_los_days'].is_null().sum() if 'hospital_los_days' in cohort_pl_t1.columns else t1_total_patients
    hospital_los_pct_missing = (hospital_los_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    icu_los_n_missing = cohort_pl_t1['icu_los_days'].is_null().sum() if 'icu_los_days' in cohort_pl_t1.columns else t1_total_patients
    icu_los_pct_missing = (icu_los_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    inpatient_mortality_n_missing = cohort_pl_t1['inpatient_mortality'].is_null().sum() if 'inpatient_mortality' in cohort_pl_t1.columns else t1_total_patients
    inpatient_mortality_pct_missing = (inpatient_mortality_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    icu_mortality_n_missing = cohort_pl_t1['icu_mortality'].is_null().sum() if 'icu_mortality' in cohort_pl_t1.columns else t1_total_patients
    icu_mortality_pct_missing = (icu_mortality_n_missing / t1_total_patients * 100) if t1_total_patients > 0 else 0

    # ============================================================


    t1_total_pd = cohort_pl_t1.with_columns(
        ((pl.col('end_dttm') - pl.col('start_dttm')).dt.total_seconds() / 86400).alias('icu_los_days')
    )

    # Calculate demographics
    t1_mean_age = cohort_pl_t1['age_at_admission'].mean() if 'age_at_admission' in cohort_pl_t1.columns else None
    t1_sd_age = cohort_pl_t1['age_at_admission'].std() if 'age_at_admission' in cohort_pl_t1.columns else None

    # BMI
    t1_mean_bmi = cohort_pl_t1['bmi'].mean() if 'bmi' in cohort_pl_t1.columns else None
    t1_sd_bmi = cohort_pl_t1['bmi'].std() if 'bmi' in cohort_pl_t1.columns else None

    # Sex
    t1_pct_female = None
    t1_n_female = None
    if 'sex_category' in cohort_pl_t1.columns:
        t1_n_female = cohort_pl_t1.filter(pl.col('sex_category') == 'Female').shape[0]
        t1_pct_female = (t1_n_female / t1_total_patients * 100)

    # Race/Ethnicity breakdown
    t1_race_ethnicity_counts = {}
    if 'race_ethnicity' in cohort_pl_t1.columns:
        race_groups = cohort_pl_t1.group_by('race_ethnicity').agg(pl.count().alias('n'))
        for t1_row in race_groups.iter_rows(named=True):
            t1_race_ethnicity_counts[t1_row['race_ethnicity']] = {
                'n': t1_row['n'],
                'pct': (t1_row['n'] / t1_total_patients * 100)
            }

    # ICU LOS
    t1_mean_icu_los = cohort_pl_t1['icu_los_days'].mean()
    t1_sd_icu_los = cohort_pl_t1['icu_los_days'].std()

    # Hospital LOS
    t1_mean_hospital_los = cohort_pl_t1['hospital_los_days'].mean()
    t1_sd_hospital_los = cohort_pl_t1['hospital_los_days'].std()

    # Clinical characteristics
    t1_mean_highest_temp = cohort_pl_t1['highest_temperature'].mean()
    t1_sd_highest_temp = cohort_pl_t1['highest_temperature'].std()

    t1_mean_lowest_temp = cohort_pl_t1['lowest_temperature'].mean()
    t1_sd_lowest_temp = cohort_pl_t1['lowest_temperature'].std()

    t1_mean_lowest_map = cohort_pl_t1['lowest_map'].mean()
    t1_sd_lowest_map = cohort_pl_t1['lowest_map'].std()

    t1_mean_highest_sofa = cohort_pl_t1['sofa_total'].mean()
    t1_sd_highest_sofa = cohort_pl_t1['sofa_total'].std()

    # Comorbidity Scores
    t1_mean_elix_score = cohort_pl_t1['elix_score'].mean() if 'elix_score' in cohort_pl_t1.columns else None
    t1_sd_elix_score = cohort_pl_t1['elix_score'].std() if 'elix_score' in cohort_pl_t1.columns else None

    t1_mean_cci_score = cohort_pl_t1['cci_score'].mean() if 'cci_score' in cohort_pl_t1.columns else None
    t1_sd_cci_score = cohort_pl_t1['cci_score'].std() if 'cci_score' in cohort_pl_t1.columns else None

    t1_chronic_pulm_n = None
    t1_chronic_pulm_pct = None
    if 'chronic_pulmonary_disease' in cohort_pl_t1.columns:
        t1_chronic_pulm_n = cohort_pl_t1.filter(pl.col('chronic_pulmonary_disease') == 1).shape[0]
        t1_chronic_pulm_pct = (t1_chronic_pulm_n / t1_total_patients * 100)

    # Interventions (n, %)
    t1_vasopressor_ever_n = cohort_pl_t1['vasopressor_ever'].sum()
    t1_vasopressor_ever_pct = (t1_vasopressor_ever_n / t1_total_patients * 100)

    t1_nippv_ever_n = cohort_pl_t1['NIPPV_ever'].sum()
    t1_nippv_ever_pct = (t1_nippv_ever_n / t1_total_patients * 100)

    t1_hfno_ever_n = cohort_pl_t1['HFNO_ever'].sum()
    t1_hfno_ever_pct = (t1_hfno_ever_n / t1_total_patients * 100)

    t1_imv_ever_n = cohort_pl_t1['IMV_ever'].sum()
    t1_imv_ever_pct = (t1_imv_ever_n / t1_total_patients * 100)

    # Labs
    t1_mean_highest_wbc = cohort_pl_t1['highest_wbc'].mean()
    t1_sd_highest_wbc = cohort_pl_t1['highest_wbc'].std()

    t1_mean_highest_cr = cohort_pl_t1['highest_creatinine'].mean()
    t1_sd_highest_cr = cohort_pl_t1['highest_creatinine'].std()

    # Outcomes
    t1_inpatient_mortality_n = cohort_pl_t1['inpatient_mortality'].sum()
    t1_inpatient_mortality_pct = (t1_inpatient_mortality_n / t1_total_patients * 100)

    t1_icu_mortality_n = cohort_pl_t1['icu_mortality'].sum()
    t1_icu_mortality_pct = (t1_icu_mortality_n / t1_total_patients * 100)

    # ============================================================
    # ANTIBIOTIC METRICS
    # ============================================================

    # Top 15 antibiotics by DOT per 1000 PD
    t1_top_antibiotics = (
        dot_antibiotic_level
        .sort('dot_per_1000_pd', descending=True)
        .head(15)
        .select(['antibiotic', 'dot_per_1000_pd'])
    )

    # Overall DOT per 1000 PD
    t1_overall_dot_per_1000_pd = dot_cohort_level.filter(
        pl.col('metric') == 'Overall DOT per 1000 PD'
    )['value'][0]

    # Daily ASC scores (Days 0-10)
    t1_daily_asc_dict = {}
    for t1_row in daily_asc_summary.iter_rows(named=True):
        t1_day = int(t1_row['window_num'])
        t1_daily_asc_dict[f'day_{t1_day}_mean'] = float(t1_row['mean_asc'])
        t1_daily_asc_dict[f'day_{t1_day}_sd'] = float(t1_row['sd_asc'])

    # DASC per 1000 PD
    t1_dasc_per_1000_pd = dasc_overall.filter(
        pl.col('metric') == 'DASC per 1000 PD'
    )['value'][0]

    # AFD metrics
    t1_afd_mean_rate = afd_summary.filter(pl.col('metric') == 'mean_afd_rate')['value'][0]
    t1_afd_sd_rate = afd_summary.filter(pl.col('metric') == 'std_afd_rate')['value'][0]
    t1_afd_mean_pct = t1_afd_mean_rate * 100  # Convert to percentage

    # Total patient-days
    t1_total_patient_days = dot_cohort_level.filter(
        pl.col('metric') == 'Total PD'
    )['value'][0]

    # ============================================================
    # BUILD TABLE 1 STRUCTURE (JSON)
    # ============================================================

    t1_table1_data = {
        "site_id": site_name,
        "demographics": {
            "total_patients": int(t1_total_patients),
            "age_mean": float(t1_mean_age) if t1_mean_age is not None else "NOT_AVAILABLE",
            "age_sd": float(t1_sd_age) if t1_sd_age is not None else "NOT_AVAILABLE",
            "bmi_mean": float(t1_mean_bmi) if t1_mean_bmi is not None else "NOT_AVAILABLE",
            "bmi_sd": float(t1_sd_bmi) if t1_sd_bmi is not None else "NOT_AVAILABLE",
            "sex_female_n": int(t1_n_female) if t1_n_female is not None else "NOT_AVAILABLE",
            "sex_female_pct": float(t1_pct_female) if t1_pct_female is not None else "NOT_AVAILABLE",
            "race_ethnicity": t1_race_ethnicity_counts,
            "icu_los_mean_days": float(t1_mean_icu_los),
            "icu_los_sd_days": float(t1_sd_icu_los),
            "hospital_los_mean_days": float(t1_mean_hospital_los),
            "hospital_los_sd_days": float(t1_sd_hospital_los)
        },
        "clinical_characteristics": {
            "highest_temperature_mean": float(t1_mean_highest_temp),
            "highest_temperature_sd": float(t1_sd_highest_temp),
            "lowest_temperature_mean": float(t1_mean_lowest_temp),
            "lowest_temperature_sd": float(t1_sd_lowest_temp),
            "lowest_map_mean": float(t1_mean_lowest_map),
            "lowest_map_sd": float(t1_sd_lowest_map),
            "highest_sofa_mean": float(t1_mean_highest_sofa),
            "highest_sofa_sd": float(t1_sd_highest_sofa),
            "elixhauser_score_mean": float(t1_mean_elix_score) if t1_mean_elix_score is not None else "NOT_AVAILABLE",
            "elixhauser_score_sd": float(t1_sd_elix_score) if t1_sd_elix_score is not None else "NOT_AVAILABLE",
            "cci_score_mean": float(t1_mean_cci_score) if t1_mean_cci_score is not None else "NOT_AVAILABLE",
            "cci_score_sd": float(t1_sd_cci_score) if t1_sd_cci_score is not None else "NOT_AVAILABLE",
            "chronic_pulmonary_disease_n": int(t1_chronic_pulm_n) if t1_chronic_pulm_n is not None else "NOT_AVAILABLE",
            "chronic_pulmonary_disease_pct": float(t1_chronic_pulm_pct) if t1_chronic_pulm_pct is not None else "NOT_AVAILABLE",
            "vasopressor_ever_n": int(t1_vasopressor_ever_n),
            "vasopressor_ever_pct": float(t1_vasopressor_ever_pct),
            "nippv_ever_n": int(t1_nippv_ever_n),
            "nippv_ever_pct": float(t1_nippv_ever_pct),
            "hfno_ever_n": int(t1_hfno_ever_n),
            "hfno_ever_pct": float(t1_hfno_ever_pct),
            "cvvh_n": "NOT_AVAILABLE",
            "cvvh_pct": "NOT_AVAILABLE",
            "highest_wbc_mean": float(t1_mean_highest_wbc),
            "highest_wbc_sd": float(t1_sd_highest_wbc),
            "highest_creatinine_mean": float(t1_mean_highest_cr),
            "highest_creatinine_sd": float(t1_sd_highest_cr)
        },
        "antibiotic_metrics": {
            "top_antibiotics_dot_per_1000_pd": {
                t1_row['antibiotic']: float(t1_row['dot_per_1000_pd'])
                for t1_row in t1_top_antibiotics.iter_rows(named=True)
            },
            "overall_dot_per_1000_pd": float(t1_overall_dot_per_1000_pd),
            "daily_asc_scores": t1_daily_asc_dict,
            "dasc_per_1000_pd": float(t1_dasc_per_1000_pd),
            "afd_mean_rate": float(t1_afd_mean_rate),
            "afd_sd_rate": float(t1_afd_sd_rate),
            "afd_mean_pct_of_icu_days": float(t1_afd_mean_pct)
        },
        "outcomes": {
            "total_patient_days": int(t1_total_patient_days),
            "hospital_los_mean_days": float(t1_mean_hospital_los),
            "hospital_los_sd_days": float(t1_sd_hospital_los),
            "icu_los_mean_days": float(t1_mean_icu_los),
            "icu_los_sd_days": float(t1_sd_icu_los),
            "inpatient_mortality_n": int(t1_inpatient_mortality_n),
            "inpatient_mortality_pct": float(t1_inpatient_mortality_pct),
            "icu_mortality_n": int(t1_icu_mortality_n),
            "icu_mortality_pct": float(t1_icu_mortality_pct)
        },
        "missingness": {
            "age_at_admission": {
                "n_missing": int(age_n_missing),
                "pct_missing": float(age_pct_missing)
            },
            "sex_category": {
                "n_missing": int(sex_n_missing),
                "pct_missing": float(sex_pct_missing)
            },
            "race_ethnicity": {
                "n_missing": int(race_n_missing),
                "pct_missing": float(race_pct_missing)
            },
            "bmi": {
                "n_missing": int(bmi_n_missing),
                "pct_missing": float(bmi_pct_missing)
            },
            "highest_temperature": {
                "n_missing": int(highest_temp_n_missing),
                "pct_missing": float(highest_temp_pct_missing)
            },
            "lowest_temperature": {
                "n_missing": int(lowest_temp_n_missing),
                "pct_missing": float(lowest_temp_pct_missing)
            },
            "lowest_map": {
                "n_missing": int(lowest_map_n_missing),
                "pct_missing": float(lowest_map_pct_missing)
            },
            "sofa_total": {
                "n_missing": int(sofa_n_missing),
                "pct_missing": float(sofa_pct_missing)
            },
            "elix_score": {
                "n_missing": int(elix_score_n_missing),
                "pct_missing": float(elix_score_pct_missing)
            },
            "cci_score": {
                "n_missing": int(cci_score_n_missing),
                "pct_missing": float(cci_score_pct_missing)
            },
            "chronic_pulmonary_disease": {
                "n_missing": int(chronic_pulm_n_missing),
                "pct_missing": float(chronic_pulm_pct_missing)
            },
            "vasopressor_ever": {
                "n_missing": int(vasopressor_n_missing),
                "pct_missing": float(vasopressor_pct_missing)
            },
            "NIPPV_ever": {
                "n_missing": int(nippv_n_missing),
                "pct_missing": float(nippv_pct_missing)
            },
            "HFNO_ever": {
                "n_missing": int(hfno_n_missing),
                "pct_missing": float(hfno_pct_missing)
            },
            "IMV_ever": {
                "n_missing": int(imv_n_missing),
                "pct_missing": float(imv_pct_missing)
            },
            "cvvh": {
                "n_missing": int(cvvh_n_missing),
                "pct_missing": float(cvvh_pct_missing)
            },
            "highest_wbc": {
                "n_missing": int(wbc_n_missing),
                "pct_missing": float(wbc_pct_missing)
            },
            "highest_creatinine": {
                "n_missing": int(creatinine_n_missing),
                "pct_missing": float(creatinine_pct_missing)
            },
            "hospital_los_days": {
                "n_missing": int(hospital_los_n_missing),
                "pct_missing": float(hospital_los_pct_missing)
            },
            "icu_los_days": {
                "n_missing": int(icu_los_n_missing),
                "pct_missing": float(icu_los_pct_missing)
            },
            "inpatient_mortality": {
                "n_missing": int(inpatient_mortality_n_missing),
                "pct_missing": float(inpatient_mortality_pct_missing)
            },
            "icu_mortality": {
                "n_missing": int(icu_mortality_n_missing),
                "pct_missing": float(icu_mortality_pct_missing)
            }
        }
    }

    # ============================================================
    # SAVE AS JSON (for aggregation)
    # ============================================================

    t1_json_path = Path('RESULTS_UPLOAD_ME') / 'table1_summary.json'
    with open(t1_json_path, 'w') as json_file:
        json.dump(t1_table1_data, json_file, indent=2)

    print(f"\n✓ Saved JSON: {t1_json_path}")

    # ============================================================
    # BUILD TABLE 1 AS CSV (for viewing)
    # ============================================================

    t1_table1_rows = []

    # Demographics section
    t1_table1_rows.append({'Category': 'DEMOGRAPHICS', 'Variable': '', 'Value': '', 'n_missing': '', 'Notes': ''})
    t1_table1_rows.append({'Category': 'Demographics', 'Variable': 'N (total patients)', 'Value': str(t1_total_patients), 'n_missing': '', 'Notes': ''})

    if t1_mean_age is not None:
        t1_table1_rows.append({'Category': 'Demographics', 'Variable': 'Age (mean, SD)',
                           'Value': f"{t1_mean_age:.1f} ± {t1_sd_age:.1f}", 'n_missing': str(age_n_missing), 'Notes': ''})
    else:
        t1_table1_rows.append({'Category': 'Demographics', 'Variable': 'Age (mean, SD)',
                           'Value': 'NOT AVAILABLE', 'n_missing': str(age_n_missing), 'Notes': ''})

    if t1_n_female is not None:
        t1_table1_rows.append({'Category': 'Demographics', 'Variable': 'Sex, female (n, %)',
                           'Value': f"{t1_n_female} ({t1_pct_female:.1f}%)", 'n_missing': str(sex_n_missing), 'Notes': ''})

    # BMI
    if t1_mean_bmi is not None:
        t1_table1_rows.append({'Category': 'Demographics', 'Variable': 'BMI (first value in hospitalization, mean SD)',
                           'Value': f"{t1_mean_bmi:.1f} ± {t1_sd_bmi:.1f}", 'n_missing': str(bmi_n_missing), 'Notes': ''})
    else:
        t1_table1_rows.append({'Category': 'Demographics', 'Variable': 'BMI (first value in hospitalization, mean SD)',
                           'Value': 'NOT AVAILABLE', 'n_missing': str(bmi_n_missing), 'Notes': ''})

    # Race/Ethnicity
    if t1_race_ethnicity_counts:
        t1_table1_rows.append({'Category': 'Demographics', 'Variable': 'Race ethnicity, n %', 'Value': '', 'n_missing': str(race_n_missing), 'Notes': ''})
        for t1_race_cat in ['Hispanic', 'Non-Hispanic White', 'Non-Hispanic Black', 'Non-Hispanic Asian', 'Other', 'Not Reported']:
            if t1_race_cat in t1_race_ethnicity_counts:
                t1_n = t1_race_ethnicity_counts[t1_race_cat]['n']
                t1_pct = t1_race_ethnicity_counts[t1_race_cat]['pct']
                t1_table1_rows.append({'Category': 'Demographics', 'Variable': f'  {t1_race_cat}',
                                   'Value': f"{t1_n} ({t1_pct:.1f}%)", 'n_missing': '', 'Notes': ''})
            else:
                t1_table1_rows.append({'Category': 'Demographics', 'Variable': f'  {t1_race_cat}',
                                   'Value': '0 (0.0%)', 'n_missing': '', 'Notes': ''})

    # Clinical characteristics section
    t1_table1_rows.append({'Category': '', 'Variable': '', 'Value': '', 'n_missing': '', 'Notes': ''})
    t1_table1_rows.append({'Category': 'CLINICAL CHARACTERISTICS', 'Variable': '', 'Value': '', 'n_missing': '', 'Notes': ''})

    t1_table1_rows.append({'Category': 'Vitals', 'Variable': 'Highest temperature (mean, SD)',
                       'Value': f"{t1_mean_highest_temp:.1f} ± {t1_sd_highest_temp:.1f}", 'n_missing': str(highest_temp_n_missing), 'Notes': ''})
    t1_table1_rows.append({'Category': 'Vitals', 'Variable': 'Lowest temperature (mean, SD)',
                       'Value': f"{t1_mean_lowest_temp:.1f} ± {t1_sd_lowest_temp:.1f}", 'n_missing': str(lowest_temp_n_missing), 'Notes': ''})
    t1_table1_rows.append({'Category': 'Vitals', 'Variable': 'Lowest mean arterial pressure (mean, SD)',
                       'Value': f"{t1_mean_lowest_map:.1f} ± {t1_sd_lowest_map:.1f}", 'n_missing': str(lowest_map_n_missing), 'Notes': ''})

    t1_table1_rows.append({'Category': 'Severity', 'Variable': 'Highest SOFA score in 1st 24 hours (mean, SD)',
                       'Value': f"{t1_mean_highest_sofa:.1f} ± {t1_sd_highest_sofa:.1f}", 'n_missing': str(sofa_n_missing), 'Notes': ''})

    # Comorbidity Scores
    if t1_mean_elix_score is not None:
        t1_table1_rows.append({'Category': 'Comorbidity', 'Variable': 'Elixhauser Score (mean, SD)',
                           'Value': f"{t1_mean_elix_score:.1f} ± {t1_sd_elix_score:.1f}", 'n_missing': str(elix_score_n_missing), 'Notes': ''})
    else:
        t1_table1_rows.append({'Category': 'Comorbidity', 'Variable': 'Elixhauser Score (mean, SD)',
                           'Value': 'NOT AVAILABLE', 'n_missing': str(elix_score_n_missing), 'Notes': ''})

    if t1_mean_cci_score is not None:
        t1_table1_rows.append({'Category': 'Comorbidity', 'Variable': 'CCI Score (mean, SD)',
                           'Value': f"{t1_mean_cci_score:.1f} ± {t1_sd_cci_score:.1f}", 'n_missing': str(cci_score_n_missing), 'Notes': ''})
    else:
        t1_table1_rows.append({'Category': 'Comorbidity', 'Variable': 'CCI Score (mean, SD)',
                           'Value': 'NOT AVAILABLE', 'n_missing': str(cci_score_n_missing), 'Notes': ''})

    if t1_chronic_pulm_n is not None:
        t1_table1_rows.append({'Category': 'Comorbidity', 'Variable': 'Chronic Pulmonary Disease (n, %)',
                           'Value': f"{t1_chronic_pulm_n} ({t1_chronic_pulm_pct:.1f}%)", 'n_missing': str(chronic_pulm_n_missing), 'Notes': ''})
    else:
        t1_table1_rows.append({'Category': 'Comorbidity', 'Variable': 'Chronic Pulmonary Disease (n, %)',
                           'Value': 'NOT AVAILABLE', 'n_missing': str(chronic_pulm_n_missing), 'Notes': ''})

    t1_table1_rows.append({'Category': 'Interventions', 'Variable': 'Vasopressor_ever (n, %)',
                       'Value': f"{t1_vasopressor_ever_n} ({t1_vasopressor_ever_pct:.1f}%)", 'n_missing': str(vasopressor_n_missing), 'Notes': ''})
    t1_table1_rows.append({'Category': 'Interventions', 'Variable': 'NIPPV_ever (n, %)',
                       'Value': f"{t1_nippv_ever_n} ({t1_nippv_ever_pct:.1f}%)", 'n_missing': str(nippv_n_missing), 'Notes': ''})
    t1_table1_rows.append({'Category': 'Interventions', 'Variable': 'HFNO_ever (n, %)',
                       'Value': f"{t1_hfno_ever_n} ({t1_hfno_ever_pct:.1f}%)", 'n_missing': str(hfno_n_missing), 'Notes': ''})
    t1_table1_rows.append({'Category': 'Interventions', 'Variable': 'IMV_ever (n, %)',
                       'Value': f"{t1_imv_ever_n} ({t1_imv_ever_pct:.1f}%)", 'n_missing': str(imv_n_missing), 'Notes': ''})
    t1_table1_rows.append({'Category': 'Interventions', 'Variable': 'CVVH (non missing and > 0 blood_flow) (n, %)',
                       'Value': 'NOT AVAILABLE', 'n_missing': str(cvvh_n_missing), 'Notes': ''})

    t1_table1_rows.append({'Category': 'Labs', 'Variable': 'Highest WBC (mean, sd)',
                       'Value': f"{t1_mean_highest_wbc:.1f} ± {t1_sd_highest_wbc:.1f}", 'n_missing': str(wbc_n_missing), 'Notes': ''})
    t1_table1_rows.append({'Category': 'Labs', 'Variable': 'Highest_Cr (mean, sd)',
                       'Value': f"{t1_mean_highest_cr:.2f} ± {t1_sd_highest_cr:.2f}", 'n_missing': str(creatinine_n_missing), 'Notes': ''})

    # Antibiotic metrics section
    t1_table1_rows.append({'Category': '', 'Variable': '', 'Value': '', 'n_missing': '', 'Notes': ''})
    t1_table1_rows.append({'Category': 'ANTIBIOTIC METRICS', 'Variable': '', 'Value': '', 'n_missing': '', 'Notes': ''})

    t1_table1_rows.append({'Category': 'Antibiotics', 'Variable': 'Common antibiotics prescribed (DOT per 1000 PD)',
                       'Value': '', 'n_missing': '', 'Notes': 'Top 15'})

    for t1_row in t1_top_antibiotics.iter_rows(named=True):
        t1_table1_rows.append({'Category': 'Antibiotics',
                           'Variable': f"  {t1_row['antibiotic']}",
                           'Value': f"{t1_row['dot_per_1000_pd']:.2f}",
                           'n_missing': '',
                           'Notes': ''})

    t1_table1_rows.append({'Category': 'Antibiotics', 'Variable': 'DOT per 1000 patient days (all antibiotics in excel)',
                       'Value': f"{t1_overall_dot_per_1000_pd:.2f}", 'n_missing': '', 'Notes': ''})

    t1_table1_rows.append({'Category': 'ASC Scores', 'Variable': 'Antibiotic Spectrum Scores (mean, SD per day)',
                       'Value': '', 'n_missing': '', 'Notes': ''})

    for t1_day in range(11):  # Days 0-10
        t1_mean_key = f'day_{t1_day}_mean'
        t1_sd_key = f'day_{t1_day}_sd'
        if t1_mean_key in t1_daily_asc_dict:
            t1_table1_rows.append({'Category': 'ASC Scores',
                               'Variable': f"  Day {t1_day} ASC score",
                               'Value': f"{t1_daily_asc_dict[t1_mean_key]:.2f} ± {t1_daily_asc_dict[t1_sd_key]:.2f}",
                               'n_missing': '',
                               'Notes': ''})

    t1_table1_rows.append({'Category': 'ASC Scores', 'Variable': 'DASC (days of ASC) per 1000 PD',
                       'Value': f"{t1_dasc_per_1000_pd:.2f}", 'n_missing': '', 'Notes': ''})

    t1_table1_rows.append({'Category': 'AFD', 'Variable': 'Antibiotic-Free Days (mean % of ICU days)',
                       'Value': f"{t1_afd_mean_pct:.1f}%", 'n_missing': '', 'Notes': ''})
    t1_table1_rows.append({'Category': 'AFD', 'Variable': 'AFD rate (mean ± SD)',
                       'Value': f"{t1_afd_mean_rate:.3f} ± {t1_afd_sd_rate:.3f}", 'n_missing': '', 'Notes': ''})

    # Outcomes section
    t1_table1_rows.append({'Category': '', 'Variable': '', 'Value': '', 'n_missing': '', 'Notes': ''})
    t1_table1_rows.append({'Category': 'OUTCOMES', 'Variable': '', 'Value': '', 'n_missing': '', 'Notes': ''})
    t1_table1_rows.append({'Category': 'Outcomes', 'Variable': 'Total patient-days',
                       'Value': f"{int(t1_total_patient_days):,}", 'n_missing': '', 'Notes': ''})
    t1_table1_rows.append({'Category': 'Outcomes', 'Variable': 'Hospital LOS (mean, SD) days',
                       'Value': f"{t1_mean_hospital_los:.1f} ± {t1_sd_hospital_los:.1f}", 'n_missing': str(hospital_los_n_missing), 'Notes': ''})
    t1_table1_rows.append({'Category': 'Outcomes', 'Variable': 'ICU LOS of index (mean, SD) days',
                       'Value': f"{t1_mean_icu_los:.1f} ± {t1_sd_icu_los:.1f}", 'n_missing': str(icu_los_n_missing), 'Notes': ''})
    t1_table1_rows.append({'Category': 'Outcomes', 'Variable': 'Inpatient mortality (n, %)',
                       'Value': f"{t1_inpatient_mortality_n} ({t1_inpatient_mortality_pct:.1f}%)", 'n_missing': str(inpatient_mortality_n_missing), 'Notes': ''})
    t1_table1_rows.append({'Category': 'Outcomes', 'Variable': 'ICU mortality (n, %)',
                       'Value': f"{t1_icu_mortality_n} ({t1_icu_mortality_pct:.1f}%)", 'n_missing': str(icu_mortality_n_missing), 'Notes': ''})

    # Convert to DataFrame and save
    t1_table1_df = pl.DataFrame(t1_table1_rows)

    t1_csv_path = Path('RESULTS_UPLOAD_ME') / 'table1_summary.csv'
    t1_table1_df.write_csv(t1_csv_path)

    print(f"✓ Saved CSV: {t1_csv_path}")
    print(f"\n=== Table 1 Summary ===")
    print(f"Total patients: {t1_total_patients:,}")
    print(f"Total patient-days: {int(t1_total_patient_days):,}")
    print(f"Top antibiotic: {t1_top_antibiotics['antibiotic'][0]} (DOT per 1000 PD: {t1_top_antibiotics['dot_per_1000_pd'][0]:.2f})")
    print(f"\n✓ Table 1 generation complete!")
    return


if __name__ == "__main__":
    app.run()
