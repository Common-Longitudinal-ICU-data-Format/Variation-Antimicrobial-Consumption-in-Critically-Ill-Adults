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


@app.cell
def _(mo):
    mo.md(r"""## Setup and Configuration""")
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from clifpy.tables import MedicationAdminIntermittent
    from pathlib import Path
    import warnings
    warnings.filterwarnings('ignore')

    print("=== Days of Therapy (DOT) Analysis ===")
    print("Setting up environment...")
    return MedicationAdminIntermittent, Path, pd


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
def _(mo):
    mo.md(r"""## Filter to ICU Stay Window""")
    return


@app.cell
def _(cohort_df, meds_intermittent_df, pd):
    # Filter medication administrations to ICU stay windows
    print("Filtering medications to ICU stay windows...")

    # Merge with cohort to get start_dttm and end_dttm
    meds_with_windows = pd.merge(
        meds_intermittent_df,
        cohort_df[['hospitalization_id', 'start_dttm', 'end_dttm']],
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


@app.cell
def _(mo):
    mo.md(r"""## Convert to Polars for Performance""")
    return


@app.cell
def _():
    import polars as pl
    from tqdm import tqdm
    from datetime import timedelta

    print("Converting to Polars for optimized processing...")
    return pl, timedelta, tqdm


@app.cell
def _(cohort_df, meds_icu_window, pl):
    # Convert to Polars
    print("Converting DataFrames to Polars...")

    # Cohort
    cohort_pl = pl.DataFrame(cohort_df)

    # Medications
    meds_pl = pl.DataFrame(meds_icu_window)

    print(f"✓ Converted to Polars")
    print(f"  Cohort shape: {cohort_pl.shape}")
    print(f"  Medications shape: {meds_pl.shape}")
    return cohort_pl, meds_pl


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Understanding DOT Calculation

    ### Key Concept: 24-Hour Windows
    We divide each ICU stay into consecutive 24-hour windows starting from `start_dttm`:
    - Window 1: Day 0 (00:00 - 23:59)
    - Window 2: Day 1 (00:00 - 23:59)
    - ...
    - Last Window: Capped at `end_dttm` (may be < 24 hours)

    ### DOT Counting Rules
    For each antibiotic in each 24-hour window:
    - If ≥1 dose administered (non-null `med_dose`) → DOT = 1
    - If 0 doses → DOT = 0

    ### Example
    Patient receives antibiotics over 3 days:
    - Day 0: Vancomycin (2 doses), Piperacillin-Tazobactam (1 dose)
    - Day 1: Vancomycin (1 dose)
    - Day 2: No antibiotics

    **Result:**
    - Vancomycin DOT = 2 (Days 0, 1)
    - Piperacillin-Tazobactam DOT = 1 (Day 0)
    - Antibiotic-Free Days = 1 (Day 2)
    """
    )
    return


@app.cell
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


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Calculate DOT per Antibiotic

    ### Overview
    This section calculates **Days of Therapy (DOT)** for each antibiotic for each hospitalization by checking every 24-hour window to see if the antibiotic was administered.

    ### Algorithm Structure
    This uses a **triple-nested loop** approach:
    ```
    For each hospitalization:
        For each 24-hour window in that hospitalization:
            For each antibiotic in the reference table:
                Check if ≥1 dose with non-null med_dose exists in this window
                If yes → increment DOT counter for that antibiotic
    ```

    ### Inputs
    - **`windows_pl`**: All 24-hour windows across all hospitalizations
    - **`meds_pl`**: All antibiotic administrations (filtered to ICU windows)
    - **`abx_spectrum`**: Reference table of all antibiotics to track

    ### Processing Strategy
    1. **Group windows by hospitalization** for efficient filtering
    2. **Filter medications per hospitalization** to reduce search space
    3. **Check each window-antibiotic combination** for doses
    4. **Track antibiotic-free days** (windows with zero antibiotics)

    ### Output: Patient-Level DOT Table
    Long-format table with three columns:
    - `hospitalization_id`: Patient identifier
    - `antibiotic`: Antibiotic name (or 'ANTIBIOTIC_FREE')
    - `dot`: Number of days that antibiotic was given

    **Example output:**
    ```
    hospitalization_id | antibiotic              | dot
    -------------------|-------------------------|----
    12345             | Vancomycin              | 5
    12345             | Piperacillin-Tazobactam | 3
    12345             | ANTIBIOTIC_FREE         | 2
    67890             | Ceftriaxone             | 1
    ```
     The code counts number of windows as Patient-Days:
      - 7-hour stay → 1 PD (should be 0.29 PD if fractional)
      - 23-hour stay → 1 PD (should be 0.96 PD if fractional)
      - 25-hour stay → 2 PD (26-hour window + 23-hour window)

    ### Performance Note
    This is the **most computationally intensive section** due to:
    - Nested loops over hospitalizations × windows × antibiotics
    - Repeated filtering operations
    - Large datasets (potentially thousands of hospitalizations × hundreds of antibiotics)

    **Progress tracking:** `tqdm` provides progress bars to monitor completion.
    """
    )
    return


@app.cell
def _(abx_spectrum, meds_pl, pl, tqdm, windows_pl):
    # Calculate DOT for each antibiotic per hospitalization
    print("Calculating DOT for each antibiotic...")

    # Get unique antibiotics from the spectrum scoring
    all_antibiotics = abx_spectrum['med_category'].str.strip().tolist()

    # Initialize DOT results
    dot_results = []

    # Group windows by hospitalization for faster processing
    hosp_windows = windows_pl.group_by('hospitalization_id').agg([
        pl.col('window_num'),
        pl.col('window_start'),
        pl.col('window_end')
    ])

    print(f"Processing {len(hosp_windows):,} hospitalizations...")

    for hosp_row in tqdm(hosp_windows.iter_rows(named=True), total=len(hosp_windows), desc="Calculating DOT"):
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

        # Process each window
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

            # Check each antibiotic
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

        # Store results for this hospitalization
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

    # Create DOT DataFrame (long format)
    dot_long_pl = pl.DataFrame(dot_results)

    print(f"✓ DOT calculation complete")
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
        # Reorder columns: hospitalization_id, PD, then all antibiotics
        .select([
            'hospitalization_id',
            'PD'
        ] + [col for col in dot_wide_pl.columns if col != 'hospitalization_id'])
    )

    print(f"✓ Hospital-level table created")
    print(f"  Shape: {dot_hospital_level.shape}")
    print(f"  Columns: {len(dot_hospital_level.columns)} (hospitalization_id, PD, + {len(dot_hospital_level.columns)-2} antibiotics)")
    return (dot_hospital_level,)


@app.cell
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

    # Get antibiotic columns (exclude hospitalization_id, PD, and ANTIBIOTIC_FREE)
    antibiotic_cols = [
        col for col in dot_hospital_level.columns
        if col not in ['hospitalization_id', 'PD', 'ANTIBIOTIC_FREE']
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
            'hospitalizations_with_antibiotic': hospitalizations_with_abx,
            'percent_hospitalizations': percent_hospitalizations
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


@app.cell
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
def _(cohort_df, dot_hospital_level, pl):
    # Calculate cohort-level and location-type-level overall DOT metrics
    print("Calculating overall DOT per 1000 Patient-Days...")

    # Merge location_type from cohort into dot_hospital_level
    print("\nMerging location_type from cohort...")
    dot_with_location = dot_hospital_level.join(
        pl.DataFrame(cohort_df[['hospitalization_id', 'location_type']]),
        on='hospitalization_id',
        how='left'
    )
    print(f"✓ Location type merged")

    # Get antibiotic columns (exclude hospitalization_id, PD, ANTIBIOTIC_FREE, location_type)
    antibiotic_cols_for_total = [
        col for col in dot_with_location.columns
        if col not in ['hospitalization_id', 'PD', 'ANTIBIOTIC_FREE', 'location_type']
    ]

    print(f"  Antibiotics for total DOT calculation: {len(antibiotic_cols_for_total)}")

    # Calculate total DOT per hospitalization (sum across all antibiotics)
    print("\nCalculating total DOT per hospitalization...")
    dot_with_total = dot_with_location.with_columns(
        pl.sum_horizontal(antibiotic_cols_for_total).alias('total_dot')
    )

    # === COHORT-LEVEL OVERALL METRICS ===
    print("\n=== Cohort-Level Overall Metrics ===")

    total_dot_cohort = dot_with_total['total_dot'].sum()
    total_pd_cohort = dot_with_total['PD'].sum()
    overall_dot_per_1000_pd_cohort = (total_dot_cohort / total_pd_cohort) * 1000

    print(f"Total DOT (all antibiotics, all hospitalizations): {total_dot_cohort:,}")
    print(f"Total PD (all hospitalizations): {total_pd_cohort:,}")
    print(f"Overall DOT per 1000 PD: {overall_dot_per_1000_pd_cohort:.2f}")

    # Create cohort-level DataFrame
    dot_cohort_level = pl.DataFrame({
        'metric': ['Total DOT', 'Total PD', 'Overall DOT per 1000 PD'],
        'value': [float(total_dot_cohort), float(total_pd_cohort), overall_dot_per_1000_pd_cohort]
    })

    # === LOCATION-TYPE-LEVEL METRICS ===
    print("\n=== Location-Type-Level Metrics ===")

    # Group by location_type and calculate metrics
    dot_location_type_level = (
        dot_with_total
        .group_by('location_type')
        .agg([
            pl.col('total_dot').sum().alias('total_dot'),
            pl.col('PD').sum().alias('total_pd'),
            pl.col('hospitalization_id').count().alias('hospitalizations')
        ])
        .with_columns(
            ((pl.col('total_dot') / pl.col('total_pd')) * 1000).alias('dot_per_1000_pd')
        )
        .sort('dot_per_1000_pd', descending=True)
    )

    print(f"Location types analyzed: {len(dot_location_type_level)}")
    print(f"\n=== DOT per 1000 PD by Location Type ===")
    print(dot_location_type_level.to_pandas().to_string(index=False))

    print(f"\n✓ Cohort-level and location-type-level metrics calculated")
    return dot_cohort_level, dot_location_type_level


@app.cell
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
def _(abx_spectrum, meds_pl, pl, tqdm, windows_pl):
    # Calculate daily ASC per window (0-10) for each hospitalization
    print("Calculating Daily Antibiotic Spectrum Coverage (ASC) per window...")

    # Create spectrum score lookup dictionary
    spectrum_lookup = dict(zip(
        abx_spectrum['med_category'].str.strip(),
        abx_spectrum['spectrum_score']
    ))

    print(f"  Spectrum scores loaded for {len(spectrum_lookup)} antibiotics")

    # Initialize results list
    daily_asc_results = []

    # Group windows by hospitalization (ALL windows - needed for DASC calculation)
    print("\nProcessing ALL windows for ASC calculation...")
    hosp_windows_asc = windows_pl.group_by('hospitalization_id').agg([
        pl.col('window_num'),
        pl.col('window_start'),
        pl.col('window_end')
    ])

    print(f"Processing {len(hosp_windows_asc):,} hospitalizations for ASC calculation...")

    # Loop through each hospitalization
    for hosp_row in tqdm(hosp_windows_asc.iter_rows(named=True), total=len(hosp_windows_asc), desc="Calculating Daily ASC"):
        h_id = hosp_row['hospitalization_id']

        # Get all medications for this hospitalization
        h_meds = meds_pl.filter(pl.col('hospitalization_id') == h_id)

        # Get windows for this hospitalization
        h_window_nums = hosp_row['window_num']
        h_window_starts = hosp_row['window_start']
        h_window_ends = hosp_row['window_end']

        # Process each window (0-10)
        for w_idx in range(len(h_window_nums)):
            w_num = h_window_nums[w_idx]
            w_start = h_window_starts[w_idx]
            w_end = h_window_ends[w_idx]

            # Filter medications in this window
            w_meds = h_meds.filter(
                (pl.col('admin_dttm') >= w_start) &
                (pl.col('admin_dttm') <= w_end) &
                (pl.col('med_dose').is_not_null())  # Only count non-null doses
            )

            # Get unique antibiotics used in this window
            antibiotics_in_window = w_meds['med_category'].unique().to_list()

            # Sum spectrum scores for all antibiotics used in this window
            daily_asc = sum(
                spectrum_lookup.get(abx.strip(), 0)
                for abx in antibiotics_in_window
            )

            # Store result
            daily_asc_results.append({
                'hospitalization_id': h_id,
                'window_num': w_num,
                'daily_asc': daily_asc
            })

    # Create Polars DataFrame
    daily_asc_patient_level = pl.DataFrame(daily_asc_results)

    print(f"\n✓ Daily ASC calculated")
    print(f"  Total records: {len(daily_asc_patient_level):,}")
    print(f"  Unique hospitalizations: {daily_asc_patient_level['hospitalization_id'].n_unique():,}")

    # Calculate summary statistics per window (for sharing across sites)
    # Filter to windows 0-10 for summary/plotting
    print("\nCalculating summary statistics per window (Days 0-10 for plotting)...")
    daily_asc_summary = (
        daily_asc_patient_level
        .filter(pl.col('window_num') <= 10)  # Only windows 0-10 for summary
        .group_by('window_num')
        .agg([
            pl.col('daily_asc').mean().alias('mean_asc'),
            pl.col('daily_asc').std().alias('sd_asc'),
            pl.col('daily_asc').median().alias('median_asc'),
            pl.col('daily_asc').min().alias('min_asc'),
            pl.col('daily_asc').max().alias('max_asc'),
            pl.col('hospitalization_id').count().alias('n_hospitalizations')
        ])
        .sort('window_num')
    )

    print(f"\n=== Daily ASC Summary (ICU Days 0-10) ===")
    print(daily_asc_summary.to_pandas().to_string(index=False))

    print(f"\n✓ Daily ASC summary calculated")
    return daily_asc_patient_level, daily_asc_summary


@app.cell
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
def _(cohort_df, daily_asc_patient_level, dot_hospital_level, pl):
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
        'value': [total_dasc_overall, float(total_pd_overall), dasc_per_1000_pd_overall]
    })

    # === DASC METRICS BY YEAR ===
    print("\n=== DASC Metrics by Year ===")

    # Extract year from cohort start_dttm
    cohort_with_year = pl.DataFrame(cohort_df).select([
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
            ((pl.col('total_dasc') / pl.col('total_pd')) * 1000).alias('dasc_per_1000_pd')
        )
        .sort('year')
    )

    print(dasc_by_year.to_pandas().to_string(index=False))

    print(f"\n✓ DASC per 1000 PD calculated")
    return dasc_by_year, dasc_overall


@app.cell
def _(mo):
    mo.md(r"""## Save Results""")
    return


@app.cell
def _(
    Path,
    daily_asc_patient_level,
    daily_asc_summary,
    dasc_by_year,
    dasc_overall,
    dot_antibiotic_level,
    dot_cohort_level,
    dot_hospital_level,
    dot_location_type_level,
):
    # Save results
    print("Saving results...")

    # Ensure output directory exists
    Path('PHI_DATA').mkdir(exist_ok=True)

    # Save hospital-level DOT table (wide format with PD column)
    print("\n1. Hospital-level DOT table (wide format):")
    hosp_output_path = Path('PHI_DATA') / 'dot_hospital_level.parquet'
    dot_hospital_level.write_parquet(hosp_output_path)
    print(f"   ✓ Saved: {hosp_output_path}")
    print(f"   Shape: {dot_hospital_level.shape}")

    # Save antibiotic-level metrics
    print("\n2. Antibiotic-level metrics:")
    abx_output_path = Path('PHI_DATA') / 'dot_antibiotic_level.parquet'
    dot_antibiotic_level.write_parquet(abx_output_path)
    print(f"   ✓ Saved: {abx_output_path}")
    print(f"   Shape: {dot_antibiotic_level.shape}")

    # Save cohort-level overall metrics
    print("\n3. Cohort-level overall metrics:")
    cohort_output_path = Path('PHI_DATA') / 'dot_cohort_level.parquet'
    dot_cohort_level.write_parquet(cohort_output_path)
    print(f"   ✓ Saved: {cohort_output_path}")
    print(f"   Shape: {dot_cohort_level.shape}")

    # Save location-type-level metrics
    print("\n4. Location-type-level metrics:")
    location_output_path = Path('PHI_DATA') / 'dot_location_type_level.parquet'
    dot_location_type_level.write_parquet(location_output_path)
    print(f"   ✓ Saved: {location_output_path}")
    print(f"   Shape: {dot_location_type_level.shape}")

    # Save daily ASC patient-level data
    print("\n5. Daily ASC patient-level (all windows):")
    daily_asc_patient_path = Path('PHI_DATA') / 'daily_asc_patient_level.parquet'
    daily_asc_patient_level.write_parquet(daily_asc_patient_path)
    print(f"   ✓ Saved: {daily_asc_patient_path}")
    print(f"   Shape: {daily_asc_patient_level.shape}")

    # Save daily ASC summary (for sharing)
    print("\n6. Daily ASC summary (windows 0-10, for sharing):")
    daily_asc_summary_path = Path('PHI_DATA') / 'daily_asc_summary.parquet'
    daily_asc_summary.write_parquet(daily_asc_summary_path)
    print(f"   ✓ Saved: {daily_asc_summary_path}")
    print(f"   Shape: {daily_asc_summary.shape}")

    # Save DASC overall metrics
    print("\n7. DASC overall metrics:")
    dasc_overall_path = Path('PHI_DATA') / 'dasc_overall.parquet'
    dasc_overall.write_parquet(dasc_overall_path)
    print(f"   ✓ Saved: {dasc_overall_path}")
    print(f"   Shape: {dasc_overall.shape}")

    # Save DASC by year metrics
    print("\n8. DASC by year metrics:")
    dasc_by_year_path = Path('PHI_DATA') / 'dasc_by_year.parquet'
    dasc_by_year.write_parquet(dasc_by_year_path)
    print(f"   ✓ Saved: {dasc_by_year_path}")
    print(f"   Shape: {dasc_by_year.shape}")

    print(f"\n✓ All results saved successfully!")
    print(f"\n=== Summary of Output Files ===")
    print(f"DOT Metrics:")
    print(f"  - dot_hospital_level.parquet (hospitalization-level, wide format)")
    print(f"  - dot_antibiotic_level.parquet (antibiotic-level summary)")
    print(f"  - dot_cohort_level.parquet (overall cohort metrics)")
    print(f"  - dot_location_type_level.parquet (by ICU location type)")
    print(f"\nASC/DASC Metrics:")
    print(f"  - daily_asc_patient_level.parquet (daily ASC per hospitalization per window)")
    print(f"  - daily_asc_summary.parquet (mean/SD per window 0-10, for sharing)")
    print(f"  - dasc_overall.parquet (overall DASC metrics)")
    print(f"  - dasc_by_year.parquet (DASC metrics by year)")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
