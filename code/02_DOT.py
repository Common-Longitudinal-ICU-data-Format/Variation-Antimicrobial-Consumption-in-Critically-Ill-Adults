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
    mo.md(r"""## Save Results""")
    return


@app.cell
def _(Path, dot_hospital_level):
    # Save hospital-level DOT table (wide format with PD column)
    print("Saving hospital-level DOT table...")

    # Ensure output directory exists
    Path('PHI_DATA').mkdir(exist_ok=True)

    # Save pivoted hospital-level table
    output_path = Path('PHI_DATA') / 'dot_hospital_level.parquet'
    dot_hospital_level.write_parquet(output_path)

    print(f"✓ Saved: {output_path}")
    print(f"  Shape: {dot_hospital_level.shape}")
    print(f"\n✓ Hospital-level table saved successfully!")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
