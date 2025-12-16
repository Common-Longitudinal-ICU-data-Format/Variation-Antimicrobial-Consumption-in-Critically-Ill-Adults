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

    ## Inclusion Criteria
    - Adults (≥18 years)
    - First ICU admission (`location_category == 'icu'`) - ALL ICU types
    - Years: 2018-2024
    - Admitted to ICU via the ED
    - Admitted to academic or community hospital (excluding LTACH)

    ## Exclusion Criteria
    - Prior hospitalization within 48 hours (ED admission within 48h of prior discharge)
    - Immunocompromised hosts (neutrophils_absolute < 500 at any timepoint)
    - Hospital LOS < 96 hours (4 days)
    - Died within 6 hours of ICU arrival

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
    - Units: WBC in 10^3/uL, Creatinine in mg/dL

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
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    import json
    import os
    from pathlib import Path
    from clifpy.tables import Adt, Hospitalization, Patient, MedicationAdminContinuous, Labs, Vitals, RespiratorySupport, CrrtTherapy, MicrobiologyNonculture, HospitalDiagnosis
    from clifpy.clif_orchestrator import ClifOrchestrator
    from clifpy.utils.outlier_handler import apply_outlier_handling
    import warnings
    warnings.filterwarnings('ignore')

    print("=== FLAME-ICU: Cohort Generation ===")
    print("Setting up environment...")
    return (
        Adt,
        ClifOrchestrator,
        CrrtTherapy,
        HospitalDiagnosis,
        Hospitalization,
        Labs,
        MedicationAdminContinuous,
        MicrobiologyNonculture,
        Path,
        Patient,
        RespiratorySupport,
        Vitals,
        apply_outlier_handling,
        json,
        os,
        pd,
        plt,
    )


@app.cell
def _(mo):
    mo.md(r"""## Load Data""")
    return


@app.cell
def _(Adt, Hospitalization, Patient):
    # Load required tables using clifpy config file
    print("Loading required tables...")

    # Load ADT data (include hospital_type and hospital_id for new criteria)
    adt_table = Adt.from_file(config_path='clif_config.json')
    adt_df = adt_table.df.copy()
    print(f"ADT data loaded: {len(adt_df):,} records")

    # Load hospitalization data
    hosp_table = Hospitalization.from_file(config_path='clif_config.json')
    hosp_df = hosp_table.df.copy()
    print(f"Hospitalization data loaded: {len(hosp_df):,} records")

    # Load patient data
    patient_table = Patient.from_file(config_path='clif_config.json')
    patient_df = patient_table.df.copy()
    print(f"Patient data loaded: {len(patient_df):,} records")
    return adt_df, hosp_df, patient_df


@app.cell
def _(HospitalDiagnosis):
    # Load hospital diagnosis data for ESRD identification
    hosp_dx_table = HospitalDiagnosis.from_file(config_path='clif_config.json')
    hosp_dx_df = hosp_dx_table.df.copy()
    print(f"Hospital diagnosis data loaded: {len(hosp_dx_df):,} records")
    return (hosp_dx_df,)


@app.cell
def _(mo):
    mo.md(r"""## Initialize CONSORT Tracking""")
    return


@app.cell
def _(hosp_df):
    # Initialize CONSORT flowchart tracking dictionary
    consort_counts = {}

    # Initial count: total hospitalizations
    initial_hosp_count = hosp_df['hospitalization_id'].nunique()
    consort_counts['initial'] = {
        'description': 'Total hospitalizations',
        'n': initial_hosp_count
    }
    print(f"=== CONSORT Tracking Initialized ===")
    print(f"Initial hospitalizations: {initial_hosp_count:,}")
    return (consort_counts,)


@app.cell
def _(mo):
    mo.md(r"""## Apply Inclusion Criteria""")
    return


@app.cell
def _(adt_df, hosp_df, pd):
    # Merge ADT with hospitalization data
    print("Merging ADT with hospitalization data...")

    # Select ADT columns including hospital_type and hospital_id if available
    adt_cols = ['hospitalization_id', 'location_category', 'location_type', 'in_dttm', 'out_dttm']
    if 'hospital_type' in adt_df.columns:
        adt_cols.append('hospital_type')
    if 'hospital_id' in adt_df.columns:
        adt_cols.append('hospital_id')

    icu_data = pd.merge(
        adt_df[adt_cols],
        hosp_df[['patient_id', 'hospitalization_id', 'age_at_admission', 'admission_dttm', 'discharge_dttm', 'discharge_category']],
        on='hospitalization_id',
        how='inner'
    )

    # Normalize location_category to lowercase for consistent matching
    icu_data['location_category'] = icu_data['location_category'].str.lower()

    print(f"Merged data: {len(icu_data):,} records")

    # Convert datetime columns
    datetime_cols = ['in_dttm', 'out_dttm', 'admission_dttm', 'discharge_dttm']
    for col in datetime_cols:
        icu_data[col] = pd.to_datetime(icu_data[col])
    return (icu_data,)


@app.cell
def _(consort_counts, icu_data):
    # INCLUSION: Year filter (2018-2024) and Adult filter (age >= 18)
    print("\n=== Applying Inclusion Criteria ===")

    # Filter for admissions 2018-2024, adults >= 18
    icu_filtered = icu_data[
        (icu_data['admission_dttm'].dt.year >= 2018) &
        (icu_data['admission_dttm'].dt.year <= 2024) &
        (icu_data['out_dttm'].dt.year <= 2024) &
        (icu_data['age_at_admission'] >= 18) &
        (icu_data['age_at_admission'].notna())
    ].copy()

    n_year_adult = icu_filtered['hospitalization_id'].nunique()
    consort_counts['year_adult'] = {
        'description': 'Adults (≥18), Years 2018-2024',
        'n': n_year_adult
    }
    print(f"After year (2018-2024) and adult (≥18) filter: {n_year_adult:,} hospitalizations")
    return (icu_filtered,)


@app.cell
def _(consort_counts, icu_filtered):
    # INCLUSION: Hospital type filter (academic or community, exclude LTACH)
    print("Applying hospital type filter...")

    if 'hospital_type' in icu_filtered.columns:
        # Normalize hospital_type
        icu_filtered['hospital_type'] = icu_filtered['hospital_type'].str.lower()

        # Filter for academic and community only (exclude LTACH)
        icu_hosp_type = icu_filtered[
            icu_filtered['hospital_type'].isin(['academic', 'community'])
        ].copy()

        n_hosp_type = icu_hosp_type['hospitalization_id'].nunique()
        consort_counts['hospital_type'] = {
            'description': 'Academic or Community hospital (exclude LTACH)',
            'n': n_hosp_type
        }
        print(f"After hospital type filter (academic/community): {n_hosp_type:,} hospitalizations")
    else:
        print("WARNING: hospital_type column not found in ADT data. Skipping hospital type filter.")
        icu_hosp_type = icu_filtered.copy()
        consort_counts['hospital_type'] = {
            'description': 'Hospital type filter skipped (column not available)',
            'n': icu_hosp_type['hospitalization_id'].nunique()
        }
    return (icu_hosp_type,)


@app.cell
def _(icu_hosp_type):
    # Consolidate consecutive ADT stays AFTER filters (performance optimization)
    # Only for hospitalizations with ED before ICU

    print("Consolidating consecutive ADT location stays...")

    df = icu_hosp_type.copy()

    # Find hospitalizations with ED followed by ICU (in time order)
    def has_ed_then_icu(group):
        group = group.sort_values('in_dttm')
        cats = group['location_category'].tolist()
        if 'ed' in cats and 'icu' in cats:
            return cats.index('ed') < cats.index('icu')
        return False

    relevant_hosps = df.groupby('hospitalization_id').filter(
        lambda x: has_ed_then_icu(x)
    )['hospitalization_id'].unique().tolist()

    print(f"Hospitalizations with ED→ICU pathway: {len(relevant_hosps):,}")

    # Filter to relevant hospitalizations
    df = df[df['hospitalization_id'].isin(relevant_hosps)].copy()
    df = df.sort_values(['hospitalization_id', 'in_dttm']).reset_index(drop=True)

    # Handle ICU→OR→ICU pattern
    df['effective_category'] = df['location_category'].copy()

    for hosp_id in df['hospitalization_id'].unique():
        mask = df['hospitalization_id'] == hosp_id
        hosp_subset = df.loc[mask].copy()

        for i in range(1, len(hosp_subset) - 1):
            idx = hosp_subset.index[i]
            prev_idx = hosp_subset.index[i - 1]
            next_idx = hosp_subset.index[i + 1]

            curr_cat = hosp_subset.loc[idx, 'location_category']
            prev_cat = hosp_subset.loc[prev_idx, 'effective_category']
            next_cat = hosp_subset.loc[next_idx, 'location_category']

            if curr_cat == 'or' and prev_cat == 'icu' and next_cat == 'icu':
                df.loc[idx, 'effective_category'] = 'icu'

    # Group consecutive same-category stays
    df['category_change'] = (
        (df['effective_category'] != df['effective_category'].shift()) |
        (df['hospitalization_id'] != df['hospitalization_id'].shift())
    ).astype(int)
    df['stay_group'] = df['category_change'].cumsum()

    # Aggregate
    agg_dict = {
        'hospitalization_id': 'first',
        'patient_id': 'first',
        'location_category': 'first',
        'effective_category': 'first',
        'location_type': 'first',
        'in_dttm': 'min',
        'out_dttm': 'max',
        'age_at_admission': 'first',
        'admission_dttm': 'first',
        'discharge_dttm': 'first',
        'discharge_category': 'first'
    }

    for opt_col in ['hospital_type', 'hospital_id']:
        if opt_col in df.columns:
            agg_dict[opt_col] = 'first'

    icu_hosp_type_consolidated = df.groupby('stay_group').agg(agg_dict).reset_index(drop=True)
    icu_hosp_type_consolidated['location_category'] = icu_hosp_type_consolidated['effective_category']
    icu_hosp_type_consolidated = icu_hosp_type_consolidated.drop(columns=['effective_category'])

    print(f"Records after consolidation: {len(icu_hosp_type_consolidated):,}")
    return (icu_hosp_type_consolidated,)


@app.cell
def _(consort_counts, icu_hosp_type_consolidated):
    # INCLUSION: First ICU admission (ALL ICU types)
    print("Filtering for first ICU admission (all ICU types)...")

    # Filter for ICU locations only
    icu_only = icu_hosp_type_consolidated[
        icu_hosp_type_consolidated['location_category'] == 'icu'
    ].copy()

    # Get first ICU stay per hospitalization (sorted by in_dttm)
    first_icu = icu_only.sort_values('in_dttm').groupby('hospitalization_id').first().reset_index()

    # Rename columns for clarity
    first_icu = first_icu.rename(columns={
        'in_dttm': 'start_dttm',
        'out_dttm': 'end_dttm'
    })

    n_first_icu = len(first_icu)
    consort_counts['first_icu'] = {
        'description': 'First ICU admission (all ICU types)',
        'n': n_first_icu
    }
    print(f"First ICU admissions (all ICU types): {n_first_icu:,} hospitalizations")
    return (first_icu,)


@app.cell
def _(consort_counts, first_icu, icu_hosp_type_consolidated, pd):
    # INCLUSION: Admitted to ICU via ED
    print("Filtering for ED-to-ICU pathway...")

    # Get ED records from consolidated ADT (consecutive ED stays already merged)
    ed_records = icu_hosp_type_consolidated[
        icu_hosp_type_consolidated['location_category'].str.lower() == 'ed'
    ][['hospitalization_id', 'in_dttm', 'out_dttm']].copy()

    ed_records = ed_records.rename(columns={
        'in_dttm': 'ed_in_dttm',
        'out_dttm': 'ed_out_dttm'
    })

    # Convert datetime
    ed_records['ed_in_dttm'] = pd.to_datetime(ed_records['ed_in_dttm'])
    ed_records['ed_out_dttm'] = pd.to_datetime(ed_records['ed_out_dttm'])

    print(f"ED records found: {len(ed_records):,}")

    # Merge ED records with first ICU stays
    first_icu_with_ed = pd.merge(
        first_icu,
        ed_records,
        on='hospitalization_id',
        how='left'
    )

    # Filter: ED record must exist AND ED discharge <= ICU admission
    # This ensures patient came through ED before ICU
    first_icu_ed_pathway = first_icu_with_ed[
        (first_icu_with_ed['ed_in_dttm'].notna()) &
        (first_icu_with_ed['ed_out_dttm'] <= first_icu_with_ed['start_dttm'])
    ].copy()

    # Keep earliest ED record per hospitalization (in case multiple ED visits)
    first_icu_ed_pathway = first_icu_ed_pathway.sort_values('ed_in_dttm').groupby('hospitalization_id').first().reset_index()

    n_ed_to_icu = len(first_icu_ed_pathway)
    consort_counts['ed_to_icu'] = {
        'description': 'Admitted to ICU via ED',
        'n': n_ed_to_icu
    }
    print(f"After ED-to-ICU pathway filter: {n_ed_to_icu:,} hospitalizations")
    return (first_icu_ed_pathway,)


@app.cell
def _(mo):
    mo.md(r"""## Apply Exclusion Criteria""")
    return


@app.cell
def _(consort_counts, first_icu_ed_pathway, hosp_df, pd):
    # EXCLUSION: Prior hospitalization within 48 hours
    print("\n=== Applying Exclusion Criteria ===")
    print("Checking for prior hospitalizations within 48 hours...")

    # Get all hospitalizations for patients in cohort
    cohort_patients = first_icu_ed_pathway['patient_id'].unique()

    # Get discharge times for all hospitalizations of these patients
    all_hosp = hosp_df[hosp_df['patient_id'].isin(cohort_patients)][
        ['patient_id', 'hospitalization_id', 'admission_dttm', 'discharge_dttm']
    ].copy()
    all_hosp['admission_dttm'] = pd.to_datetime(all_hosp['admission_dttm'])
    all_hosp['discharge_dttm'] = pd.to_datetime(all_hosp['discharge_dttm'])

    # Merge cohort with ED admission time
    cohort_with_timing = first_icu_ed_pathway[['hospitalization_id', 'patient_id', 'ed_in_dttm']].copy()

    # For each hospitalization, find if there's a prior discharge within 48 hours
    def check_prior_48hr(row, all_hosp_df):
        patient_hosps = all_hosp_df[
            (all_hosp_df['patient_id'] == row['patient_id']) &
            (all_hosp_df['hospitalization_id'] != row['hospitalization_id']) &
            (all_hosp_df['discharge_dttm'] < row['ed_in_dttm'])
        ]
        if len(patient_hosps) == 0:
            return False  # No prior hospitalizations
        # Check if any discharge is within 48 hours of ED admission
        hours_since_discharge = (row['ed_in_dttm'] - patient_hosps['discharge_dttm']).dt.total_seconds() / 3600
        return (hours_since_discharge <= 48).any()

    # Apply check
    cohort_with_timing['has_prior_48hr'] = cohort_with_timing.apply(
        lambda row: check_prior_48hr(row, all_hosp), axis=1
    )

    # Get hospitalizations to exclude
    exclude_48hr = cohort_with_timing[cohort_with_timing['has_prior_48hr']]['hospitalization_id'].tolist()

    # Filter cohort
    cohort_no_readmit = first_icu_ed_pathway[
        ~first_icu_ed_pathway['hospitalization_id'].isin(exclude_48hr)
    ].copy()

    n_excluded_48hr = len(first_icu_ed_pathway) - len(cohort_no_readmit)
    consort_counts['excl_48hr_readmit'] = {
        'description': 'Excluded: Prior hospitalization within 48 hours',
        'n_excluded': n_excluded_48hr,
        'n_remaining': len(cohort_no_readmit)
    }
    print(f"Excluded (prior hospitalization within 48hr): {n_excluded_48hr:,}")
    print(f"Remaining: {len(cohort_no_readmit):,} hospitalizations")
    return (cohort_no_readmit,)


@app.cell
def _(Labs, apply_outlier_handling, cohort_no_readmit, consort_counts):
    # EXCLUSION: Immunocompromised (neutrophils_absolute < 500)
    print("Checking for immunocompromised patients (neutrophils_absolute < 500)...")

    # Get hospitalization IDs
    cohort_hosp_ids_neutro = cohort_no_readmit['hospitalization_id'].astype(str).unique().tolist()

    # Load neutrophils lab data
    try:
        neutrophil_table = Labs.from_file(
            config_path='clif_config.json',
            filters={
                'hospitalization_id': cohort_hosp_ids_neutro,
                'lab_category': ['neutrophils_absolute']
            },
            columns=['hospitalization_id', 'lab_result_dttm', 'lab_category', 'lab_value_numeric']
        )

        neutrophil_df = neutrophil_table.df.copy()
        print(f"Neutrophil labs loaded: {len(neutrophil_df):,} records")

        # Apply outlier handling
        apply_outlier_handling(neutrophil_table)
        neutrophil_df = neutrophil_table.df.copy()

        # Count hospitalizations with neutrophil data
        hosp_with_neutro = neutrophil_df['hospitalization_id'].nunique()
        hosp_missing_neutro = len(cohort_no_readmit) - hosp_with_neutro
        print(f"  Hospitalizations with neutrophil data: {hosp_with_neutro:,}")
        print(f"  Hospitalizations missing neutrophil data: {hosp_missing_neutro:,} (NOT excluded)")

        # Find hospitalizations with ANY neutrophil < 500 at any timepoint
        # NOTE: Only exclude if neutrophil < 500 is FOUND. Missing labs = NOT excluded.
        immunocompromised_hosp = neutrophil_df[
            neutrophil_df['lab_value_numeric'] < 0.5
        ]['hospitalization_id'].unique()

        # Filter cohort - exclude only those with neutrophil < 500 (missing labs stay in cohort)
        cohort_no_immunocomp = cohort_no_readmit[
            ~cohort_no_readmit['hospitalization_id'].isin(immunocompromised_hosp)
        ].copy()

        n_excluded_immunocomp = len(cohort_no_readmit) - len(cohort_no_immunocomp)

    except Exception as e:
        print(f"WARNING: Could not load neutrophils_absolute lab data: {e}")
        print("Skipping immunocompromised exclusion.")
        cohort_no_immunocomp = cohort_no_readmit.copy()
        n_excluded_immunocomp = 0

    consort_counts['excl_immunocompromised'] = {
        'description': 'Excluded: Immunocompromised (neutrophils < 500)',
        'n_excluded': n_excluded_immunocomp,
        'n_remaining': len(cohort_no_immunocomp)
    }
    print(f"Excluded (immunocompromised): {n_excluded_immunocomp:,}")
    print(f"Remaining: {len(cohort_no_immunocomp):,} hospitalizations")
    return cohort_no_immunocomp, neutrophil_df


@app.cell
def _(neutrophil_df):
    neutrophil_df['lab_value_numeric'].value_counts()
    return


@app.cell
def _(cohort_no_immunocomp, consort_counts):
    # EXCLUSION: Hospital LOS < 96 hours (4 days)
    print("Checking for hospital LOS < 96 hours...")

    # Calculate hospital LOS in hours
    cohort_no_immunocomp['hospital_los_hours'] = (
        cohort_no_immunocomp['discharge_dttm'] - cohort_no_immunocomp['admission_dttm']
    ).dt.total_seconds() / 3600

    # Filter: hospital LOS >= 96 hours
    cohort_min_los = cohort_no_immunocomp[
        cohort_no_immunocomp['hospital_los_hours'] >= 96
    ].copy()

    n_excluded_short_los = len(cohort_no_immunocomp) - len(cohort_min_los)
    consort_counts['excl_short_hospital_los'] = {
        'description': 'Excluded: Hospital LOS < 96 hours',
        'n_excluded': n_excluded_short_los,
        'n_remaining': len(cohort_min_los)
    }
    print(f"Excluded (hospital LOS < 96 hours): {n_excluded_short_los:,}")
    print(f"Remaining: {len(cohort_min_los):,} hospitalizations")
    return (cohort_min_los,)


@app.cell
def _(cohort_min_los, consort_counts, patient_df, pd):
    # EXCLUSION: Died within 6 hours of ICU arrival
    print("Checking for deaths within 6 hours of ICU arrival...")

    # Merge with patient death_dttm
    cohort_with_death = pd.merge(
        cohort_min_los,
        patient_df[['patient_id', 'death_dttm']],
        on='patient_id',
        how='left'
    )

    # Convert death_dttm
    cohort_with_death['death_dttm'] = pd.to_datetime(cohort_with_death['death_dttm'])

    # Calculate hours from ICU admission (start_dttm) to death
    cohort_with_death['hours_to_death'] = (
        cohort_with_death['death_dttm'] - cohort_with_death['start_dttm']
    ).dt.total_seconds() / 3600

    # Exclude if death within 6 hours of ICU arrival
    # Keep if: no death_dttm (survived), died after 6 hours, or death before ICU (data issue)
    cohort_no_early_death = cohort_with_death[
        (cohort_with_death['death_dttm'].isna()) |  # Did not die
        (cohort_with_death['hours_to_death'] > 6) |  # Died after 6 hours
        (cohort_with_death['hours_to_death'] < 0)    # Death recorded before ICU admission
    ].copy()

    n_excluded_early_death = len(cohort_min_los) - len(cohort_no_early_death)
    consort_counts['excl_early_death'] = {
        'description': 'Excluded: Died within 6 hours of ICU arrival',
        'n_excluded': n_excluded_early_death,
        'n_remaining': len(cohort_no_early_death)
    }
    print(f"Excluded (died within 6 hours of ICU): {n_excluded_early_death:,}")
    print(f"Remaining: {len(cohort_no_early_death):,} hospitalizations")

    # Final cohort count
    consort_counts['final'] = {
        'description': 'Final analytic cohort',
        'n': len(cohort_no_early_death)
    }
    print(f"\n=== FINAL COHORT: {len(cohort_no_early_death):,} hospitalizations ===")
    return (cohort_no_early_death,)


@app.cell
def _(mo):
    mo.md(r"""## Generate CONSORT Flowchart""")
    return


@app.cell
def _(consort_counts, json, os, plt):
    # Create CONSORT flowchart
    def create_consort_flowchart(counts, output_path='PHI_DATA/consort_flowchart.png'):
        """
        Generate a CONSORT-style inclusion/exclusion flowchart.
        """
        fig, ax = plt.subplots(figsize=(14, 18))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 22)
        ax.axis('off')

        # Colors
        inclusion_color = '#B3D9FF'  # Light blue
        exclusion_color = '#FFFFB3'  # Light yellow
        final_color = '#B3FFB3'      # Light green

        # Y positions for flow
        y_pos = {
            'initial': 20,
            'year_adult': 17,
            'hospital_type': 14,
            'first_icu': 11,
            'ed_to_icu': 8,
            'exclusions': 5,
            'final': 1.5
        }

        # Box dimensions
        box_width = 4
        box_height = 1.5

        def draw_box(x, y, text, color, fontsize=9):
            """Draw a rounded box with text."""
            box = plt.Rectangle((x - box_width/2, y - box_height/2),
                                box_width, box_height,
                                facecolor=color, edgecolor='black',
                                linewidth=1.5, zorder=2,
                                joinstyle='round')
            ax.add_patch(box)
            ax.text(x, y, text, ha='center', va='center',
                   fontsize=fontsize, wrap=True, zorder=3)

        def draw_arrow(x1, y1, x2, y2, color='black'):
            """Draw an arrow between two points."""
            ax.annotate('', xy=(x2, y2 + box_height/2),
                       xytext=(x1, y1 - box_height/2),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        # Main flow (center x = 5)
        main_x = 5

        # 1. Initial box
        if 'initial' in counts:
            text = f"Total hospitalizations\nN = {counts['initial']['n']:,}"
            draw_box(main_x, y_pos['initial'], text, inclusion_color)

        # 2. Year & Adult filter
        if 'year_adult' in counts:
            text = f"Adults (≥18), 2018-2024\nN = {counts['year_adult']['n']:,}"
            draw_box(main_x, y_pos['year_adult'], text, inclusion_color)
            draw_arrow(main_x, y_pos['initial'], main_x, y_pos['year_adult'])

        # 3. Hospital type filter
        if 'hospital_type' in counts:
            text = f"Academic/Community hospital\n(exclude LTACH)\nN = {counts['hospital_type']['n']:,}"
            draw_box(main_x, y_pos['hospital_type'], text, inclusion_color)
            draw_arrow(main_x, y_pos['year_adult'], main_x, y_pos['hospital_type'])

        # 4. First ICU
        if 'first_icu' in counts:
            text = f"First ICU admission\n(all ICU types)\nN = {counts['first_icu']['n']:,}"
            draw_box(main_x, y_pos['first_icu'], text, inclusion_color)
            draw_arrow(main_x, y_pos['hospital_type'], main_x, y_pos['first_icu'])

        # 5. ED to ICU
        if 'ed_to_icu' in counts:
            text = f"Admitted via ED\nN = {counts['ed_to_icu']['n']:,}"
            draw_box(main_x, y_pos['ed_to_icu'], text, inclusion_color)
            draw_arrow(main_x, y_pos['first_icu'], main_x, y_pos['ed_to_icu'])

        # Exclusions box (right side)
        excl_x = 9.5
        excl_y = y_pos['exclusions']

        # Calculate total exclusions
        exclusion_items = []
        if 'excl_48hr_readmit' in counts:
            exclusion_items.append(f"Prior hospitalization <48h: n={counts['excl_48hr_readmit']['n_excluded']:,}")
        if 'excl_immunocompromised' in counts:
            exclusion_items.append(f"Immunocompromised: n={counts['excl_immunocompromised']['n_excluded']:,}")
        if 'excl_short_hospital_los' in counts:
            exclusion_items.append(f"Hospital LOS <96h: n={counts['excl_short_hospital_los']['n_excluded']:,}")
        if 'excl_early_death' in counts:
            exclusion_items.append(f"Died <6h of ICU: n={counts['excl_early_death']['n_excluded']:,}")

        total_excluded = sum([
            counts.get('excl_48hr_readmit', {}).get('n_excluded', 0),
            counts.get('excl_immunocompromised', {}).get('n_excluded', 0),
            counts.get('excl_short_hospital_los', {}).get('n_excluded', 0),
            counts.get('excl_early_death', {}).get('n_excluded', 0)
        ])

        if exclusion_items:
            excl_text = "EXCLUDED\n" + "\n".join(exclusion_items) + f"\n\nTotal excluded: n={total_excluded:,}"
            # Larger box for exclusions
            excl_box = plt.Rectangle((excl_x - 2.5, excl_y - 2),
                                     5, 4,
                                     facecolor=exclusion_color, edgecolor='black',
                                     linewidth=1.5, zorder=2)
            ax.add_patch(excl_box)
            ax.text(excl_x, excl_y, excl_text, ha='center', va='center',
                   fontsize=8, wrap=True, zorder=3)

            # Arrow from ED to ICU to exclusions
            ax.annotate('', xy=(excl_x - 2.5, excl_y),
                       xytext=(main_x + box_width/2, y_pos['ed_to_icu']),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        # Arrow from exclusions area to final
        draw_arrow(main_x, y_pos['ed_to_icu'] - 1, main_x, y_pos['final'] + 1)

        # Final cohort box
        if 'final' in counts:
            text = f"FINAL ANALYTIC COHORT\nN = {counts['final']['n']:,}"
            # Larger final box
            final_box = plt.Rectangle((main_x - 2, y_pos['final'] - 0.75),
                                      4, 1.5,
                                      facecolor=final_color, edgecolor='black',
                                      linewidth=2, zorder=2)
            ax.add_patch(final_box)
            ax.text(main_x, y_pos['final'], text, ha='center', va='center',
                   fontsize=10, fontweight='bold', wrap=True, zorder=3)

        # Title
        ax.set_title('CONSORT Flow Diagram: ICU Cohort Selection',
                    fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"CONSORT flowchart saved to: {output_path}")

        return fig

    # Create PHI_DATA directory if needed
    os.makedirs('PHI_DATA', exist_ok=True)

    # Generate flowchart
    consort_fig = create_consort_flowchart(consort_counts)

    # Save counts to JSON
    with open('PHI_DATA/consort_counts.json', 'w') as f:
        json.dump(consort_counts, f, indent=2, default=str)
    print("CONSORT counts saved to: PHI_DATA/consort_counts.json")
    return


@app.cell
def _(cohort_no_early_death):
    cohort_no_early_death
    return


@app.cell
def _(mo):
    mo.md(r"""## Add Demographics""")
    return


@app.cell
def _(cohort_no_early_death, patient_df, pd):
    # Merge with patient demographics
    print("Adding patient demographics...")

    cohort_df = pd.merge(
        cohort_no_early_death,
        patient_df[['patient_id', 'sex_category', 'ethnicity_category', 'race_category', 'language_category']],
        on='patient_id',
        how='left'
    )

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

    # Filter out negative ICU LOS (data quality issue)
    n_negative_los = (cohort_df['icu_los_days'] < 0).sum()
    if n_negative_los > 0:
        print(f"WARNING: Removing {n_negative_los} records with negative ICU LOS")
        cohort_df = cohort_df[cohort_df['icu_los_days'] >= 0].copy()

    # Calculate ICU Mortality (binary: 1 = died during ICU stay, 0 = did not die during ICU stay)
    cohort_df['icu_mortality'] = (
        (cohort_df['death_dttm'].notna()) &  # death_dttm is not null
        (cohort_df['death_dttm'] >= cohort_df['start_dttm']) &  # death occurred after/at ICU start
        (cohort_df['death_dttm'] <= cohort_df['end_dttm'])  # death occurred before/at ICU end
    ).astype(int)

    print(f"\n=== ICU Mortality ===")
    print(f"Deaths during ICU stay: {cohort_df['icu_mortality'].sum():,} ({cohort_df['icu_mortality'].mean()*100:.2f}%)")
    print(f"Survived ICU stay: {(cohort_df['icu_mortality'] == 0).sum():,}")

    print(f"Final cohort: {len(cohort_df):,} hospitalizations")
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

    print(f"Vitals loaded: {len(vitals_table.df):,} records")

    # Apply outlier handling using clifpy
    print("Applying outlier handling to vitals...")
    apply_outlier_handling(vitals_table)
    print(f"Outlier handling applied")
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

    print(f"Vitals filtered to ICU windows: {len(vitals_icu_window):,} records")

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

    print(f"Vital aggregates calculated for {len(vitals_pivot):,} hospitalizations")
    print(f"  Columns: {[col for col in vitals_pivot.columns if col != 'hospitalization_id']}")

    # Merge vitals back to cohort_df
    cohort_with_vitals = pd.merge(
        cohort_df,
        vitals_pivot,
        on='hospitalization_id',
        how='left'
    )

    print(f"Vitals merged to cohort: {len(cohort_with_vitals):,} hospitalizations")

    # Return final cohort with vitals (renamed to avoid circular dependency)
    return (cohort_with_vitals,)


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
    print(f"Medications loaded: {len(meds_df):,} records")
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

    print(f"Medications filtered to ICU windows: {len(meds_icu_window):,} records")

    # Calculate vasopressor metrics per hospitalization
    print("Calculating vasopressor metrics...")

    # Count unique vasopressor categories per hospitalization
    vaso_summary = meds_icu_window.groupby('hospitalization_id').agg({
        'med_category': lambda x: x.nunique()  # count unique vasopressor categories
    }).reset_index()
    vaso_summary.columns = ['hospitalization_id', 'no_of_vasopressor']

    # Add binary flag: 1 if any vasopressor used, 0 otherwise
    vaso_summary['vasopressor_ever'] = 1

    print(f"Vasopressor metrics calculated for {len(vaso_summary):,} hospitalizations")
    print(f"  Vasopressor usage distribution:")
    print(vaso_summary['no_of_vasopressor'].value_counts().sort_index().to_dict())
    return (vaso_summary,)


@app.cell
def _(cohort_with_vitals, pd, vaso_summary):
    # Merge vasopressor metrics back to cohort
    print("Merging vasopressor metrics to cohort...")

    cohort_with_meds = pd.merge(
        cohort_with_vitals,
        vaso_summary,
        on='hospitalization_id',
        how='left'
    )

    # Fill NaN (no vasopressors) with 0
    cohort_with_meds['vasopressor_ever'] = cohort_with_meds['vasopressor_ever'].fillna(0).astype(int)
    cohort_with_meds['no_of_vasopressor'] = cohort_with_meds['no_of_vasopressor'].fillna(0).astype(int)

    print(f"Vasopressor metrics merged to cohort: {len(cohort_with_meds):,} hospitalizations")
    print(f"  Hospitalizations with vasopressors: {(cohort_with_meds['vasopressor_ever'] == 1).sum():,} ({(cohort_with_meds['vasopressor_ever'] == 1).mean()*100:.1f}%)")
    print(f"  Hospitalizations without vasopressors: {(cohort_with_meds['vasopressor_ever'] == 0).sum():,} ({(cohort_with_meds['vasopressor_ever'] == 0).mean()*100:.1f}%)")

    # Return final cohort with all features
    return (cohort_with_meds,)


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

    print(f"Labs loaded: {len(labs_table.df):,} records")

    # Apply outlier handling using clifpy
    print("Applying outlier handling to labs...")
    apply_outlier_handling(labs_table)
    print(f"Outlier handling applied")
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

    print(f"Labs filtered to ICU windows: {len(labs_icu_window):,} records")

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

    print(f"Lab aggregates calculated for {len(labs_pivot):,} hospitalizations")
    print(f"  Columns: {[col for col in labs_pivot.columns if col != 'hospitalization_id']}")
    return (labs_pivot,)


@app.cell
def _(cohort_with_meds, labs_pivot, pd):
    # Merge labs back to cohort
    print("Merging labs to cohort...")

    cohort_with_labs = pd.merge(
        cohort_with_meds,
        labs_pivot,
        on='hospitalization_id',
        how='left'
    )

    print(f"Labs merged to cohort: {len(cohort_with_labs):,} hospitalizations")

    # Return complete cohort with all features
    return (cohort_with_labs,)


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
    print(f"Respiratory support loaded: {len(resp_df):,} records")

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

    print(f"Respiratory support filtered to ICU windows: {len(resp_icu_window):,} records")

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

    print(f"Respiratory support metrics calculated for {len(resp_summary):,} hospitalizations")
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
def _(cohort_with_labs, pd, resp_summary):
    # Merge respiratory support metrics to cohort
    print("Merging respiratory support metrics to cohort...")

    cohort_with_resp = pd.merge(
        cohort_with_labs,
        resp_summary,
        on='hospitalization_id',
        how='left'
    )

    # Fill NaN (no NIPPV/HFNO/IMV) with 0
    cohort_with_resp['NIPPV_ever'] = cohort_with_resp['NIPPV_ever'].fillna(0).astype(int)
    cohort_with_resp['HFNO_ever'] = cohort_with_resp['HFNO_ever'].fillna(0).astype(int)
    cohort_with_resp['IMV_ever'] = cohort_with_resp['IMV_ever'].fillna(0).astype(int)

    print(f"Respiratory support metrics merged to cohort: {len(cohort_with_resp):,} hospitalizations")

    # Return complete cohort with all features including respiratory support
    return (cohort_with_resp,)


@app.cell
def _(mo):
    mo.md(r"""## Load and Process CRRT (Continuous Renal Replacement Therapy)""")
    return


@app.cell
def _(CrrtTherapy, cohort_df):
    # Load CRRT data
    print("Loading CRRT data using clifpy...")

    # Extract cohort hospitalization IDs
    cohort_hosp_ids_crrt = cohort_df['hospitalization_id'].astype(str).unique().tolist()
    print(f"Loading CRRT for {len(cohort_hosp_ids_crrt):,} hospitalizations")

    # Load CRRT table with hospitalization_id filter
    crrt_table = CrrtTherapy.from_file(
        config_path='clif_config.json',
        filters={
            'hospitalization_id': cohort_hosp_ids_crrt
        },
        columns=['hospitalization_id', 'recorded_dttm', 'blood_flow_rate']
    )

    crrt_df = crrt_table.df.copy()
    print(f"CRRT data loaded: {len(crrt_df):,} records")
    return (crrt_df,)


@app.cell
def _(cohort_df, crrt_df, pd):
    # Filter CRRT to ICU stay windows and create crrt_ever flag
    print("Filtering CRRT to ICU stay windows...")

    # Merge CRRT with cohort to get ICU stay windows
    crrt_with_windows = pd.merge(
        crrt_df,
        cohort_df[['hospitalization_id', 'start_dttm', 'end_dttm']],
        on='hospitalization_id',
        how='inner'
    )

    # Convert datetime column
    crrt_with_windows['recorded_dttm'] = pd.to_datetime(crrt_with_windows['recorded_dttm'])

    # Filter CRRT to ICU stay window (start_dttm <= recorded_dttm <= end_dttm)
    crrt_icu_window = crrt_with_windows[
        (crrt_with_windows['recorded_dttm'] >= crrt_with_windows['start_dttm']) &
        (crrt_with_windows['recorded_dttm'] <= crrt_with_windows['end_dttm'])
    ].copy()

    print(f"CRRT filtered to ICU windows: {len(crrt_icu_window):,} records")

    # Create crrt_ever flag (any CRRT record)
    print("Creating crrt_ever flag...")
    crrt_summary = crrt_icu_window.groupby('hospitalization_id').size().reset_index(name='crrt_count')
    crrt_summary['crrt_ever'] = 1

    # Create crrt_with_flow flag (CRRT with blood_flow_rate > 0) for AKI staging
    print("Creating crrt_with_flow flag (blood_flow_rate > 0)...")
    _crrt_with_flow = crrt_icu_window[crrt_icu_window['blood_flow_rate'] > 0]
    _crrt_flow_summary = _crrt_with_flow.groupby('hospitalization_id').size().reset_index(name='crrt_flow_count')
    _crrt_flow_summary['crrt_with_flow'] = 1

    # Merge flow flag into summary
    crrt_summary = pd.merge(
        crrt_summary,
        _crrt_flow_summary[['hospitalization_id', 'crrt_with_flow']],
        on='hospitalization_id',
        how='left'
    )
    crrt_summary['crrt_with_flow'] = crrt_summary['crrt_with_flow'].fillna(0).astype(int)

    print(f"CRRT metrics calculated for {len(crrt_summary):,} hospitalizations")
    print("\n=== CRRT Usage ===")
    print(f"Hospitalizations with any CRRT record: {len(crrt_summary):,}")
    print(f"Hospitalizations with CRRT blood_flow > 0: {(crrt_summary['crrt_with_flow'] == 1).sum():,}")
    return (crrt_summary,)


@app.cell
def _(cohort_with_resp, crrt_summary, pd):
    # Merge CRRT metrics to cohort
    print("Merging CRRT metrics to cohort...")

    cohort_with_crrt = pd.merge(
        cohort_with_resp,
        crrt_summary[['hospitalization_id', 'crrt_ever', 'crrt_with_flow']],
        on='hospitalization_id',
        how='left'
    )

    # Fill NaN (no CRRT) with 0
    cohort_with_crrt['crrt_ever'] = cohort_with_crrt['crrt_ever'].fillna(0).astype(int)
    cohort_with_crrt['crrt_with_flow'] = cohort_with_crrt['crrt_with_flow'].fillna(0).astype(int)

    print(f"CRRT metrics merged to cohort: {len(cohort_with_crrt):,} hospitalizations")
    print(f"  Hospitalizations with CRRT: {(cohort_with_crrt['crrt_ever'] == 1).sum():,} ({(cohort_with_crrt['crrt_ever'] == 1).mean()*100:.1f}%)")
    print(f"  Hospitalizations with CRRT blood_flow > 0: {(cohort_with_crrt['crrt_with_flow'] == 1).sum():,} ({(cohort_with_crrt['crrt_with_flow'] == 1).mean()*100:.1f}%)")

    # Return complete cohort with all features including CRRT
    return (cohort_with_crrt,)


@app.cell
def _(mo):
    mo.md(r"""## Load C. diff Microbiology Data""")
    return


@app.cell
def _(MicrobiologyNonculture, cohort_with_crrt):
    # Load C. diff microbiology data
    print("\n=== Loading C. diff Microbiology Data ===")

    _cohort_hosp_ids_cdiff = cohort_with_crrt['hospitalization_id'].astype(str).unique().tolist()

    cdiff_table = MicrobiologyNonculture.from_file(
        config_path='clif_config.json',
        filters={
            'hospitalization_id': _cohort_hosp_ids_cdiff,
            'organism_category': ['clostridium_difficile'],
            'result_category': ['detected']
        },
        columns=['hospitalization_id', 'result_dttm', 'collect_dttm', 'organism_category', 'result_category']
    )

    cdiff_df = cdiff_table.df.copy()
    print(f"C. diff positive tests loaded: {len(cdiff_df):,} records")
    return (cdiff_df,)


@app.cell
def _(cdiff_df, cohort_with_crrt, pd):
    # Filter C. diff to ICU start -> hospital discharge window
    # Using collect_dttm (specimen collection time) for timing
    print("Filtering C. diff to time window and removing duplicates...")

    cdiff_with_windows = pd.merge(
        cdiff_df,
        cohort_with_crrt[['hospitalization_id', 'start_dttm', 'discharge_dttm']],
        on='hospitalization_id',
        how='inner'
    )

    cdiff_with_windows['collect_dttm'] = pd.to_datetime(cdiff_with_windows['collect_dttm'])

    # Filter to ICU time window using collection datetime
    cdiff_window = cdiff_with_windows[
        (cdiff_with_windows['collect_dttm'] >= cdiff_with_windows['start_dttm']) &
        (cdiff_with_windows['collect_dttm'] <= cdiff_with_windows['discharge_dttm'])
    ].copy()

    print(f"C. diff tests in time window: {len(cdiff_window):,}")

    # Remove duplicates within 14 days per hospitalization (using collect_dttm)
    cdiff_window = cdiff_window.sort_values(['hospitalization_id', 'collect_dttm'])

    def _remove_14day_duplicates(group):
        if len(group) <= 1:
            return group

        _keep_indices = [group.index[0]]  # Always keep first test
        _last_kept_time = group.iloc[0]['collect_dttm']

        for _idx, _row in group.iloc[1:].iterrows():
            _days_since_last = (_row['collect_dttm'] - _last_kept_time).days
            if _days_since_last >= 14:
                _keep_indices.append(_idx)
                _last_kept_time = _row['collect_dttm']

        return group.loc[_keep_indices]

    cdiff_unique = cdiff_window.groupby('hospitalization_id', group_keys=False).apply(_remove_14day_duplicates)

    print(f"C. diff tests after removing 14-day duplicates: {len(cdiff_unique):,}")

    # Create binary flag and get first positive collection timestamp
    cdiff_summary = cdiff_unique.groupby('hospitalization_id').agg(
        cdiff_count=('collect_dttm', 'size'),
        cdiff_first_collect_dttm=('collect_dttm', 'min')
    ).reset_index()
    cdiff_summary['cdiff_positive'] = 1

    print(f"Hospitalizations with C. diff positive: {len(cdiff_summary):,}")
    return (cdiff_summary,)


@app.cell
def _(cdiff_summary, cohort_with_crrt, pd):
    # Merge C. diff flag with cohort
    print("Merging C. diff flag with cohort...")

    cohort_with_cdiff = pd.merge(
        cohort_with_crrt,
        cdiff_summary[['hospitalization_id', 'cdiff_positive', 'cdiff_first_collect_dttm']],
        on='hospitalization_id',
        how='left'
    )

    cohort_with_cdiff['cdiff_positive'] = cohort_with_cdiff['cdiff_positive'].fillna(0).astype(int)

    print(f"C. diff merged to cohort: {len(cohort_with_cdiff):,} hospitalizations")
    print(f"  C. diff positive: {(cohort_with_cdiff['cdiff_positive'] == 1).sum():,} ({(cohort_with_cdiff['cdiff_positive'] == 1).mean()*100:.2f}%)")
    return (cohort_with_cdiff,)


@app.cell
def _(mo):
    mo.md(r"""## Compute AKI Staging (ACORN JAMA 2024)""")
    return


@app.cell
def _(Labs, apply_outlier_handling, cohort_with_cdiff, pd):
    # Calculate baseline creatinine (measured or estimated)
    print("\n=== AKI Staging: Calculating Baseline Creatinine ===")
    print("Priority: Minimum creatinine from 12 months before ED presentation")
    print("Fallback: Estimated formula (0.74 - 0.2*female + 0.08*Black + 0.003*age)")

    # Load creatinine for baseline lookup (12 months before ED)
    cohort_hosp_ids_baseline = cohort_with_cdiff['hospitalization_id'].astype(str).unique().tolist()

    baseline_cr_table = Labs.from_file(
        config_path='clif_config.json',
        filters={
            'hospitalization_id': cohort_hosp_ids_baseline,
            'lab_category': ['creatinine']
        },
        columns=['hospitalization_id', 'lab_result_dttm', 'lab_category', 'lab_value_numeric']
    )

    apply_outlier_handling(baseline_cr_table)
    baseline_cr_df = baseline_cr_table.df.copy()
    print(f"Creatinine labs loaded for baseline lookup: {len(baseline_cr_df):,} records")

    # Merge with cohort to get ed_in_dttm
    baseline_cr_with_ed = pd.merge(
        baseline_cr_df,
        cohort_with_cdiff[['hospitalization_id', 'ed_in_dttm']],
        on='hospitalization_id',
        how='inner'
    )

    baseline_cr_with_ed['lab_result_dttm'] = pd.to_datetime(baseline_cr_with_ed['lab_result_dttm'])
    baseline_cr_with_ed['ed_in_dttm'] = pd.to_datetime(baseline_cr_with_ed['ed_in_dttm'])

    # Calculate 12 months before ED
    baseline_cr_with_ed['baseline_window_start'] = baseline_cr_with_ed['ed_in_dttm'] - pd.Timedelta(days=365)

    # Filter to 12 months before ED (excluding day of ED presentation)
    baseline_window = baseline_cr_with_ed[
        (baseline_cr_with_ed['lab_result_dttm'] >= baseline_cr_with_ed['baseline_window_start']) &
        (baseline_cr_with_ed['lab_result_dttm'] < baseline_cr_with_ed['ed_in_dttm'])
    ].copy()

    print(f"Creatinine records in 12-month baseline window: {len(baseline_window):,}")

    # Get MINIMUM creatinine per hospitalization (most conservative baseline)
    measured_baseline = baseline_window.groupby('hospitalization_id').agg(
        cr_baseline_measured=('lab_value_numeric', 'min')
    ).reset_index()

    print(f"Hospitalizations with measured baseline: {len(measured_baseline):,}")

    # Merge measured baseline with cohort
    cohort_with_baseline = pd.merge(
        cohort_with_cdiff,
        measured_baseline,
        on='hospitalization_id',
        how='left'
    )

    # Estimated baseline formula function
    def estimate_baseline_cr(row):
        cr_est = 0.74
        if str(row['sex_category']).lower() == 'female':
            cr_est -= 0.2
        race = str(row['race_category']).lower() if pd.notna(row['race_category']) else ''
        if 'black' in race or 'african' in race:
            cr_est += 0.08
        if pd.notna(row['age_at_admission']):
            cr_est += 0.003 * row['age_at_admission']
        return cr_est

    # Calculate estimated baseline for all
    cohort_with_baseline['cr_baseline_estimated'] = cohort_with_baseline.apply(estimate_baseline_cr, axis=1)

    # Use measured if available, otherwise estimated
    cohort_with_baseline['cr_baseline'] = cohort_with_baseline['cr_baseline_measured'].fillna(
        cohort_with_baseline['cr_baseline_estimated']
    )

    # Track baseline source
    cohort_with_baseline['cr_baseline_source'] = cohort_with_baseline['cr_baseline_measured'].apply(
        lambda x: 'measured' if pd.notna(x) else 'estimated'
    )

    # Summary
    n_measured = (cohort_with_baseline['cr_baseline_source'] == 'measured').sum()
    n_estimated = (cohort_with_baseline['cr_baseline_source'] == 'estimated').sum()
    print(f"\nBaseline creatinine source:")
    print(f"  Measured (12-month min): {n_measured:,} ({100*n_measured/len(cohort_with_baseline):.1f}%)")
    print(f"  Estimated (formula): {n_estimated:,} ({100*n_estimated/len(cohort_with_baseline):.1f}%)")
    print(f"\nBaseline creatinine statistics:")
    print(f"  Mean: {cohort_with_baseline['cr_baseline'].mean():.3f} mg/dL")
    print(f"  Median: {cohort_with_baseline['cr_baseline'].median():.3f} mg/dL")
    return (cohort_with_baseline,)


@app.cell
def _(Labs, apply_outlier_handling, cohort_with_baseline):
    # Load creatinine for AKI window (ICU stay only)
    print("\n=== Loading AKI Window Creatinine (ICU stay only) ===")

    cohort_hosp_ids_aki = cohort_with_baseline['hospitalization_id'].astype(str).unique().tolist()

    aki_cr_table = Labs.from_file(
        config_path='clif_config.json',
        filters={
            'hospitalization_id': cohort_hosp_ids_aki,
            'lab_category': ['creatinine']
        },
        columns=['hospitalization_id', 'lab_result_dttm', 'lab_category', 'lab_value_numeric']
    )

    apply_outlier_handling(aki_cr_table)
    aki_cr_df = aki_cr_table.df.copy()
    print(f"Creatinine labs loaded: {len(aki_cr_df):,} records")
    return (aki_cr_df,)


@app.cell
def _(aki_cr_df, cohort_with_baseline, pd):
    # Filter creatinine to AKI window (ICU stay: start_dttm to end_dttm)
    print("Filtering creatinine to AKI window (ICU stay only)...")

    aki_cr_with_windows = pd.merge(
        aki_cr_df,
        cohort_with_baseline[['hospitalization_id', 'start_dttm', 'end_dttm']],
        on='hospitalization_id',
        how='inner'
    )

    aki_cr_with_windows['lab_result_dttm'] = pd.to_datetime(aki_cr_with_windows['lab_result_dttm'])

    # Filter: ICU admission to ICU discharge
    aki_window_cr = aki_cr_with_windows[
        (aki_cr_with_windows['lab_result_dttm'] >= aki_cr_with_windows['start_dttm']) &
        (aki_cr_with_windows['lab_result_dttm'] <= aki_cr_with_windows['end_dttm'])
    ].copy()

    print(f"Creatinine records in ICU stay: {len(aki_window_cr):,}")

    # Get highest creatinine per hospitalization
    aki_cr_summary = aki_window_cr.groupby('hospitalization_id').agg(
        cr_highest_icu=('lab_value_numeric', 'max')
    ).reset_index()

    # Track hospitalizations with missing creatinine in AKI window
    hosp_with_cr = set(aki_cr_summary['hospitalization_id'].unique())
    hosp_total = set(cohort_with_baseline['hospitalization_id'].unique())
    hosp_missing_cr = hosp_total - hosp_with_cr

    print(f"Hospitalizations with ICU creatinine: {len(aki_cr_summary):,}")
    print(f"Hospitalizations MISSING creatinine in ICU: {len(hosp_missing_cr):,} ({100*len(hosp_missing_cr)/len(hosp_total):.1f}%)")
    return (aki_cr_summary,)


@app.cell
def _(aki_cr_summary, cohort_with_baseline, hosp_df, hosp_dx_df, pd):
    # Merge AKI window creatinine with cohort
    cohort_with_aki_cr = pd.merge(
        cohort_with_baseline,
        aki_cr_summary,
        on='hospitalization_id',
        how='left'
    )

    # Calculate creatinine ratio and increase (using highest ICU creatinine)
    cohort_with_aki_cr['cr_ratio'] = (
        cohort_with_aki_cr['cr_highest_icu'] / cohort_with_aki_cr['cr_baseline']
    )
    cohort_with_aki_cr['cr_increase'] = (
        cohort_with_aki_cr['cr_highest_icu'] - cohort_with_aki_cr['cr_baseline']
    )

    # ESRD identification from prior hospitalizations
    ESRD_ICD10_CODES = ['I12.0', 'I13.11', 'I13.2', 'I27.2', 'N18.6', 'Z49.01', 'Z49.31']

    def get_prior_esrd_status(cohort_df, hosp_df, hosp_dx_df):
        """
        For each hospitalization in cohort, check if patient had ESRD diagnosis
        in any PRIOR hospitalization (not including current).
        """
        # Merge diagnosis codes with hospitalization dates
        dx_with_dates = hosp_dx_df.merge(
            hosp_df[['hospitalization_id', 'patient_id', 'admission_dttm']],
            on='hospitalization_id',
            how='left'
        )

        # Filter for ESRD codes only
        esrd_dx = dx_with_dates[dx_with_dates['diagnosis_code'].isin(ESRD_ICD10_CODES)]

        # For each row in cohort, check for prior ESRD
        def check_prior_esrd(row):
            prior = esrd_dx[
                (esrd_dx['patient_id'] == row['patient_id']) &
                (esrd_dx['admission_dttm'] < row['admission_dttm'])
            ]
            return len(prior) > 0

        return cohort_df.apply(check_prior_esrd, axis=1)

    # Add ESRD status based on prior hospitalizations
    cohort_with_aki_cr['has_prior_esrd'] = get_prior_esrd_status(
        cohort_with_aki_cr, hosp_df, hosp_dx_df
    )
    esrd_count = cohort_with_aki_cr['has_prior_esrd'].sum()
    print(f"\nESRD patients (from prior hospitalizations): {esrd_count:,} ({100*esrd_count/len(cohort_with_aki_cr):.1f}%)")

    # Calculate AKI stage (ACORN JAMA 2024 definition)
    # ESRD patients: only stage 0 (alive) or 4 (death)
    def calculate_aki_stage(row):
        # ESRD patients: only stage 0 (alive) or 4 (death)
        if row.get('has_prior_esrd', False):
            if row['icu_mortality'] == 1:
                return 4
            return 0

        # Stage 4: Death in ICU (non-ESRD patients)
        if row['icu_mortality'] == 1:
            return 4

        cr_ratio = row['cr_ratio'] if pd.notna(row['cr_ratio']) else 0
        cr_increase = row['cr_increase'] if pd.notna(row['cr_increase']) else 0
        cr_highest = row['cr_highest_icu'] if pd.notna(row['cr_highest_icu']) else 0
        crrt_flow = row['crrt_with_flow'] if pd.notna(row['crrt_with_flow']) else 0

        # Stage 3: cr >= 3.0x baseline OR cr >= 4.0 mg/dL OR new CRRT with blood flow > 0
        if cr_ratio >= 3.0 or cr_highest >= 4.0 or crrt_flow == 1:
            return 3

        # Stage 2: cr 2.0-2.9x baseline
        if cr_ratio >= 2.0:
            return 2

        # Stage 1: cr 1.5-1.9x baseline OR increase >= 0.3 mg/dL
        if cr_ratio >= 1.5 or cr_increase >= 0.3:
            return 1

        # Stage 0: No AKI
        return 0

    cohort_with_aki_cr['aki_stage'] = cohort_with_aki_cr.apply(calculate_aki_stage, axis=1)

    print("\n=== AKI Stage Distribution ===")
    _aki_counts = cohort_with_aki_cr['aki_stage'].value_counts().sort_index()
    for _stage, _count in _aki_counts.items():
        _pct = 100 * _count / len(cohort_with_aki_cr)
        print(f"  Stage {_stage}: {_count:,} ({_pct:.1f}%)")

    # Show ESRD-specific breakdown
    print("\n=== AKI Stage by ESRD Status ===")
    print("Non-ESRD patients:")
    non_esrd = cohort_with_aki_cr[~cohort_with_aki_cr['has_prior_esrd']]
    for _stage, _count in non_esrd['aki_stage'].value_counts().sort_index().items():
        _pct = 100 * _count / len(non_esrd)
        print(f"  Stage {_stage}: {_count:,} ({_pct:.1f}%)")
    print("ESRD patients:")
    esrd = cohort_with_aki_cr[cohort_with_aki_cr['has_prior_esrd']]
    if len(esrd) > 0:
        for _stage, _count in esrd['aki_stage'].value_counts().sort_index().items():
            _pct = 100 * _count / len(esrd)
            print(f"  Stage {_stage}: {_count:,} ({_pct:.1f}%)")
    else:
        print("  No ESRD patients in cohort")
    return (cohort_with_aki_cr,)


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
    print("ClifOrchestrator initialized for SOFA")
    return (co_sofa,)


@app.cell
def _(cohort_with_aki_cr, pd):
    # Prepare cohort for SOFA computation (whole ICU stay)
    print("Preparing cohort for SOFA score computation (whole ICU stay)...")

    sofa_cohort_df = pd.DataFrame({
        'hospitalization_id': cohort_with_aki_cr['hospitalization_id'],
        'start_time': cohort_with_aki_cr['start_dttm'],
        'end_time': cohort_with_aki_cr['end_dttm']
    })

    print(f"SOFA cohort prepared: {len(sofa_cohort_df):,} hospitalizations (whole ICU stay window)")
    return (sofa_cohort_df,)


@app.cell
def _(cohort_with_aki_cr):
    # Extract hospitalization IDs for SOFA table filtering
    print("Extracting hospitalization IDs for SOFA data filtering...")

    sofa_cohort_ids = cohort_with_aki_cr['hospitalization_id'].astype(str).unique().tolist()

    print(f"Extracted {len(sofa_cohort_ids):,} hospitalization IDs")
    return (sofa_cohort_ids,)


@app.cell
def _():
    return


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

    print("All SOFA tables loaded with category filters")
    return


@app.cell
def _(co_sofa):
    # Clean medication data (remove null/NaN med_dose and med_dose_unit)
    print("Cleaning medication data...")

    med_df_sofa = co_sofa.medication_admin_continuous.df.copy()
    initial_count = len(med_df_sofa)

    # Remove null med_dose
    med_df_sofa = med_df_sofa[med_df_sofa['med_dose'].notna()]
    # Remove null med_dose_unit
    med_df_sofa = med_df_sofa[med_df_sofa['med_dose_unit'].notna()]
    # Remove 'nan' string values
    med_df_sofa = med_df_sofa[~med_df_sofa['med_dose_unit'].astype(str).str.lower().isin(['nan', 'none', ''])]

    final_count = len(med_df_sofa)
    co_sofa.medication_admin_continuous.df = med_df_sofa

    print(f"Medication data cleaned: {initial_count:,} → {final_count:,} records ({initial_count - final_count:,} removed)")
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

    print(f"Unit conversion complete: {success_count:,} / {total_count:,} successful ({100*success_count/total_count:.1f}%)")
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
    print(f"SOFA scores computed: {sofa_scores.shape}")
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

    print(f"BMI calculated for {bmi_final['bmi'].notna().sum():,} hospitalizations")
    print(f"  Missing BMI: {bmi_final['bmi'].isna().sum():,}")
    if bmi_final['bmi'].notna().any():
        print(f"  Mean BMI: {bmi_final['bmi'].mean():.2f}")
        print(f"  Median BMI: {bmi_final['bmi'].median():.2f}")
    return (bmi_final,)


@app.cell
def _(bmi_final, cohort_with_aki_cr, pd, sofa_scores):
    # Merge SOFA scores and BMI with cohort
    print("Merging SOFA scores with cohort...")

    cohort_with_sofa_temp = pd.merge(
        cohort_with_aki_cr,
        sofa_scores,
        on='hospitalization_id',
        how='left'
    )

    # Print SOFA merge summary (inline to avoid variable conflicts)
    print(f"SOFA scores merged: {cohort_with_sofa_temp.shape}")
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

    print(f"BMI merged to cohort: {len(cohort_with_sofa):,} hospitalizations")
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
        print(f"Highest Temperature (C):")
        print(f"  Mean: {cohort_with_sofa['highest_temperature'].mean():.2f}")
        print(f"  Median: {cohort_with_sofa['highest_temperature'].median():.2f}")
        print(f"  Missing: {cohort_with_sofa['highest_temperature'].isna().sum():,} ({cohort_with_sofa['highest_temperature'].isna().mean()*100:.1f}%)")

    if 'lowest_temperature' in cohort_with_sofa.columns:
        print(f"Lowest Temperature (C):")
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
        print(f"Highest WBC (10^3/uL):")
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

    if 'crrt_ever' in cohort_with_sofa.columns:
        print(f"\nHospitalizations with CRRT: {(cohort_with_sofa['crrt_ever'] == 1).sum():,} ({(cohort_with_sofa['crrt_ever'] == 1).mean()*100:.1f}%)")
        print(f"Hospitalizations without CRRT: {(cohort_with_sofa['crrt_ever'] == 0).sum():,} ({(cohort_with_sofa['crrt_ever'] == 0).mean()*100:.1f}%)")

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

    print("\n=== AKI Staging (ACORN JAMA 2024) ===")
    if 'aki_stage' in cohort_with_sofa.columns:
        print("AKI Stage Distribution:")
        _stage_desc = {0: 'No AKI', 1: 'Stage 1', 2: 'Stage 2', 3: 'Stage 3', 4: 'Death in ICU'}
        _aki_summary = cohort_with_sofa['aki_stage'].value_counts().sort_index()
        for _s, _cnt in _aki_summary.items():
            _p = 100 * _cnt / len(cohort_with_sofa)
            print(f"  {_stage_desc[_s]}: {_cnt:,} ({_p:.1f}%)")

        print("\nBaseline Creatinine (12-month min before ED, or estimated):")
        if 'cr_baseline_source' in cohort_with_sofa.columns:
            _n_measured = (cohort_with_sofa['cr_baseline_source'] == 'measured').sum()
            _n_estimated = (cohort_with_sofa['cr_baseline_source'] == 'estimated').sum()
            print(f"  Measured (12-month min before ED): {_n_measured:,} ({100*_n_measured/len(cohort_with_sofa):.1f}%)")
            print(f"  Estimated (formula): {_n_estimated:,} ({100*_n_estimated/len(cohort_with_sofa):.1f}%)")
        print(f"  Mean: {cohort_with_sofa['cr_baseline'].mean():.3f} mg/dL")
        print(f"  Median: {cohort_with_sofa['cr_baseline'].median():.3f} mg/dL")

        print("\nHighest Creatinine (ICU stay only):")
        print(f"  Mean: {cohort_with_sofa['cr_highest_icu'].mean():.2f} mg/dL")
        print(f"  Median: {cohort_with_sofa['cr_highest_icu'].median():.2f} mg/dL")
        _n_missing_cr = cohort_with_sofa['cr_highest_icu'].isna().sum()
        print(f"  Missing: {_n_missing_cr:,} ({100*_n_missing_cr/len(cohort_with_sofa):.1f}%)")

    print("\n=== C. difficile Infection ===")
    if 'cdiff_positive' in cohort_with_sofa.columns:
        _cdiff_pos = (cohort_with_sofa['cdiff_positive'] == 1).sum()
        _cdiff_neg = (cohort_with_sofa['cdiff_positive'] == 0).sum()
        _cdiff_pct = 100 * _cdiff_pos / len(cohort_with_sofa)
        print(f"C. diff positive: {_cdiff_pos:,} ({_cdiff_pct:.2f}%)")
        print(f"C. diff negative: {_cdiff_neg:,} ({100 - _cdiff_pct:.2f}%)")

    # Show hospital_type distribution if available
    if 'hospital_type' in cohort_with_sofa.columns:
        print(f"\n=== Hospital Type Distribution ===")
        print(cohort_with_sofa['hospital_type'].value_counts())
    return


@app.cell
def _(mo):
    mo.md(r"""## Save Cohort to PHI_DATA""")
    return


@app.cell
def _(Path, cohort_with_sofa):
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
def _(cohort_with_sofa):
    cohort_with_sofa
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
