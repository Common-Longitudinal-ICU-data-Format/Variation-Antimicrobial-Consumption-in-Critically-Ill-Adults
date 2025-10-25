# Variation in Antimicrobial Consumption in Critically Ill Adults

## Project Overview

Antibiotics are potentially life-saving medications, but their inappropriate use is leading to alarming emergence of resistant organisms and deaths from antibiotic resistance. Although administration of antibiotics in the ICU is pivotal in the management of severe infections and sepsis, antibiotic guidelines differ between ICUs across institutions depending on patient factors, resistance patterns, accessibility, and clinical practice patterns.

This multi-site project aims to characterize the heterogeneity in antibiotic consumption across a diverse set of US medical healthcare centers using the CLIF (Critical Care Learning from Intensive Care and Feeding) consortium data.

### Aim

To evaluate the between-hospital variation in prescription of antibiotics in critically ill adults.

### Data Sources

**Database**: CLIF

**CLIF Sites**: No restriction

**Study Period**: 2018-2024

## Setup

### 1. Install UV Package Manager

**Mac/Linux**:

``` bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows**:

``` powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install Dependencies

``` bash
uv sync
```

### 3. Configure Site

Create/update `clif_config.json` (rename \_template.json) with your site-specific configuration:

``` json
{
    "site": "your_site_name",
    "data_directory": "/path/to/your/clif/data",
    "filetype": "parquet",
    "timezone": "US/Central"
}
```

### 4. Prepare Reference Files

This file contains: - `Antibiotic`: Antibiotic names matching CLIF `med_category` values - `Score`: Spectrum coverage scores for each antibiotic

## Required CLIF Tables

| Table | Columns | Categories/Filters |
|------------------|---------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| **adt** | hospitalization_id, location_category, location_type, in_dttm, out_dttm | location_category == 'icu' |
| **hospitalization** | patient_id, hospitalization_id, age_at_admission, admission_dttm, discharge_dttm, discharge_category | \- |
| **patient** | patient_id, sex_category, ethnicity_category, race_category, language_category, death_dttm | \- |
| **vitals** | hospitalization_id, recorded_dttm, vital_category, vital_value | temp_c, map, spo2, weight_kg |
| **labs** | hospitalization_id, lab_result_dttm, lab_category, lab_value, lab_value_numeric | wbc, creatinine, platelet_count, po2_arterial, bilirubin_total |
| **patient_assessments** | hospitalization_id, recorded_dttm, assessment_category, numerical_value | gcs_total |
| **medication_admin_continuous** | All columns (including med_dose, med_dose_unit) | norepinephrine, epinephrine, phenylephrine, angiotensin, vasopressin, dopamine, dobutamine, milrinone, isoproterenol |
| **medication_admin_intermittent** | hospitalization_id, admin_dttm, med_category, med_route_category, med_dose, med_dose_unit | All antibiotics from antibiotic_spectrum_scoring.csv |
| **respiratory_support** | hospitalization_id, recorded_dttm, device_category, fio2_set | All device categories (NIPPV, High Flow NC, etc.) |

## Study Population

### Inclusion Criteria

-   Adults (≥18 years old)
-   First (index) ICU admission within `hospitalization_id` (`location_category == 'icu'`)
-   Medical ICU only (`location_type == 'medical_icu'`)
-   ICU admission between 2018-2024
-   ICU discharge ≤2024 (excludes partial 2025 data)
-   ICU length of stay \> 6 hours (0.25 days)

### Exclusion Criteria

-   ICU stays ≤6 hours (0.25 days)
-   ICU discharge dates in 2025

## Clinical Features Computed

All features are filtered to the ICU stay window (`start_dttm` to `end_dttm`):

### Vital Signs

-   `highest_temperature` (°C) - Maximum temperature during ICU stay
-   `lowest_temperature` (°C) - Minimum temperature during ICU stay
-   `lowest_map` (mmHg) - Minimum mean arterial pressure during ICU stay

### Laboratory Values

-   `highest_wbc` (10³/μL) - Maximum white blood cell count during ICU stay
-   `highest_creatinine` (mg/dL) - Maximum creatinine during ICU stay

### Respiratory Support

-   `NIPPV_ever` (binary) - Non-invasive positive pressure ventilation usage
-   `HFNO_ever` (binary) - High-flow nasal cannula usage

### Medications

-   `vasopressor_ever` (binary) - Any vasopressor usage during ICU stay
-   `no_of_vasopressor` (count) - Number of unique vasopressor categories used

### Severity Scores

-   SOFA scores - Computed using clifpy ClifOrchestrator
    -   Includes respiratory, cardiovascular, hepatic, coagulation, renal, and neurological components

### Outcomes

-   `hospital_los_days` - Hospital length of stay (days)
-   `icu_los_days` - ICU length of stay (days)
-   `inpatient_mortality` (binary) - Death during hospitalization
-   `icu_mortality` (binary) - Death during ICU stay

## Methodology

### Step 0: Create 24-Hour Windows

ICU "days" are defined as **24-hour windows starting from ICU admission time**, not calendar days:

-   **Window 0**: First 24 hours after ICU admission (ICU Day 0)
-   **Window 1**: Second 24 hours (ICU Day 1)
-   **Window N**: Nth 24-hour period

Each window represents **1 patient-day**.

**Example**: Patient admitted to ICU on 2024-01-15 at 08:00, discharged 2024-01-17 at 14:00 - Window 0: 2024-01-15 08:00 → 2024-01-16 08:00 (24 hours) - Window 1: 2024-01-16 08:00 → 2024-01-17 08:00 (24 hours) - Window 2: 2024-01-17 08:00 → 2024-01-17 14:00 (6 hours, capped at discharge) - **Total**: 3 windows = 3 patient-days

### Step 1: Calculate DOT per Antibiotic

For each `hospitalization_id` and each antibiotic:

1.  Count the number of 24-hour windows containing ≥1 dose of that antibiotic
2.  If antibiotic is administered at any time during a window → count as **1 DOT** for that antibiotic
3.  Track **antibiotic-free days** (windows with zero antibiotic doses)

**Formula**: DOT = Number of windows with ≥1 dose of antibiotic

**Output**: Patient-level table with DOT per antibiotic per hospitalization

### Step 2: Calculate Patient-Days (PD)

Calculate the total number of patient-days across all hospitalizations.

**Formula**: `PD = Σ (number of 24-hour windows per hospitalization)`

**Example**: If cohort has 10,000 hospitalizations with average ICU LOS of 5 days → PD = 50,000

### Step 3: Calculate DOT per 1000 PD by Antibiotic

For each individual antibiotic, calculate standardized consumption rate.

**Formula**: `DOT per 1000 PD = (Total DOT for antibiotic / Total PD) × 1000`

**Example**: - Vancomycin total DOT = 5,000 days (across all hospitalizations) - Total PD = 50,000 patient-days - Vancomycin DOT per 1000 PD = (5,000 / 50,000) × 1000 = **100**

**Interpretation**: For every 1000 patient-days in the ICU, Vancomycin is administered on 100 days.

**Output**: Antibiotic-level metrics table (one row per antibiotic)

### Step 4: Calculate Overall DOT per 1000 PD (Cohort-Level)

Calculate overall antibiotic consumption across **ALL** antibiotics for the MICU cohort.

**Formula**: `Overall DOT per 1000 PD = (Total DOT all antibiotics / Total PD) × 1000`

**Difference from Step 3**: Step 3 calculates per individual antibiotic; Step 4 aggregates across all antibiotics.

**Note**: Since the cohort is restricted to Medical ICU only, location-type stratification is not performed.

**Output**: Cohort-level overall metrics

### Step 5: Calculate Daily ASC per Window

For each 24-hour window (ICU Days 0-10), calculate **Antibiotic Spectrum Coverage (ASC)**.

**Formula**: `Daily ASC = Σ (spectrum_score for each antibiotic used in window)`

**IMPORTANT**: This is **NOT** `DOT × spectrum_score`. The formula sums spectrum scores of all antibiotics used in a window.

**Example**: - **Window 0 (ICU Day 0)**: Vancomycin (score=3.5) + Piperacillin-Tazobactam (score=4.2) - Daily ASC = 3.5 + 4.2 = **7.7** - **Window 1 (ICU Day 1)**: Vancomycin only (score=3.5) - Daily ASC = **3.5** - **Window 2 (ICU Day 2)**: No antibiotics - Daily ASC = **0**

**Output**: Mean and standard deviation of daily ASC for windows 0-10 (for multi-site comparison)

### Step 6: Calculate DASC per 1000 PD

Calculate **Days of Antibiotic Spectrum Coverage (DASC)** per 1000 patient-days.

#### Process:

1.  For each hospitalization, sum daily ASC across **ALL** ICU windows → **DASC**
2.  Calculate DASC per 1000 PD

**Formula**: `DASC per 1000 PD = (Total DASC across all hospitalizations / Total PD) × 1000`

**Example**: - Hospitalization with 5 ICU days: - Window 0: Daily ASC = 7.5 - Window 1: Daily ASC = 7.5 - Window 2: Daily ASC = 3.5 - Window 3: Daily ASC = 3.5 - Window 4: Daily ASC = 0 - **DASC** = 7.5 + 7.5 + 3.5 + 3.5 + 0 = **22.0**

**Output**: - **Overall**: DASC per 1000 PD for entire cohort (all years combined) - **By Year**: DASC per 1000 PD for each year (2018-2024) for temporal trend analysis

### Step 7: Calculate Antibiotic-Free Days (AFD)

Count the number of 24-hour windows without any antibiotic administration.

**Formula**: `AFD Rate = (Antibiotic-free windows / Total windows) per hospitalization`

**Example**: - Hospitalization with 5 ICU days: - Day 0: Antibiotics given - Day 1: Antibiotics given - Day 2: **No antibiotics** (antibiotic-free) - Day 3: **No antibiotics** (antibiotic-free) - Day 4: Antibiotics given - **AFD Rate** = 2 / 5 = **0.40 (40%)**

**Output**: Mean and standard deviation of AFD rate across all hospitalizations

### Step 8: Calculate Year-Based ASC Summary

Group daily ASC data by year (based on window start timestamp) to enable time series analysis.

**Process**: 1. Join daily ASC data with window timestamps 2. Extract year from `window_start` 3. Calculate summary statistics per year: mean, SD, median, min, max

**Output**: Year-based ASC summary table (2018-2024)

### Step 9: Generate Visualizations

Create publication-ready plots for multi-site comparison:

1.  **ASC by Year Plot**: Smoothed spline with ±1 SD band showing temporal trends
2.  **ASC by ICU Day Plot**: Smoothed spline with ±1 SD band for Days 0-10 post-admission

**Output**: PNG files saved to `RESULTS_UPLOAD_ME/`

### Step 10: Generate Table 1 Summary Statistics

Generate comprehensive summary statistics table including:

-   **Demographics**: Age, sex, race/ethnicity, BMI, ICU/hospital LOS
-   **Clinical Characteristics**: Vitals, SOFA scores, interventions (vasopressors, NIPPV, HFNO), labs
-   **Antibiotic Metrics**: Top 15 antibiotics by DOT per 1000 PD, overall DOT, daily ASC scores (Days 0-10), DASC per 1000 PD, AFD rate
-   **Outcomes**: Total patient-days, mortality rates

**Output**: - `table1_summary.json` (machine-readable for aggregation) - `table1_summary.csv` (human-readable for review)

## Key Metrics Definitions

| Metric | Abbreviation | Definition |
|------------------------------------------------------|------------------------------------|-------------------------------------------------------------------------------|
| Days of Therapy | DOT | Number of 24-hour windows where patient receives ≥1 dose of antibiotic |
| Patient-Days | PD | Total number of 24-hour windows across all hospitalizations |
| Antibiotic Spectrum Coverage | ASC | Sum of spectrum scores for all antibiotics used in a 24-hour window |
| Days of Antibiotic Spectrum Coverage | DASC | Sum of daily ASC values across all ICU windows for a hospitalization |
| Antibiotic-Free Days | AFD | Number of 24-hour windows without any antibiotic administration |
| SOFA Score | SOFA | Sequential Organ Failure Assessment score (computed via clifpy ClifOrchestrator) |

## Execution Guide

### Step 1: Generate ICU Cohort

This step creates the ICU cohort with all clinical features and SOFA scores.

``` bash
uv run code/01_cohort.py
```

### Step 2: Calculate DOT and Antibiotic Metrics

This step performs all DOT, ASC, DASC, and AFD calculations, generates visualizations, and creates Table 1.

``` bash
uv run code/02_DOT.py
```

## Output Files

### PHI_DATA/ (Patient-Level - DO NOT SHARE)

These files contain patient-level data and should **NOT** be shared outside your institution:

-   `cohort_icu_first_stay.parquet` - Complete cohort with all features and SOFA scores
-   `dot_hospital_level.parquet` - DOT per antibiotic per hospitalization (wide format)
-   `daily_asc_patient_level.parquet` - Daily ASC for each window per hospitalization
-   `afd_patient_level.parquet` - Antibiotic-free days per hospitalization

### RESULTS_UPLOAD_ME/ (Safe to Share - Summary Statistics)

These files contain only aggregate summary statistics and are **safe to share** for multi-site comparison:

-   `dot_antibiotic_level.csv` - DOT per 1000 PD by antibiotic (cohort-level)
-   `dot_cohort_level.csv` - Overall DOT metrics (all antibiotics combined)
-   `daily_asc_summary.csv` - Mean/SD of daily ASC for Days 0-10 (for plotting)
-   `dasc_overall.csv` - DASC per 1000 PD (all years combined)
-   `dasc_by_year.csv` - DASC per 1000 PD by year (2018-2024)
-   `asc_by_year_summary.csv` - Year-based ASC trends for time series analysis
-   `afd_summary.csv` - Antibiotic-free days summary statistics
-   `table1_summary.json` - Complete Table 1 in JSON format (machine-readable)
-   `table1_summary.csv` - Complete Table 1 in CSV format (human-readable)
-   `asc_by_year_plot.png` - Visualization of ASC trends over years
-   `asc_by_window_plot.png` - Visualization of ASC by ICU day (Days 0-10)

## Data Sharing Instructions

### After Successful Pipeline Completion

Once both scripts (`01_cohort.py` and `02_DOT.py`) have completed successfully and all output files are generated in `RESULTS_UPLOAD_ME/`, contact the RUSH coordinating site to obtain the BOX folder link for uploading your results.

**Contact**: [Vaishvik_Chaudhari\@rush.edu](Vaishvik_Chaudhari@rush.edu)

### What to Upload

**IMPORTANT**: Upload **ONLY** files from the `RESULTS_UPLOAD_ME/` folder. **DO NOT** upload files from `PHI_DATA/` folder as they contain patient-level data.

**Upload Checklist**: - \[ \] `dot_antibiotic_level.csv` - \[ \] `dot_cohort_level.csv` - \[ \] `daily_asc_summary.csv` - \[ \] `dasc_overall.csv` - \[ \] `dasc_by_year.csv` - \[ \] `asc_by_year_summary.csv` - \[ \] `afd_summary.csv` - \[ \] `table1_summary.json` - \[ \] `table1_summary.csv` - \[ \] `asc_by_year_plot.png` - \[ \] `asc_by_window_plot.png`