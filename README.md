# Variation in Antimicrobial Consumption in Critically Ill Adults

## Introduction

Antibiotics are a potentially life-saving medication, but their inappropriate use is leading to alarming emergence of resistant organisms and deaths from antibiotic resistance. Although administration of antibiotics in the ICU are pivotal in the management of severe infections and sepsis, antibiotic guidelines differ between ICUs across institutions depending on patient factors, resistance patterns, accessibility, and clinical practice patterns.

This project aims to characterize the heterogeneity in antibiotic consumption across a diverse set of US medical healthcare centers.

## Aim

To evaluate the between-hospital variation in prescription of antibiotics in critically ill adults.

## Data Sources

**Database**: CLIF (Critical Care Learning from Intensive Care and Feeding)

**CLIF Sites**: No restriction

**Required Tables**: - `adt` - `hospitalization` - `crrt` - `labs` - `medication_admin_intermittent` - `patient` - `respiratory_support`

## Study Population

### Inclusion Criteria

-   Adult patients aged 18 years or older
-   First (index) ICU admission within `hospitalization_id` (`location_category == icu`)
-   Years: 2018-2024
-   ICU data only, first stay

### Exclusion Criteria

-   None

### Censoring

Last row of hospitalization in ICU (with exception of outcomes which need full hospitalization). Analysis is grouped by `hospitalization_id` (not `patient_id`).

## Research Objectives

1.  **Objective 1**: Describe trends in antibiotic consumption across an ICU stay (ICU day 0 to 10) and evaluate differences by healthcare center

2.  **Objective 2**: Describe trends in antibiotic consumption across years (2018-2024) and evaluate differences by healthcare center

## Methodology

### Step 0: Calculate Days of Therapy (DOT) for Each Antibiotic

For each `hospitalization_id` that meets inclusion criteria and has no exclusion criteria, calculate DOT for **each antibiotic** listed in the reference Excel file (use antibiotic names only; ASC scores applied in later steps).

### Step 1: Calculate Patient-Days (PD)

Calculate PD (Days Present) as the aggregate number of patients housed in ICU anytime throughout a day.

**Formula**: PD = Total number of patient-days across all hospitalizations

### Step 2: Calculate DOT per 1000 Patient-Days by Antibiotic

Calculate the standardized antibiotic consumption rate for each individual antibiotic.

**Formula**: DOT per 1000 PD = (Total DOT for antibiotic / Total PD) × 1000

**Output**: Metric per healthcare system for each individual antibiotic

### Step 3: Calculate Total DOT per Hospitalization

Sum all DOT across all antibiotics for each `hospitalization_id`.

**Formula**: Total DOT = Σ (DOT for each antibiotic) per hospitalization

### Step 4: Calculate Overall DOT per 1000 Patient-Days

Calculate the overall antibiotic consumption rate across all antibiotics.

**Formula**: DOT per 1000 PD = (Total DOT / Total PD) × 1000

**Output**: Metric per healthcare system for all antibiotics used in ICU

### Step 5: Calculate Daily Antibiotic Spectrum Coverage (ASC)

Calculate the ASC for each day post-ICU admission (days 0 → 10) using the sum of ASC scores (from reference Excel file) of all antibiotics used in a day.

**Formula**: Daily ASC = Σ (DOT × spectrum score for each antibiotic) per day

**Output**: Mean and standard deviation per ICU days 0 to 10

### Step 6: Calculate DASC per 1000 Patient-Days

Calculate DASC (Days of ASC) per 1000 PD to quantify antibiotic spectrum coverage relative to patient-days, scaled per 1000 for comparability between hospitals.

**Process**: 1. Calculate the sum of ASC scores for all days for each `hospitalization_id` in the ICU 2. Calculate DASC per 1000 PD

**Formula**: DASC per 1000 PD = (DASC / PD) × 1000

**Output**: - a) Metric per healthcare system for all years of data - b) Metric per healthcare system for each individual year

### Step 7: Calculate Antibiotic-Free Days (AFD)

Count the number of calendar days that each `hospitalization_id` does not have antibiotics.

**Formula**: AFD rate = (Days without antibiotics / Days in ICU) per hospitalization

**Output**: Mean and standard deviation of AFD rate

## Key Metrics Definitions

| Metric | Abbreviation | Definition |
|------------------|-----------------------------|-------------------------|
| Days of Therapy | DOT | Number of days a patient receives a specific antibiotic |
| Patient-Days | PD | Aggregate number of patients housed in ICU throughout a day |
| Antibiotic Spectrum Coverage | ASC | Sum of spectrum scores for all antibiotics used on a given day |
| Days of Antibiotic Spectrum Coverage | DASC | Sum of ASC scores across all ICU days for a hospitalization |
| Antibiotic-Free Days | AFD | Number of calendar days without antibiotic administration |

------------------------------------------------------------------------

**Note**: Antibiotic names and corresponding ASC scores are maintained in a separate reference Excel file.