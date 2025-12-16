"""
ASE.py - Adult Sepsis Event Detection Module

Implements the CDC Adult Sepsis Event (ASE) surveillance definition using:
- clifpy for CLIF data loading and table access
- duckdb for efficient SQL-based data processing

References:
    CDC Hospital Toolkit for Adult Sepsis Surveillance (March 2018)
    https://www.cdc.gov/sepsis/pdfs/sepsis-surveillance-toolkit-mar-2018_508.pdf

CDC ASE Definition (Page 5):
    "ASE: Adult Sepsis Event
    (Must include the 2 components of criteria A AND include one or more
    organ dysfunction listed among B criteria)

    A. Presumed Infection (presence of both 1 and 2):
       1. Blood culture obtained (irrespective of the result), AND
       2. At least 4 Qualifying Antimicrobial Days (QAD)

    B. Organ Dysfunction (at least 1 of following criteria met within ±2 days
       of blood culture):
       1. Initiation of a new vasopressor infusion
       2. Initiation of invasive mechanical ventilation
       3. Doubling of serum creatinine (excluding ESRD)
       4. Total bilirubin ≥2.0 mg/dL and increase by 100% from baseline
       5. Platelet count <100 AND ≥50% decline from baseline
       6. Optional: Serum lactate ≥2.0 mmol/L"

Validation Reference (Page 4):
    "This definition was validated by Rhee, et al. and shown to be present
    in 6% of hospital admissions in a study of nearly 400 hospitals."
"""

import duckdb
import pandas as pd
from typing import List, Optional

# clifpy imports
from clifpy.tables import (
    Hospitalization,
    Patient,
    Labs,
    MedicationAdminContinuous,
    MedicationAdminIntermittent,
    MicrobiologyCulture,
    RespiratorySupport,
    HospitalDiagnosis,
)

# =============================================================================
# Constants
# =============================================================================

# CDC Page 15 (Appendix B): Vasopressors Included in Adult Sepsis Event Definition
# "Eligible vasopressors must have been administered via continuous intravenous
#  infusion. Vasopressors administered in an operating room are excluded."
VASOPRESSORS = [
    "norepinephrine",
    "dopamine",
    "epinephrine",
    "phenylephrine",
    "vasopressin",
]

# CDC Page 5: "excluding patients with ICD-10 code for end-stage renal disease (N18.6)"
ESRD_ICD10 = "N18.6"

# CDC Page 6: "window period extending both 2 days before and 2 days after the blood culture"
WINDOW_DAYS = 2  # ±2 days from blood culture

# CDC Page 6-7: QAD window
# "the first QAD is the first day in window period extending both 2 days before
#  and 2 days after the patient receives a new antimicrobial"
QAD_WINDOW_START = -2  # days relative to blood culture
QAD_WINDOW_END = 6  # days relative to blood culture

# CDC Page 5: "At least 4 Qualifying Antimicrobial Days (QAD)"
MIN_QAD = 4

# Outlier thresholds for lab values
OUTLIERS = {
    "creatinine_max": 20,
    "bilirubin_max": 80,
    "platelet_max": 2000,
    "lactate_max": 30,
}


# =============================================================================
# Data Loading Functions
# =============================================================================


def _load_clif_tables(
    hospitalization_ids: List[str],
    config_path: str = "clif_config.json",
) -> duckdb.DuckDBPyConnection:
    """
    Load CLIF tables into DuckDB for efficient querying.

    Parameters
    ----------
    hospitalization_ids : List[str]
        List of hospitalization IDs to filter data
    config_path : str
        Path to clifpy config file

    Returns
    -------
    duckdb.DuckDBPyConnection
        DuckDB connection with registered tables
    """
    con = duckdb.connect(":memory:")

    # Load hospitalization table
    hosp_table = Hospitalization.from_file(
        config_path=config_path,
        filters={"hospitalization_id": hospitalization_ids},
    )
    con.register("hospitalization", hosp_table.df)

    # Load patient table - get patient_ids from hospitalization
    patient_ids = hosp_table.df["patient_id"].unique().tolist()
    patient_table = Patient.from_file(
        config_path=config_path,
        filters={"patient_id": patient_ids},
    )
    con.register("patient", patient_table.df)

    # Load labs table
    labs_table = Labs.from_file(
        config_path=config_path,
        filters={"hospitalization_id": hospitalization_ids},
        columns=[
            "hospitalization_id",
            "lab_category",
            "lab_value",
            "lab_value_numeric",
            "lab_result_dttm",
            "lab_order_dttm",
        ],
    )
    con.register("labs", labs_table.df)

    # Load microbiology culture table
    micro_table = MicrobiologyCulture.from_file(
        config_path=config_path,
        filters={"hospitalization_id": hospitalization_ids},
    )
    con.register("microbiology", micro_table.df)

    # Load continuous medication table (for vasopressors)
    med_cont_table = MedicationAdminContinuous.from_file(
        config_path=config_path,
        filters={"hospitalization_id": hospitalization_ids},
    )
    con.register("med_continuous", med_cont_table.df)

    # Load intermittent medication table (for antibiotics/QAD)
    med_int_table = MedicationAdminIntermittent.from_file(
        config_path=config_path,
        filters={"hospitalization_id": hospitalization_ids},
    )
    con.register("med_intermittent", med_int_table.df)

    # Load respiratory support table (for IMV)
    resp_table = RespiratorySupport.from_file(
        config_path=config_path,
        filters={"hospitalization_id": hospitalization_ids},
    )
    con.register("respiratory", resp_table.df)

    # Load hospital diagnosis table (for ESRD)
    dx_table = HospitalDiagnosis.from_file(
        config_path=config_path,
        filters={"hospitalization_id": hospitalization_ids},
        columns=["hospitalization_id", "diagnosis_code", "diagnosis_code_format"],
    )
    con.register("diagnosis", dx_table.df)

    return con


# =============================================================================
# Blood Culture Functions
# =============================================================================


def _get_blood_cultures(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Get blood cultures and identify the earliest blood culture per hospitalization.

    Returns earliest blood culture regardless of admission timing (full hospitalization).

    CDC Definition (Page 6):
        "Qualifying cultures include those drawn for bacterial (aerobic and/or
        anaerobic), acid-fast bacilli (AFB), and fungal cultures. Blood cultures
        for specific viruses (e.g., cytomegalovirus) are excluded. For ASE, blood
        cultures merely need to have been drawn, regardless of result."

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: hospitalization_id, blood_culture_dttm, admission_dttm
    """
    return con.execute("""
        WITH bc AS (
            SELECT
                m.hospitalization_id,
                m.order_dttm as blood_culture_dttm,
                h.admission_dttm,
                h.discharge_dttm,
                -- Calculate hospital day (admission = day 1)
                DATEDIFF('day', DATE(h.admission_dttm), DATE(m.order_dttm)) + 1 as bc_hospital_day
            FROM microbiology m
            JOIN hospitalization h USING (hospitalization_id)
            WHERE m.fluid_category = 'blood_buffy'
        )
        SELECT
            hospitalization_id,
            MIN(blood_culture_dttm) as blood_culture_dttm,
            admission_dttm,
            discharge_dttm
        FROM bc
        GROUP BY hospitalization_id, admission_dttm, discharge_dttm
    """).df()


# =============================================================================
# QAD (Qualifying Antimicrobial Days) Functions
# =============================================================================


def _calculate_qad(
    con: duckdb.DuckDBPyConnection,
    blood_cultures: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate consecutive qualifying antimicrobial days (QAD).

    CDC Definition (Page 6-7):
        "For ASE events, the first QAD is the first day in window period extending
        both 2 days before and 2 days after the patient receives a new antimicrobial.
        A new antimicrobial is defined as an antimicrobial not previously administered
        in the prior 2 calendar days."

        "There must be at least one new parenteral (intravenous or intramuscular)
        antimicrobial administered within the window period for the QADs to satisfy
        the definition."

        "A gap of a single calendar day between administrations of the same antibiotic
        (oral or intravenous) count as QADs as long as the gap is not greater than 1 day."

    CDC QAD Censoring (Page 8):
        "If a patient's care transitions to comfort measures only, or the patient dies,
        is discharged to another hospital, or discharged to hospice before 4 QADs have
        elapsed, then the presumed infection criteria can be met with less than 4 QADs
        as long as they have consecutive QADs until day of, or 1 day prior to, death
        or discharge."

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        DuckDB connection with loaded tables
    blood_cultures : pd.DataFrame
        Blood culture data from _get_blood_cultures()

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: hospitalization_id, total_qad, first_qad_dttm
    """
    # Register blood cultures as temp table
    con.register("blood_cultures_temp", blood_cultures)

    return con.execute("""
        WITH bc_hosp AS (
            SELECT * FROM blood_cultures_temp
        ),
        abx_admin AS (
            -- Get qualifying antimicrobial administrations
            SELECT
                m.hospitalization_id,
                m.admin_dttm,
                m.med_name,
                m.med_route_category,
                DATE(m.admin_dttm) as admin_date,
                bc.blood_culture_dttm,
                -- Calculate day relative to blood culture
                DATEDIFF('day', DATE(bc.blood_culture_dttm), DATE(m.admin_dttm)) as day_from_bc
            FROM med_intermittent m
            JOIN bc_hosp bc USING (hospitalization_id)
            WHERE m.med_group = 'CMS_sepsis_qualifying_antibiotics'
              AND DATEDIFF('day', DATE(bc.blood_culture_dttm), DATE(m.admin_dttm)) BETWEEN -2 AND 6
        ),
        daily_abx AS (
            -- Aggregate to daily level - one row per hospitalization-date
            SELECT
                hospitalization_id,
                admin_date,
                day_from_bc,
                blood_culture_dttm,
                MAX(CASE WHEN med_route_category = 'iv' THEN 1 ELSE 0 END) as has_iv,
                MIN(admin_dttm) as first_admin_of_day
            FROM abx_admin
            GROUP BY hospitalization_id, admin_date, day_from_bc, blood_culture_dttm
        ),
        -- Check if there's at least one IV antibiotic in the window
        iv_check AS (
            SELECT
                hospitalization_id,
                MAX(has_iv) as has_iv_in_window
            FROM daily_abx
            GROUP BY hospitalization_id
        ),
        consecutive_runs AS (
            -- Find consecutive day runs using gaps-and-islands technique
            SELECT
                d.hospitalization_id,
                d.admin_date,
                d.day_from_bc,
                d.first_admin_of_day,
                d.day_from_bc - ROW_NUMBER() OVER (
                    PARTITION BY d.hospitalization_id
                    ORDER BY d.day_from_bc
                ) as run_group
            FROM daily_abx d
            JOIN iv_check ic USING (hospitalization_id)
            WHERE ic.has_iv_in_window = 1
        ),
        run_lengths AS (
            SELECT
                hospitalization_id,
                run_group,
                COUNT(*) as run_length,
                MIN(admin_date) as first_qad_date,
                MIN(first_admin_of_day) as first_qad_dttm
            FROM consecutive_runs
            GROUP BY hospitalization_id, run_group
        ),
        best_runs AS (
            -- Get the longest run for each hospitalization
            SELECT
                hospitalization_id,
                MAX(run_length) as total_qad,
                FIRST(first_qad_dttm ORDER BY run_length DESC, first_qad_dttm) as first_qad_dttm
            FROM run_lengths
            GROUP BY hospitalization_id
        )
        SELECT
            hospitalization_id,
            total_qad,
            first_qad_dttm
        FROM best_runs
    """).df()


# =============================================================================
# ESRD Detection
# =============================================================================


def _get_esrd_flags(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Identify patients with End-Stage Renal Disease (ICD-10: N18.6).

    CDC Definition (Page 5):
        "Doubling of serum creatinine... excluding patients with ICD-10 code
        for end-stage renal disease (N18.6)."

    These patients are excluded from AKI organ dysfunction criteria.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: hospitalization_id, esrd
    """
    return con.execute("""
        SELECT DISTINCT
            hospitalization_id,
            1 as esrd
        FROM diagnosis
        WHERE diagnosis_code = 'N18.6'
           OR diagnosis_code LIKE 'N18.6%'
    """).df()


# =============================================================================
# Organ Dysfunction - Vasopressors
# =============================================================================


def _get_vasopressor_dysfunction(
    con: duckdb.DuckDBPyConnection,
    blood_cultures: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify new vasopressor initiation within ±2 days of blood culture.

    CDC Definition (Page 5):
        "Initiation of a new vasopressor infusion (norepinephrine, dopamine,
        epinephrine, phenylephrine, OR vasopressin). To count as a new vasopressor,
        that specific vasopressor cannot have been administered in the prior
        calendar day."

    CDC Appendix B (Page 15):
        "Eligible vasopressors must have been administered via continuous intravenous
        infusion. Vasopressors administered in an operating room are excluded as these
        are frequently needed to counteract hypotension induced by sedative medication
        administration. Since the location of administration may be challenging to
        identify in an EHR, single bolus injections of vasopressors (a frequent method
        of delivering perioperative vasopressors) are generally excluded."

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: hospitalization_id, vasopressor_dttm
    """
    # Register blood cultures as temp table
    con.register("blood_cultures_temp", blood_cultures)

    return con.execute("""
        WITH bc_hosp AS (
            SELECT * FROM blood_cultures_temp
        ),
        vaso_admin AS (
            SELECT
                m.hospitalization_id,
                m.admin_dttm,
                m.med_name,
                m.med_category,
                DATE(m.admin_dttm) as admin_date,
                bc.blood_culture_dttm,
                LAG(DATE(m.admin_dttm)) OVER (
                    PARTITION BY m.hospitalization_id, m.med_category
                    ORDER BY m.admin_dttm
                ) as prev_admin_date
            FROM med_continuous m
            JOIN bc_hosp bc USING (hospitalization_id)
            WHERE m.med_group = 'vasoactives'
              AND m.med_dose > 0
        ),
        new_vaso_in_window AS (
            -- Only count new vasopressors (not given in prior day) within ±2 days of BC
            SELECT *
            FROM vaso_admin
            WHERE (prev_admin_date IS NULL OR DATEDIFF('day', prev_admin_date, admin_date) > 1)
              AND admin_dttm BETWEEN
                  blood_culture_dttm - INTERVAL '2 days'
                  AND blood_culture_dttm + INTERVAL '2 days'
        )
        SELECT
            hospitalization_id,
            MIN(admin_dttm) as vasopressor_dttm
        FROM new_vaso_in_window
        GROUP BY hospitalization_id
    """).df()


# =============================================================================
# Organ Dysfunction - IMV
# =============================================================================


def _get_imv_dysfunction(
    con: duckdb.DuckDBPyConnection,
    blood_cultures: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify new invasive mechanical ventilation within ±2 days of blood culture.

    CDC Definition (Page 5):
        "Initiation of invasive mechanical ventilation (must be greater than 1
        calendar day between mechanical ventilation episodes)."

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: hospitalization_id, imv_dttm
    """
    # Register blood cultures as temp table
    con.register("blood_cultures_temp", blood_cultures)

    return con.execute("""
        WITH bc_hosp AS (
            SELECT * FROM blood_cultures_temp
        ),
        imv_episodes AS (
            SELECT
                r.hospitalization_id,
                r.recorded_dttm,
                DATE(r.recorded_dttm) as imv_date,
                bc.blood_culture_dttm,
                LAG(DATE(r.recorded_dttm)) OVER (
                    PARTITION BY r.hospitalization_id
                    ORDER BY r.recorded_dttm
                ) as prev_imv_date
            FROM respiratory r
            JOIN bc_hosp bc USING (hospitalization_id)
            WHERE r.device_category = 'IMV'
        ),
        new_imv_in_window AS (
            -- Only count new IMV (>1 day gap from previous) within ±2 days of BC
            SELECT *
            FROM imv_episodes
            WHERE (prev_imv_date IS NULL OR DATEDIFF('day', prev_imv_date, imv_date) > 1)
              AND recorded_dttm BETWEEN
                  blood_culture_dttm - INTERVAL '2 days'
                  AND blood_culture_dttm + INTERVAL '2 days'
        )
        SELECT
            hospitalization_id,
            MIN(recorded_dttm) as imv_dttm
        FROM new_imv_in_window
        GROUP BY hospitalization_id
    """).df()


# =============================================================================
# Organ Dysfunction - Lab Criteria
# =============================================================================


def _get_lab_dysfunction(
    con: duckdb.DuckDBPyConnection,
    blood_cultures: pd.DataFrame,
    esrd_flags: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate lab-based organ dysfunction criteria.

    CDC Definitions (Page 5):
        AKI: "Doubling of serum creatinine OR decrease by ≥50% of estimated
             glomerular filtration rate (eGFR) relative to baseline, excluding
             patients with ICD-10 code for end-stage renal disease (N18.6)."

        Hyperbilirubinemia: "Total bilirubin ≥2.0 mg/dL and increase by 100%
                           from baseline."

        Thrombocytopenia: "Platelet count <100 cells/µL AND ≥50% decline from
                         baseline - baseline must be ≥100 cells/µL."

        Lactate: "Optional: Serum lactate ≥2.0 mmol/L, note that serum lactate
                 has become an increasingly common test to measure tissue perfusion.
                 When serum lactate is included in the surveillance definition,
                 the likely effect will be to slightly increase the number of
                 sepsis cases identified."

    CDC Baseline Definitions (Page 9):
        Community-Onset Events:
        - Creatinine baseline: lowest value during hospitalization
        - Bilirubin baseline: lowest value during hospitalization
        - Platelet baseline: highest value during hospitalization (must be ≥100)

        Hospital-Onset Events:
        - Creatinine baseline: lowest value within ±2 days of blood culture
        - Bilirubin baseline: lowest value within ±2 days of blood culture
        - Platelet baseline: highest value within ±2 days of blood culture (must be ≥100)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: hospitalization_id, aki_dttm, hyperbilirubinemia_dttm,
        thrombocytopenia_dttm, lactate_dttm
    """
    # Register temp tables
    con.register("blood_cultures_temp", blood_cultures)
    con.register("esrd_temp", esrd_flags)

    return con.execute(f"""
        WITH bc_hosp AS (
            SELECT * FROM blood_cultures_temp
        ),
        labs_window AS (
            -- Labs within ±2 days of blood culture
            SELECT
                l.hospitalization_id,
                l.lab_category,
                l.lab_value_numeric as value,
                l.lab_result_dttm,
                bc.blood_culture_dttm,
                bc.admission_dttm,
                -- Hospital day of lab result
                DATEDIFF('day', DATE(bc.admission_dttm), DATE(l.lab_result_dttm)) + 1 as lab_hospital_day,
                -- Hospital day of blood culture (to determine onset type)
                DATEDIFF('day', DATE(bc.admission_dttm), DATE(bc.blood_culture_dttm)) + 1 as bc_hospital_day
            FROM labs l
            JOIN bc_hosp bc USING (hospitalization_id)
            WHERE l.lab_category IN ('creatinine', 'bilirubin_total', 'platelet_count', 'lactate')
              AND l.lab_result_dttm BETWEEN
                  bc.blood_culture_dttm - INTERVAL '2 days'
                  AND bc.blood_culture_dttm + INTERVAL '2 days'
              AND l.lab_value_numeric IS NOT NULL
        ),
        labs_all AS (
            -- All labs during hospitalization (for community-onset baseline)
            SELECT
                l.hospitalization_id,
                l.lab_category,
                l.lab_value_numeric as value,
                l.lab_result_dttm
            FROM labs l
            WHERE l.lab_category IN ('creatinine', 'bilirubin_total', 'platelet_count')
              AND l.lab_value_numeric IS NOT NULL
        ),
        -- Community-onset baselines (whole hospitalization)
        baseline_community AS (
            SELECT
                hospitalization_id,
                MIN(CASE WHEN lab_category = 'creatinine' AND value <= {OUTLIERS['creatinine_max']} THEN value END) as cr_baseline_co,
                MIN(CASE WHEN lab_category = 'bilirubin_total' AND value <= {OUTLIERS['bilirubin_max']} THEN value END) as bili_baseline_co,
                MAX(CASE WHEN lab_category = 'platelet_count' AND value <= {OUTLIERS['platelet_max']} AND value >= 100 THEN value END) as plt_baseline_co
            FROM labs_all
            GROUP BY hospitalization_id
        ),
        -- Hospital-onset baselines (within ±2 days of blood culture)
        baseline_hospital AS (
            SELECT
                hospitalization_id,
                MIN(CASE WHEN lab_category = 'creatinine' AND value <= {OUTLIERS['creatinine_max']} THEN value END) as cr_baseline_ho,
                MIN(CASE WHEN lab_category = 'bilirubin_total' AND value <= {OUTLIERS['bilirubin_max']} THEN value END) as bili_baseline_ho,
                MAX(CASE WHEN lab_category = 'platelet_count' AND value <= {OUTLIERS['platelet_max']} AND value >= 100 THEN value END) as plt_baseline_ho
            FROM labs_window
            GROUP BY hospitalization_id
        ),
        -- Get onset type per hospitalization
        onset_type AS (
            SELECT DISTINCT
                hospitalization_id,
                bc_hospital_day,
                CASE WHEN bc_hospital_day <= 2 THEN 'community' ELSE 'hospital' END as onset
            FROM labs_window
        ),
        -- AKI detection
        aki AS (
            SELECT
                lw.hospitalization_id,
                MIN(lw.lab_result_dttm) as aki_dttm
            FROM labs_window lw
            LEFT JOIN baseline_community bc ON lw.hospitalization_id = bc.hospitalization_id
            LEFT JOIN baseline_hospital bh ON lw.hospitalization_id = bh.hospitalization_id
            LEFT JOIN esrd_temp e ON lw.hospitalization_id = e.hospitalization_id
            LEFT JOIN onset_type ot ON lw.hospitalization_id = ot.hospitalization_id
            WHERE lw.lab_category = 'creatinine'
              AND lw.value <= {OUTLIERS['creatinine_max']}  -- outlier filter
              AND e.esrd IS NULL  -- exclude ESRD patients
              AND (
                  -- Use appropriate baseline based on onset type
                  (ot.onset = 'community' AND bc.cr_baseline_co IS NOT NULL AND lw.value >= 2.0 * bc.cr_baseline_co) OR
                  (ot.onset = 'hospital' AND bh.cr_baseline_ho IS NOT NULL AND lw.value >= 2.0 * bh.cr_baseline_ho)
              )
            GROUP BY lw.hospitalization_id
        ),
        -- Hyperbilirubinemia detection
        hyperbili AS (
            SELECT
                lw.hospitalization_id,
                MIN(lw.lab_result_dttm) as hyperbilirubinemia_dttm
            FROM labs_window lw
            LEFT JOIN baseline_community bc ON lw.hospitalization_id = bc.hospitalization_id
            LEFT JOIN baseline_hospital bh ON lw.hospitalization_id = bh.hospitalization_id
            LEFT JOIN onset_type ot ON lw.hospitalization_id = ot.hospitalization_id
            WHERE lw.lab_category = 'bilirubin_total'
              AND lw.value >= 2.0  -- Must be >=2.0 mg/dL
              AND lw.value <= {OUTLIERS['bilirubin_max']}  -- outlier filter
              AND (
                  (ot.onset = 'community' AND bc.bili_baseline_co IS NOT NULL AND lw.value >= 2.0 * bc.bili_baseline_co) OR
                  (ot.onset = 'hospital' AND bh.bili_baseline_ho IS NOT NULL AND lw.value >= 2.0 * bh.bili_baseline_ho)
              )
            GROUP BY lw.hospitalization_id
        ),
        -- Thrombocytopenia detection
        thrombocytopenia AS (
            SELECT
                lw.hospitalization_id,
                MIN(lw.lab_result_dttm) as thrombocytopenia_dttm
            FROM labs_window lw
            LEFT JOIN baseline_community bc ON lw.hospitalization_id = bc.hospitalization_id
            LEFT JOIN baseline_hospital bh ON lw.hospitalization_id = bh.hospitalization_id
            LEFT JOIN onset_type ot ON lw.hospitalization_id = ot.hospitalization_id
            WHERE lw.lab_category = 'platelet_count'
              AND lw.value < 100  -- Must be <100
              AND lw.value <= {OUTLIERS['platelet_max']}  -- outlier filter
              AND (
                  -- Baseline must be >=100 and current value must be <=50% of baseline
                  (ot.onset = 'community' AND bc.plt_baseline_co IS NOT NULL AND bc.plt_baseline_co >= 100 AND lw.value <= 0.5 * bc.plt_baseline_co) OR
                  (ot.onset = 'hospital' AND bh.plt_baseline_ho IS NOT NULL AND bh.plt_baseline_ho >= 100 AND lw.value <= 0.5 * bh.plt_baseline_ho)
              )
            GROUP BY lw.hospitalization_id
        ),
        -- Elevated lactate detection (no baseline required)
        lactate AS (
            SELECT
                lw.hospitalization_id,
                MIN(lw.lab_result_dttm) as lactate_dttm
            FROM labs_window lw
            WHERE lw.lab_category = 'lactate'
              AND lw.value >= 2.0  -- Must be >=2.0 mmol/L
              AND lw.value <= {OUTLIERS['lactate_max']}  -- outlier filter
            GROUP BY lw.hospitalization_id
        )
        -- Combine all lab dysfunction
        SELECT
            bc.hospitalization_id,
            aki.aki_dttm,
            hyperbili.hyperbilirubinemia_dttm,
            thrombocytopenia.thrombocytopenia_dttm,
            lactate.lactate_dttm
        FROM bc_hosp bc
        LEFT JOIN aki ON bc.hospitalization_id = aki.hospitalization_id
        LEFT JOIN hyperbili ON bc.hospitalization_id = hyperbili.hospitalization_id
        LEFT JOIN thrombocytopenia ON bc.hospitalization_id = thrombocytopenia.hospitalization_id
        LEFT JOIN lactate ON bc.hospitalization_id = lactate.hospitalization_id
    """).df()


# =============================================================================
# Presumed Infection Determination
# =============================================================================


def _determine_presumed_infection(
    con: duckdb.DuckDBPyConnection,
    blood_cultures: pd.DataFrame,
    qad_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Determine presumed infection status.

    CDC Definition - Criteria A (Page 5):
        "Presumed Infection (presence of both 1 and 2):
        1. Blood culture obtained (irrespective of the result), AND
        2. At least 4 Qualifying Antimicrobial Days (QAD)"

    CDC QAD Censoring Exception (Page 8):
        "If a patient's care transitions to comfort measures only, or the patient
        dies, is discharged to another hospital, or discharged to hospice before
        4 QADs have elapsed, then the presumed infection criteria can be met with
        less than 4 QADs as long as they have consecutive QADs until day of, or
        1 day prior to, death or discharge."

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: hospitalization_id, blood_culture_dttm, total_qad,
        first_qad_dttm, presumed_infection
    """
    # Register temp tables
    con.register("blood_cultures_temp", blood_cultures)
    con.register("qad_temp", qad_results)

    return con.execute("""
        WITH bc_hosp AS (
            SELECT * FROM blood_cultures_temp
        ),
        qad AS (
            SELECT * FROM qad_temp
        ),
        censoring AS (
            -- Get death/discharge info for censoring logic
            SELECT
                h.hospitalization_id,
                h.discharge_dttm,
                h.discharge_category,
                p.death_dttm,
                CASE
                    WHEN p.death_dttm IS NOT NULL THEN p.death_dttm
                    ELSE h.discharge_dttm
                END as censor_dttm,
                CASE
                    WHEN h.discharge_category IN ('expired', 'Expired', 'acute_care_hospital', 'Acute Care Hospital', 'hospice', 'Hospice')
                    THEN 1
                    ELSE 0
                END as qualifies_for_censoring
            FROM hospitalization h
            LEFT JOIN patient p USING (patient_id)
        )
        SELECT
            bc.hospitalization_id,
            bc.blood_culture_dttm,
            bc.admission_dttm,
            bc.discharge_dttm,
            COALESCE(qad.total_qad, 0) as total_qad,
            qad.first_qad_dttm,
            CASE
                -- Standard: >=4 QAD
                WHEN qad.total_qad >= 4 THEN 1
                -- Censored: >=1 QAD and patient died/transferred before completing 4 days
                WHEN qad.total_qad >= 1
                     AND c.qualifies_for_censoring = 1
                     AND c.censor_dttm IS NOT NULL
                     AND c.censor_dttm <= qad.first_qad_dttm + INTERVAL '1 day' * (4 - 1)
                THEN 1
                ELSE 0
            END as presumed_infection
        FROM bc_hosp bc
        LEFT JOIN qad ON bc.hospitalization_id = qad.hospitalization_id
        LEFT JOIN censoring c ON bc.hospitalization_id = c.hospitalization_id
    """).df()


# =============================================================================
# Final ASE Determination
# =============================================================================


def _calculate_final_ase(
    presumed_infection: pd.DataFrame,
    vasopressor_df: pd.DataFrame,
    imv_df: pd.DataFrame,
    lab_dysfunction: pd.DataFrame,
    esrd_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine all criteria to determine final ASE status.

    CDC ASE Definition (Page 5):
        "ASE: Adult Sepsis Event
        (Must include the 2 components of criteria A AND include one or more
        organ dysfunction listed among B criteria)"

    CDC Onset Type Classification (Page 8):
        "Hospital-Onset Events require onset date to be on hospital day 3 or later,
        counting the date of admission as hospital day 1."

        "Community-Onset Events require onset date to be on hospital day 2 or earlier,
        when the date of admission counts as hospital day 1."

    CDC Onset Date Definition (Page 8):
        "For ASE, onset date is defined as the earliest day in the window period
        extending both 2 days before and 2 days after the blood culture when EITHER
        the blood culture, first QAD, OR organ dysfunction criteria is met."

    Returns
    -------
    pd.DataFrame
        Final ASE results with all specified columns
    """
    # Merge all dataframes
    result = presumed_infection.copy()
    result = result.merge(vasopressor_df, on="hospitalization_id", how="left")
    result = result.merge(imv_df, on="hospitalization_id", how="left")
    result = result.merge(lab_dysfunction, on="hospitalization_id", how="left")
    result = result.merge(esrd_df, on="hospitalization_id", how="left")

    # Fill ESRD nulls with 0
    result["esrd"] = result["esrd"].fillna(0).astype(int)

    # Define organ dysfunction columns
    organ_cols_w_lactate = [
        "vasopressor_dttm",
        "imv_dttm",
        "aki_dttm",
        "hyperbilirubinemia_dttm",
        "thrombocytopenia_dttm",
        "lactate_dttm",
    ]
    organ_cols_wo_lactate = [
        "vasopressor_dttm",
        "imv_dttm",
        "aki_dttm",
        "hyperbilirubinemia_dttm",
        "thrombocytopenia_dttm",
    ]

    # Has any organ dysfunction
    result["has_organ_dysfunction_w_lactate"] = (
        result[organ_cols_w_lactate].notna().any(axis=1)
    )
    result["has_organ_dysfunction_wo_lactate"] = (
        result[organ_cols_wo_lactate].notna().any(axis=1)
    )

    # Determine ASE (sepsis) status - with lactate version (primary)
    result["sepsis"] = (
        (result["presumed_infection"] == 1)
        & (result["has_organ_dysfunction_w_lactate"])
    ).astype(int)

    # Determine sepsis without lactate version
    result["sepsis_wo_lactate"] = (
        (result["presumed_infection"] == 1)
        & (result["has_organ_dysfunction_wo_lactate"])
    ).astype(int)

    def get_earliest_and_criteria(row, include_lactate=True):
        """Get earliest datetime and corresponding criteria name for organ dysfunction only."""
        candidates = {
            "vasopressor": row.get("vasopressor_dttm"),
            "imv": row.get("imv_dttm"),
            "aki": row.get("aki_dttm"),
            "hyperbilirubinemia": row.get("hyperbilirubinemia_dttm"),
            "thrombocytopenia": row.get("thrombocytopenia_dttm"),
        }
        if include_lactate:
            candidates["lactate"] = row.get("lactate_dttm")

        valid = {k: v for k, v in candidates.items() if pd.notna(v)}
        if not valid:
            return pd.NaT, None
        earliest_criteria = min(valid, key=valid.get)
        return valid[earliest_criteria], earliest_criteria

    def get_onset_dttm(row, include_lactate=True):
        """
        Get ASE onset datetime as the earliest of:
        - Blood culture order time
        - First QAD administration time
        - First organ dysfunction time
        """
        candidates = []

        # Add blood culture time
        if pd.notna(row.get("blood_culture_dttm")):
            candidates.append(("blood_culture", row["blood_culture_dttm"]))

        # Add first QAD time
        if pd.notna(row.get("first_qad_dttm")):
            candidates.append(("first_qad", row["first_qad_dttm"]))

        # Add organ dysfunction times
        organ_times = {
            "vasopressor": row.get("vasopressor_dttm"),
            "imv": row.get("imv_dttm"),
            "aki": row.get("aki_dttm"),
            "hyperbilirubinemia": row.get("hyperbilirubinemia_dttm"),
            "thrombocytopenia": row.get("thrombocytopenia_dttm"),
        }
        if include_lactate:
            organ_times["lactate"] = row.get("lactate_dttm")

        for name, dttm in organ_times.items():
            if pd.notna(dttm):
                candidates.append((name, dttm))

        if not candidates:
            return pd.NaT, None

        earliest = min(candidates, key=lambda x: x[1])
        return earliest[1], earliest[0]

    # Apply to get onset times and first criteria
    result[["ase_onset_w_lactate_dttm", "ase_first_criteria_w_lactate"]] = result.apply(
        lambda r: pd.Series(get_onset_dttm(r, include_lactate=True)), axis=1
    )
    result[["ase_onset_wo_lactate_dttm", "ase_first_criteria_wo_lactate"]] = result.apply(
        lambda r: pd.Series(get_onset_dttm(r, include_lactate=False)), axis=1
    )

    # Presumed infection onset (earliest of blood culture and first QAD)
    def get_presumed_infection_onset(row):
        if row["presumed_infection"] != 1:
            return pd.NaT
        candidates = []
        if pd.notna(row.get("blood_culture_dttm")):
            candidates.append(row["blood_culture_dttm"])
        if pd.notna(row.get("first_qad_dttm")):
            candidates.append(row["first_qad_dttm"])
        if not candidates:
            return pd.NaT
        return min(candidates)

    result["presumed_infection_onset_dttm"] = result.apply(
        get_presumed_infection_onset, axis=1
    )

    # Determine type (community vs hospital) based on onset day
    def get_onset_type(row):
        onset = row.get("ase_onset_w_lactate_dttm")
        admission = row.get("admission_dttm")
        if pd.isna(onset) or pd.isna(admission):
            return None

        # Convert to date for day calculation
        try:
            onset_date = pd.Timestamp(onset).date()
            admission_date = pd.Timestamp(admission).date()
            hospital_day = (onset_date - admission_date).days + 1
            return "community" if hospital_day <= 2 else "hospital"
        except (ValueError, TypeError, AttributeError):
            return None

    result["type"] = result.apply(get_onset_type, axis=1)

    # Select and order final columns
    final_columns = [
        "hospitalization_id",
        "presumed_infection",
        "sepsis",
        "type",
        "presumed_infection_onset_dttm",
        "ase_onset_w_lactate_dttm",
        "ase_onset_wo_lactate_dttm",
        "ase_first_criteria_w_lactate",
        "ase_first_criteria_wo_lactate",
        "vasopressor_dttm",
        "imv_dttm",
        "aki_dttm",
        "hyperbilirubinemia_dttm",
        "thrombocytopenia_dttm",
        "lactate_dttm",
        "esrd",
    ]

    return result[final_columns]


# =============================================================================
# Validation
# =============================================================================


def _validate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply quality checks per CDC guidelines.

    Validates:
    - If sepsis == 1, then presumed_infection must == 1

    Returns
    -------
    pd.DataFrame
        Validated DataFrame (unchanged if valid)

    Raises
    ------
    ValueError
        If validation fails
    """
    # If sepsis == 1, then presumed_infection must == 1
    invalid = df[(df["sepsis"] == 1) & (df["presumed_infection"] == 0)]
    if len(invalid) > 0:
        raise ValueError(
            f"Found {len(invalid)} invalid rows: sepsis=1 with presumed_infection=0"
        )

    return df


# =============================================================================
# Main Public Function
# =============================================================================


def calculate_ase(
    hospitalization_ids: List[str],
    config_path: str = "clif_config.json",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Calculate Adult Sepsis Event (ASE) for given hospitalizations.

    Implements the CDC Adult Sepsis Event surveillance definition:
    - Criteria A: Presumed Infection (blood culture + ≥4 QAD)
    - Criteria B: Organ Dysfunction within ±2 days of blood culture

    Parameters
    ----------
    hospitalization_ids : List[str]
        List of hospitalization IDs to evaluate
    config_path : str, default "clif_config.json"
        Path to clifpy config file
    verbose : bool, default True
        Print progress messages

    Returns
    -------
    pd.DataFrame
        ASE results with columns:
        - hospitalization_id: Unique encounter ID
        - presumed_infection: 1 = met criteria, 0 = not met
        - sepsis: 1 = ASE case, 0 = not ASE
        - type: "community" or "hospital" (based on onset day)
        - presumed_infection_onset_dttm: Earliest of blood culture/first QAD
        - ase_onset_w_lactate_dttm: ASE onset including lactate
        - ase_onset_wo_lactate_dttm: ASE onset excluding lactate
        - ase_first_criteria_w_lactate: First criteria met (with lactate)
        - ase_first_criteria_wo_lactate: First criteria met (without lactate)
        - vasopressor_dttm: First qualifying vasopressor time
        - imv_dttm: First qualifying IMV time
        - aki_dttm: First AKI time
        - hyperbilirubinemia_dttm: First hyperbilirubinemia time
        - thrombocytopenia_dttm: First thrombocytopenia time
        - lactate_dttm: First elevated lactate time
        - esrd: 1 = has ESRD, 0 = no ESRD

    Example
    -------
    >>> from code.ASE import calculate_ase
    >>> hosp_ids = ['H001', 'H002', 'H003']
    >>> results = calculate_ase(hosp_ids, config_path='clif_config.json')
    >>> results.to_parquet('output/ase_results.parquet')
    """
    if verbose:
        print("=== Adult Sepsis Event (ASE) Calculation ===")
        print(f"Processing {len(hospitalization_ids):,} hospitalizations...")

    # Step 1: Load CLIF tables
    if verbose:
        print("Loading CLIF tables...")
    con = _load_clif_tables(hospitalization_ids, config_path)

    # Step 2: Get blood cultures
    if verbose:
        print("Identifying blood cultures...")
    blood_cultures = _get_blood_cultures(con)
    if verbose:
        print(f"  Found blood cultures for {len(blood_cultures):,} hospitalizations")

    if len(blood_cultures) == 0:
        if verbose:
            print("No blood cultures found. Returning empty results.")
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=[
            "hospitalization_id", "presumed_infection", "sepsis", "type",
            "presumed_infection_onset_dttm", "ase_onset_w_lactate_dttm",
            "ase_onset_wo_lactate_dttm", "ase_first_criteria_w_lactate",
            "ase_first_criteria_wo_lactate", "vasopressor_dttm", "imv_dttm",
            "aki_dttm", "hyperbilirubinemia_dttm", "thrombocytopenia_dttm",
            "lactate_dttm", "esrd"
        ])

    # Step 3: Calculate QAD
    if verbose:
        print("Calculating Qualifying Antimicrobial Days (QAD)...")
    qad_results = _calculate_qad(con, blood_cultures)
    if verbose:
        qad_with_value = qad_results[qad_results["total_qad"] >= 4]
        print(f"  {len(qad_with_value):,} hospitalizations with ≥4 QAD")

    # Step 4: Get ESRD flags
    if verbose:
        print("Identifying ESRD patients...")
    esrd_flags = _get_esrd_flags(con)
    if verbose:
        print(f"  Found {len(esrd_flags):,} hospitalizations with ESRD")

    # Step 5: Determine presumed infection
    if verbose:
        print("Determining presumed infection status...")
    presumed_infection = _determine_presumed_infection(con, blood_cultures, qad_results)
    pi_count = presumed_infection["presumed_infection"].sum()
    if verbose:
        print(f"  {pi_count:,} hospitalizations with presumed infection")

    # Step 6: Get organ dysfunction
    if verbose:
        print("Evaluating organ dysfunction criteria...")

    # 6a: Vasopressors
    vasopressor_df = _get_vasopressor_dysfunction(con, blood_cultures)
    if verbose:
        print(f"  Vasopressor: {len(vasopressor_df):,} hospitalizations")

    # 6b: IMV
    imv_df = _get_imv_dysfunction(con, blood_cultures)
    if verbose:
        print(f"  IMV: {len(imv_df):,} hospitalizations")

    # 6c: Lab-based dysfunction
    lab_dysfunction = _get_lab_dysfunction(con, blood_cultures, esrd_flags)
    aki_count = lab_dysfunction["aki_dttm"].notna().sum()
    bili_count = lab_dysfunction["hyperbilirubinemia_dttm"].notna().sum()
    plt_count = lab_dysfunction["thrombocytopenia_dttm"].notna().sum()
    lac_count = lab_dysfunction["lactate_dttm"].notna().sum()
    if verbose:
        print(f"  AKI: {aki_count:,} hospitalizations")
        print(f"  Hyperbilirubinemia: {bili_count:,} hospitalizations")
        print(f"  Thrombocytopenia: {plt_count:,} hospitalizations")
        print(f"  Elevated Lactate: {lac_count:,} hospitalizations")

    # Step 7: Calculate final ASE
    if verbose:
        print("Calculating final ASE status...")
    result = _calculate_final_ase(
        presumed_infection,
        vasopressor_df,
        imv_df,
        lab_dysfunction,
        esrd_flags,
    )

    # Step 8: Validate results
    if verbose:
        print("Validating results...")
    result = _validate_results(result)

    # Summary
    if verbose:
        ase_count = result["sepsis"].sum()
        community_count = (result["type"] == "community").sum()
        hospital_count = (result["type"] == "hospital").sum()
        print("\n=== ASE Calculation Complete ===")
        print(f"Total hospitalizations processed: {len(result):,}")
        print(f"Presumed infections: {result['presumed_infection'].sum():,}")
        print(f"ASE cases (sepsis=1): {ase_count:,}")
        print(f"  Community-onset: {community_count:,}")
        print(f"  Hospital-onset: {hospital_count:,}")

    # Close connection
    con.close()

    return result


# =============================================================================
# Convenience Functions
# =============================================================================


def calculate_ase_from_cohort(
    cohort_path: str,
    config_path: str = "clif_config.json",
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Calculate ASE from a cohort parquet file.

    Parameters
    ----------
    cohort_path : str
        Path to cohort parquet file (must have hospitalization_id column)
    config_path : str
        Path to clifpy config file
    output_path : str, optional
        If provided, save results to this path
    verbose : bool
        Print progress messages

    Returns
    -------
    pd.DataFrame
        ASE results
    """
    # Load cohort
    cohort = pd.read_parquet(cohort_path)
    hosp_ids = cohort["hospitalization_id"].astype(str).unique().tolist()

    # Calculate ASE
    results = calculate_ase(hosp_ids, config_path=config_path, verbose=verbose)

    # Save if output path provided
    if output_path:
        results.to_parquet(output_path, index=False)
        if verbose:
            print(f"Results saved to: {output_path}")

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Calculate Adult Sepsis Event (ASE)")
    parser.add_argument(
        "--cohort",
        type=str,
        default="PHI_DATA/cohort_icu_first_stay.parquet",
        help="Path to cohort parquet file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="clif_config.json",
        help="Path to clifpy config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="PHI_DATA/ase_results.parquet",
        help="Path to save results",
    )

    args = parser.parse_args()

    results = calculate_ase_from_cohort(
        cohort_path=args.cohort,
        config_path=args.config,
        output_path=args.output,
        verbose=True,
    )

    print("\nSample results:")
    print(results.head())
