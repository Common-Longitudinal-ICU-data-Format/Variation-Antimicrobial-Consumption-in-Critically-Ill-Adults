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
    # Adult Sepsis Event (ASE) Testing Notebook

    This notebook tests the ASE detection module using the ICU cohort from `01_cohort.py`.

    ## CDC ASE Definition (Page 5)

    > "ASE: Adult Sepsis Event
    > (Must include the 2 components of criteria A AND include one or more
    > organ dysfunction listed among B criteria)"

    **Criteria A: Presumed Infection** (presence of both):
    1. Blood culture obtained (irrespective of the result), AND
    2. At least 4 Qualifying Antimicrobial Days (QAD)

    **Criteria B: Organ Dysfunction** (at least 1 within ±2 days of blood culture):
    1. Initiation of a new vasopressor infusion
    2. Initiation of invasive mechanical ventilation
    3. Doubling of serum creatinine (excluding ESRD patients with ICD-10 N18.6)
    4. Total bilirubin ≥2.0 mg/dL and increase by 100% from baseline
    5. Platelet count <100 AND ≥50% decline from baseline (baseline must be ≥100)
    6. Optional: Serum lactate ≥2.0 mmol/L

    ## Validation Reference (Page 4)
    > "This definition was validated by Rhee, et al. and shown to be present
    > in 6% of hospital admissions in a study of nearly 400 hospitals."

    ---
    **Reference:** CDC Hospital Toolkit for Adult Sepsis Surveillance (March 2018)
    https://www.cdc.gov/sepsis/pdfs/sepsis-surveillance-toolkit-mar-2018_508.pdf
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
    from pathlib import Path
    import warnings
    warnings.filterwarnings('ignore')

    # Import ASE module
    from ASE import calculate_ase

    print("=== ASE Testing Notebook ===")
    print("Imports loaded successfully")
    return Path, calculate_ase, pd, plt


@app.cell
def _(mo):
    mo.md(r"""## Load ICU Cohort""")
    return


@app.cell
def _(Path, pd):
    print("Loading ICU cohort from 01_cohort.py output...")
    cohort_path = Path('PHI_DATA/cohort_icu_first_stay.parquet')

    if not cohort_path.exists():
        raise FileNotFoundError(
            f"Cohort file not found at {cohort_path}. "
            "Please run 01_cohort.py first to generate the cohort."
        )

    cohort_df = pd.read_parquet(cohort_path)
    hosp_ids = cohort_df['hospitalization_id'].astype(str).unique().tolist()

    print(f"Cohort loaded: {len(cohort_df):,} hospitalizations")
    print(f"Unique hospitalization IDs: {len(hosp_ids):,}")
    return (hosp_ids,)


@app.cell
def _(mo):
    mo.md(r"""## Run ASE Calculation""")
    return


@app.cell
def _(calculate_ase, hosp_ids):
    print("\n=== Running ASE Calculation ===")
    print("This may take a few minutes depending on cohort size...")

    ase_results = calculate_ase(
        hospitalization_ids=hosp_ids,
        config_path='clif_config.json',
        verbose=True
    )

    print(f"\nASE calculation complete. Results: {len(ase_results):,} rows")
    return (ase_results,)


@app.cell
def _(mo):
    mo.md(r"""## Summary Statistics""")
    return


@app.cell
def _(ase_results, mo):
    total = len(ase_results)
    presumed_inf = ase_results['presumed_infection'].sum()
    sepsis = ase_results['sepsis'].sum()

    # Handle type column which may have None values
    type_counts = ase_results['type'].value_counts()
    community = type_counts.get('community', 0)
    hospital = type_counts.get('hospital', 0)

    # Calculate percentages safely
    pi_pct = presumed_inf/total*100 if total > 0 else 0
    sep_pct = sepsis/total*100 if total > 0 else 0
    comm_pct = community/total*100 if total > 0 else 0
    hosp_pct = hospital/total*100 if total > 0 else 0

    mo.md(f"""
    ## ASE Results Summary

    | Metric | Count | Percentage |
    |--------|-------|------------|
    | Total Hospitalizations | {total:,} | 100% |
    | Presumed Infection | {presumed_inf:,} | {pi_pct:.1f}% |
    | **ASE Cases (Sepsis)** | **{sepsis:,}** | **{sep_pct:.1f}%** |
    | Community-Onset | {community:,} | {comm_pct:.1f}% |
    | Hospital-Onset | {hospital:,} | {hosp_pct:.1f}% |

    > **CDC Reference (Page 4):** "This definition was validated by Rhee, et al. and shown
    > to be present in 6% of hospital admissions in a study of nearly 400 hospitals."
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""## Organ Dysfunction Breakdown""")
    return


@app.cell
def _(ase_results, mo):
    vaso = ase_results['vasopressor_dttm'].notna().sum()
    imv = ase_results['imv_dttm'].notna().sum()
    aki = ase_results['aki_dttm'].notna().sum()
    bili = ase_results['hyperbilirubinemia_dttm'].notna().sum()
    plt_low = ase_results['thrombocytopenia_dttm'].notna().sum()
    lactate = ase_results['lactate_dttm'].notna().sum()
    esrd_count = ase_results['esrd'].sum()

    mo.md(f"""
    ## Organ Dysfunction Criteria Met

    | Criterion | Count | CDC Definition Reference |
    |-----------|-------|--------------------------|
    | Vasopressor | {vaso:,} | Page 5: "norepinephrine, dopamine, epinephrine, phenylephrine, OR vasopressin" |
    | IMV | {imv:,} | Page 5: ">1 calendar day between mechanical ventilation episodes" |
    | AKI | {aki:,} | Page 5: "Doubling of serum creatinine... excluding ESRD (N18.6)" |
    | Hyperbilirubinemia | {bili:,} | Page 5: "Total bilirubin >=2.0 mg/dL and increase by 100% from baseline" |
    | Thrombocytopenia | {plt_low:,} | Page 5: "Platelet count <100 cells/uL AND >=50% decline from baseline" |
    | Lactate >= 2.0 | {lactate:,} | Page 5: "Optional: Serum lactate >=2.0 mmol/L" |

    ### ESRD Exclusions
    | Metric | Count |
    |--------|-------|
    | Patients with ESRD (N18.6) | {esrd_count:,} |

    > **Note:** ESRD patients are excluded from AKI criteria per CDC definition.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""## First Criteria Distribution (ASE Cases Only)""")
    return


@app.cell
def _(ase_results, plt):
    # Filter to ASE cases only
    ase_cases = ase_results[ase_results['sepsis'] == 1]

    if len(ase_cases) > 0:
        # Count first criteria occurrences
        criteria_counts = ase_cases['ase_first_criteria_w_lactate'].value_counts()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Define colors for each criterion type
        colors = {
            'blood_culture': '#1f77b4',
            'first_qad': '#2ca02c',
            'vasopressor': '#d62728',
            'imv': '#9467bd',
            'aki': '#8c564b',
            'hyperbilirubinemia': '#e377c2',
            'thrombocytopenia': '#7f7f7f',
            'lactate': '#bcbd22'
        }

        # Get colors for each bar
        bar_colors = [colors.get(c, '#17becf') for c in criteria_counts.index]

        criteria_counts.plot(kind='barh', ax=ax, color=bar_colors)
        ax.set_xlabel('Count', fontsize=12)
        ax.set_ylabel('First Criteria Met', fontsize=12)
        ax.set_title('Distribution of First ASE Criteria Met (with Lactate)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')

        # Add value labels on bars
        for i, (idx, val) in enumerate(criteria_counts.items()):
            ax.text(val + 0.5, i, str(val), va='center', fontsize=10)

        plt.tight_layout()
        fig
    else:
        print("No ASE cases found in the cohort")
        fig = None
    return


@app.cell
def _(mo):
    mo.md(r"""## Onset Type Distribution""")
    return


@app.cell
def _(ase_results, plt):
    # Filter to ASE cases with onset type
    ase_with_type = ase_results[
        (ase_results['sepsis'] == 1) &
        (ase_results['type'].notna())
    ]

    if len(ase_with_type) > 0:
        type_counts_fig = ase_with_type['type'].value_counts()

        fig2, ax2 = plt.subplots(figsize=(8, 6))

        colors_type = {'community': '#2E86AB', 'hospital': '#A23B72'}
        bar_colors_type = [colors_type.get(t, '#F18F01') for t in type_counts_fig.index]

        bars = ax2.bar(type_counts_fig.index, type_counts_fig.values, color=bar_colors_type, edgecolor='black')

        ax2.set_xlabel('Onset Type', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('ASE Cases by Onset Type', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Add value labels on bars
        for bar, bar_count in zip(bars, type_counts_fig.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(bar_count), ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        fig2
    else:
        print("No ASE cases with onset type found")
        fig2 = None
    return


@app.cell
def _(mo):
    mo.md(r"""## Quality Validation Checks""")
    return


@app.cell
def _(ase_results, mo):
    # Validate: sepsis=1 should always have presumed_infection=1
    # Per CDC Page 5: ASE requires BOTH Criteria A (Presumed Infection) AND Criteria B
    invalid_rows = ase_results[
        (ase_results['sepsis'] == 1) &
        (ase_results['presumed_infection'] == 0)
    ]

    check1_status = "PASSED" if len(invalid_rows) == 0 else "FAILED"
    check1_color = "green" if len(invalid_rows) == 0 else "red"

    # Check 2: ASE cases should have at least one organ dysfunction
    organ_cols = ['vasopressor_dttm', 'imv_dttm', 'aki_dttm',
                  'hyperbilirubinemia_dttm', 'thrombocytopenia_dttm', 'lactate_dttm']

    ase_cases_check = ase_results[ase_results['sepsis'] == 1]
    if len(ase_cases_check) > 0:
        has_any_organ = ase_cases_check[organ_cols].notna().any(axis=1)
        invalid_organ = ase_cases_check[~has_any_organ]
        check2_status = "PASSED" if len(invalid_organ) == 0 else "FAILED"
        check2_color = "green" if len(invalid_organ) == 0 else "red"
    else:
        check2_status = "N/A (no ASE cases)"
        check2_color = "gray"

    mo.md(f"""
    ## Quality Validation Checks

    | Check | CDC Reference | Status |
    |-------|---------------|--------|
    | sepsis=1 implies presumed_infection=1 | Page 5: "Must include the 2 components of criteria A" | <span style="color:{check1_color}">**{check1_status}**</span> |
    | ASE cases have >= 1 organ dysfunction | Page 5: "AND include one or more organ dysfunction listed among B criteria" | <span style="color:{check2_color}">**{check2_status}**</span> |

    > **CDC Page 5:** "ASE: Adult Sepsis Event (Must include the 2 components of criteria A
    > AND include one or more organ dysfunction listed among B criteria)"
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""## Save Results""")
    return


@app.cell
def _(ase_results):
    ase_results
    return


@app.cell
def _(ase_results):
    ase_results.hospitalization_id.nunique()
    return


@app.cell
def _(Path, ase_results):
    output_path = Path('PHI_DATA/ase_results.parquet')
    ase_results.to_parquet(output_path, index=False)

    print("=== Results Saved ===")
    print(f"Location: {output_path}")
    print(f"Rows: {len(ase_results):,}")
    print(f"Columns: {list(ase_results.columns)}")

    # Also display sample of results
    print("\n=== Sample Results (first 5 rows) ===")
    return


@app.cell
def _(ase_results):
    # Display sample of ASE positive cases
    ase_positive = ase_results[ase_results['sepsis'] == 1].head(10)
    if len(ase_positive) > 0:
        print("Sample ASE-positive cases:")
        display_cols = ['hospitalization_id', 'presumed_infection', 'sepsis', 'type',
                       'ase_first_criteria_w_lactate']
        ase_positive[display_cols]
    else:
        print("No ASE-positive cases found")
        ase_positive = None
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Notebook Complete

    **Summary:**
    - ASE calculation completed using CDC Adult Sepsis Event surveillance definition
    - Results saved to `PHI_DATA/ase_results.parquet`
    - Quality validation checks passed

    **CDC Reference:** Hospital Toolkit for Adult Sepsis Surveillance (March 2018)
    https://www.cdc.gov/sepsis/pdfs/sepsis-surveillance-toolkit-mar-2018_508.pdf
    """
    )
    return


if __name__ == "__main__":
    app.run()
