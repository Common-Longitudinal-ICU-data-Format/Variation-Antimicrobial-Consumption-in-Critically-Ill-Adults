#!/usr/bin/env python3
"""
VACCIA Multi-Site Data Merger
Automatically merges results from all site folders into aggregated all_sites results.
"""

import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil
from collections import OrderedDict

# Configuration
RESULTS_FOLDER = "RESULTS_UPLOAD_ME"
OUTPUT_FOLDER = "all_sites"

# File lists
CSV_FILES_WITH_SITE = [
    "afd_summary.csv",
    "asc_by_year_summary.csv",
    "daily_asc_summary.csv",
    "dasc_by_year.csv",
    "dasc_overall.csv",
]

JSON_FILES = [
    "table1_summary.json",
]

PLOT_FILES = [
    "asc_by_window_plot.png",
    "asc_by_year_plot.png",
]


def discover_sites(base_path="."):
    """Auto-detect all site folders containing RESULTS_UPLOAD_ME."""
    sites = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item != OUTPUT_FOLDER and item != ".git":
            results_path = os.path.join(item_path, RESULTS_FOLDER)
            if os.path.exists(results_path):
                sites.append(item)
    return sorted(sites)


# ============================================================================
# TABLE 1 AGGREGATION FUNCTIONS
# ============================================================================

def parse_mean_sd(value_str):
    """Parse 'mean ± SD' format. Returns (mean, sd) or (None, None)."""
    if pd.isna(value_str) or value_str == "":
        return None, None
    pattern = r'([\d.]+)\s*±\s*([\d.]+)'
    match = re.search(pattern, str(value_str))
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def parse_count_pct(value_str):
    """Parse 'n (%)' format. Returns (count, pct) or (None, None)."""
    if pd.isna(value_str) or value_str == "":
        return None, None
    pattern = r'([\d,]+)\s*\(([\d.]+)%\)'
    match = re.search(pattern, str(value_str))
    if match:
        count = int(match.group(1).replace(',', ''))
        pct = float(match.group(2))
        return count, pct
    return None, None


def parse_simple_number(value_str):
    """Parse simple number (with or without commas). Returns float or None."""
    if pd.isna(value_str) or value_str == "":
        return None
    try:
        # Remove commas and convert
        cleaned = str(value_str).replace(',', '').strip()
        return float(cleaned)
    except (ValueError, AttributeError):
        return None


def parse_percentage(value_str):
    """Parse percentage like '52.1%'. Returns float or None."""
    if pd.isna(value_str) or value_str == "":
        return None
    pattern = r'([\d.]+)%'
    match = re.search(pattern, str(value_str))
    if match:
        return float(match.group(1))
    return None


def weighted_mean(values, weights):
    """Calculate weighted mean."""
    values = np.array(values)
    weights = np.array(weights)
    if np.sum(weights) == 0:
        return np.nan
    return np.sum(values * weights) / np.sum(weights)


def pooled_sd(sds, ns):
    """Calculate pooled standard deviation."""
    sds = np.array(sds)
    ns = np.array(ns)
    if len(sds) == 0 or np.sum(ns) <= 1:
        return np.nan
    variance = np.sum((ns - 1) * sds**2) / (np.sum(ns) - 1)
    return np.sqrt(variance)


def format_mean_sd(mean, sd):
    """Format mean ± SD with appropriate decimal places."""
    if pd.isna(mean) or pd.isna(sd):
        return ""
    return f"{mean:.1f} ± {sd:.1f}"


def format_count_pct(count, total):
    """Format count (%)."""
    if pd.isna(count) or pd.isna(total) or total == 0:
        return ""
    pct = (count / total) * 100
    return f"{int(count):,} ({pct:.1f}%)"


def format_number(num):
    """Format number with commas."""
    if pd.isna(num):
        return ""
    if num == int(num):
        return f"{int(num):,}"
    return f"{num:,.2f}"


def aggregate_table1_row(variable, site_data_dict, total_n):
    """
    Aggregate a single row across sites.

    Args:
        variable: Variable name
        site_data_dict: Dict of {site_name: {'Value': val, 'n_missing': n}}
        total_n: Total number of patients across all sites

    Returns:
        aggregated_value: Combined value for All_Sites column
    """
    values = [data['Value'] for data in site_data_dict.values()]

    # Check if empty or header row
    if all(v == "" or pd.isna(v) for v in values):
        return ""

    # Check for NOT AVAILABLE
    if any(str(v) == "NOT AVAILABLE" for v in values):
        return "NOT AVAILABLE"

    # Try to parse as mean ± SD
    means_sds = [parse_mean_sd(v) for v in values]
    if all(m is not None for m, s in means_sds):
        means = [m for m, s in means_sds]
        sds = [s for m, s in means_sds]
        site_ns = [data.get('site_n', total_n / len(site_data_dict)) for data in site_data_dict.values()]

        combined_mean = weighted_mean(means, site_ns)
        combined_sd = pooled_sd(sds, site_ns)
        return format_mean_sd(combined_mean, combined_sd)

    # Try to parse as count (%)
    counts_pcts = [parse_count_pct(v) for v in values]
    if all(c is not None for c, p in counts_pcts):
        counts = [c for c, p in counts_pcts]
        total_count = sum(counts)
        return format_count_pct(total_count, total_n)

    # Try to parse as simple percentage
    pcts = [parse_percentage(v) for v in values]
    if all(p is not None for p in pcts):
        site_ns = [data.get('site_n', total_n / len(site_data_dict)) for data in site_data_dict.values()]
        # Weight percentages by site size
        combined_pct = weighted_mean(pcts, site_ns)
        return f"{combined_pct:.1f}%"

    # Try to parse as simple number
    nums = [parse_simple_number(v) for v in values]
    if all(n is not None for n in nums):
        # For counts, sum them
        if variable in ['N (total patients)', 'Total patient-days']:
            return format_number(sum(nums))
        # For rates, take weighted mean
        else:
            site_ns = [data.get('site_n', total_n / len(site_data_dict)) for data in site_data_dict.values()]
            combined = weighted_mean(nums, site_ns)
            return format_number(combined)

    # Default: return empty
    return ""


def merge_csv_with_site_column(sites, filename, output_dir):
    """Merge CSV files that already have a 'site' column."""
    print(f"  Merging {filename}...")
    dfs = []

    for site in sites:
        file_path = os.path.join(site, RESULTS_FOLDER, filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)
        else:
            print(f"    Warning: {file_path} not found, skipping.")

    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        output_path = os.path.join(output_dir, filename)
        merged_df.to_csv(output_path, index=False)
        print(f"    ✓ Saved {output_path} ({len(merged_df)} rows)")
        return merged_df
    else:
        print(f"    ✗ No data to merge for {filename}")
        return None


def merge_dot_cohort_wide(sites, output_dir):
    """Merge dot_cohort_level.csv into wide format with sites as columns."""
    filename = "dot_cohort_level.csv"
    print(f"  Merging {filename} (wide format)...")

    # Load all site data
    dfs = []
    for site in sites:
        file_path = os.path.join(site, RESULTS_FOLDER, filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)
        else:
            print(f"    Warning: {file_path} not found, skipping.")

    if not dfs:
        print(f"    ✗ No data to merge for {filename}")
        return None

    # Combine all data
    combined_df = pd.concat(dfs, ignore_index=True)

    # Pivot to wide format
    wide_df = combined_df.pivot(index='metric', columns='site', values='value').reset_index()

    # Calculate All_Sites aggregated column
    site_cols = [col for col in wide_df.columns if col != 'metric']

    all_sites_values = []
    for idx, row in wide_df.iterrows():
        metric = row['metric']
        if 'Total DOT' in metric or 'Total PD' in metric:
            # Sum for totals
            all_sites_values.append(row[site_cols].sum())
        elif 'per 1000 PD' in metric:
            # Calculate rate from totals
            total_dot_row = wide_df[wide_df['metric'] == 'Total DOT']
            total_pd_row = wide_df[wide_df['metric'] == 'Total PD']
            if not total_dot_row.empty and not total_pd_row.empty:
                total_dot = total_dot_row[site_cols].sum().sum()
                total_pd = total_pd_row[site_cols].sum().sum()
                all_sites_values.append((total_dot / total_pd) * 1000 if total_pd > 0 else 0)
            else:
                all_sites_values.append(0)
        else:
            all_sites_values.append(row[site_cols].mean())

    wide_df['All_Sites'] = all_sites_values

    # Reorder columns: metric, site1, site2, ..., All_Sites
    ordered_cols = ['metric'] + sorted([col for col in wide_df.columns if col not in ['metric', 'All_Sites']]) + ['All_Sites']
    wide_df = wide_df[ordered_cols]

    # Rename site columns to uppercase
    rename_dict = {col: col.upper() for col in wide_df.columns if col not in ['metric', 'All_Sites']}
    wide_df = wide_df.rename(columns=rename_dict)

    # Save
    output_path = os.path.join(output_dir, filename)
    wide_df.to_csv(output_path, index=False)
    print(f"    ✓ Saved {output_path} ({len(wide_df)} rows, wide format)")
    return wide_df


def merge_dot_antibiotic_wide(sites, output_dir):
    """Merge dot_antibiotic_level.csv into wide format with all_sites aggregation."""
    filename = "dot_antibiotic_level.csv"
    print(f"  Merging {filename} (wide format with all_sites aggregation)...")

    # Load all site data
    site_dfs = {}
    for site in sites:
        file_path = os.path.join(site, RESULTS_FOLDER, filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            site_dfs[site.upper()] = df
        else:
            print(f"    Warning: {file_path} not found, skipping.")

    if not site_dfs:
        print(f"    ✗ No data to merge for {filename}")
        return None

    # Get all unique antibiotics across all sites
    all_antibiotics = set()
    for df in site_dfs.values():
        all_antibiotics.update(df['antibiotic'].unique())
    all_antibiotics = sorted(all_antibiotics)

    # Build wide format result
    result_rows = []

    for antibiotic in all_antibiotics:
        row_data = {'antibiotic': antibiotic}

        all_sites_total_dot = 0
        all_sites_pd = 0
        spectrum_score = None

        # Add columns for each site
        for site in sorted(site_dfs.keys()):
            df = site_dfs[site]
            abx_row = df[df['antibiotic'] == antibiotic]

            if not abx_row.empty:
                row_data[f'{site}_total_dot'] = abx_row.iloc[0]['total_dot']
                row_data[f'{site}_total_pd'] = abx_row.iloc[0]['total_pd']
                row_data[f'{site}_dot_per_1000_pd'] = abx_row.iloc[0]['dot_per_1000_pd']

                all_sites_total_dot += abx_row.iloc[0]['total_dot']
                all_sites_pd = abx_row.iloc[0]['total_pd']  # Should be same for all

                if spectrum_score is None:
                    spectrum_score = abx_row.iloc[0]['spectrum_score']
            else:
                # Antibiotic not used at this site
                row_data[f'{site}_total_dot'] = 0
                row_data[f'{site}_total_pd'] = all_sites_pd if all_sites_pd > 0 else 0
                row_data[f'{site}_dot_per_1000_pd'] = 0.0

        # Calculate all_sites aggregation
        # Get the total PD from the first site that has this antibiotic
        if all_sites_pd == 0:
            # Get total PD from any site
            for df in site_dfs.values():
                if not df.empty:
                    all_sites_pd = df.iloc[0]['total_pd']
                    break

        # Actually, all_sites_pd should be the COMBINED pd across all sites
        # Let me recalculate: for all_sites, PD should be sum of all site PDs
        total_pd_sum = 0
        for site in sorted(site_dfs.keys()):
            df = site_dfs[site]
            if not df.empty:
                total_pd_sum = df.iloc[0]['total_pd']  # This is actually same for all rows in a site

        # Wait, I need to think about this differently
        # total_pd for a site is the same for ALL antibiotics in that site
        # So all_sites_pd should be the sum of each site's total_pd
        # Let me get it from the first row of each site's dataframe

        all_sites_pd_sum = 0
        for site in sorted(site_dfs.keys()):
            df = site_dfs[site]
            if not df.empty:
                all_sites_pd_sum += df.iloc[0]['total_pd']

        row_data['all_sites_total_dot'] = all_sites_total_dot
        row_data['all_sites_pd'] = all_sites_pd_sum
        row_data['all_sites_dot_per_1000_pd'] = (all_sites_total_dot / all_sites_pd_sum * 1000) if all_sites_pd_sum > 0 else 0.0
        row_data['spectrum_score'] = spectrum_score if spectrum_score is not None else 0

        result_rows.append(row_data)

    # Create DataFrame
    result_df = pd.DataFrame(result_rows)

    # Order columns properly
    base_cols = ['antibiotic']
    site_cols = []
    for site in sorted(site_dfs.keys()):
        site_cols.extend([f'{site}_total_dot', f'{site}_total_pd', f'{site}_dot_per_1000_pd'])

    final_cols = base_cols + site_cols + ['all_sites_total_dot', 'all_sites_pd', 'all_sites_dot_per_1000_pd', 'spectrum_score']
    result_df = result_df[final_cols]

    # Save
    output_path = os.path.join(output_dir, filename)
    result_df.to_csv(output_path, index=False)
    print(f"    ✓ Saved {output_path} ({len(result_df)} rows, wide format)")
    return result_df


def merge_table1_summary(sites, output_dir):
    """Special aggregation for table1_summary.csv with multi-column output."""
    print(f"  Merging table1_summary.csv (with aggregation)...")

    # Load all site data
    site_dfs = {}
    site_ns = {}
    for site in sites:
        file_path = os.path.join(site, RESULTS_FOLDER, "table1_summary.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            site_dfs[site.upper()] = df
            # Get N from the data
            n_row = df[df['Variable'] == 'N (total patients)']
            if not n_row.empty:
                site_ns[site.upper()] = parse_simple_number(n_row.iloc[0]['Value'])
        else:
            print(f"    Warning: {file_path} not found, skipping.")

    if not site_dfs:
        print(f"    ✗ No data to merge for table1_summary.csv")
        return None

    total_n = sum(site_ns.values())

    # Load merged antibiotic and outcome data for special handling
    dot_df = None
    dasc_df = None
    afd_df = None
    dot_path = os.path.join(output_dir, "dot_antibiotic_level.csv")
    dasc_path = os.path.join(output_dir, "dasc_overall.csv")
    afd_path = os.path.join(output_dir, "afd_summary.csv")

    if os.path.exists(dot_path):
        dot_df = pd.read_csv(dot_path)
    if os.path.exists(dasc_path):
        dasc_df = pd.read_csv(dasc_path)
    if os.path.exists(afd_path):
        afd_df = pd.read_csv(afd_path)

    # Get the structure from the first site
    template_df = list(site_dfs.values())[0]

    # Build result rows
    result_rows = []

    for idx, row in template_df.iterrows():
        category = row['Category']
        variable = row['Variable']
        notes = row['Notes']

        # Build site data dict
        site_data = {}
        for site, df in site_dfs.items():
            site_row = df[(df['Category'] == category) & (df['Variable'] == variable)]
            if not site_row.empty:
                site_data[site] = {
                    'Value': site_row.iloc[0]['Value'],
                    'n_missing': site_row.iloc[0]['n_missing'],
                    'site_n': site_ns.get(site, total_n / len(sites))
                }

        # Calculate aggregated value
        all_sites_value = ""

        # Convert variable to string for checking
        var_str = str(variable) if not pd.isna(variable) else ""

        # Special handling for antibiotic lists
        if var_str.strip().startswith('Common antibiotics prescribed'):
            all_sites_value = ""
        # Special handling for DOT per 1000 PD (must come before general antibiotic handling)
        elif 'DOT per 1000 patient days' in var_str:
            # Calculate from dot_cohort_level (now in wide format)
            dot_cohort_path = os.path.join(output_dir, "dot_cohort_level.csv")
            if os.path.exists(dot_cohort_path):
                dot_cohort = pd.read_csv(dot_cohort_path)
                # Get All_Sites values from wide format
                dot_row = dot_cohort[dot_cohort['metric'] == 'Overall DOT per 1000 PD']
                if not dot_row.empty:
                    all_sites_value = f"{dot_row.iloc[0]['All_Sites']:.2f}"
        # Special handling for DASC
        elif 'DASC' in var_str and 'per 1000 PD' in var_str:
            if dasc_df is not None:
                total_dasc = dasc_df[dasc_df['metric'] == 'Total DASC']['value'].sum()
                total_pd = dasc_df[dasc_df['metric'] == 'Total PD']['value'].sum()
                all_sites_value = f"{(total_dasc / total_pd) * 1000:.2f}"
        # Special handling for AFD
        elif 'Antibiotic-Free Days' in var_str and '%' in var_str:
            if afd_df is not None:
                mean_rates = afd_df[afd_df['metric'] == 'mean_afd_rate']['value'].values
                site_counts = afd_df[afd_df['metric'] == 'total_hospitalizations']['value'].values
                if len(mean_rates) > 0 and len(site_counts) > 0:
                    combined_rate = weighted_mean(mean_rates, site_counts)
                    all_sites_value = f"{combined_rate * 100:.1f}%"
        elif 'AFD rate' in var_str:
            if afd_df is not None:
                mean_rates = afd_df[afd_df['metric'] == 'mean_afd_rate']['value'].values
                std_rates = afd_df[afd_df['metric'] == 'std_afd_rate']['value'].values
                site_counts = afd_df[afd_df['metric'] == 'total_hospitalizations']['value'].values
                if len(mean_rates) > 0:
                    combined_mean = weighted_mean(mean_rates, site_counts)
                    combined_sd = pooled_sd(std_rates, site_counts)
                    all_sites_value = format_mean_sd(combined_mean, combined_sd)
        # General antibiotic handling (for individual antibiotics)
        elif category == 'Antibiotics' and var_str.strip() and dot_df is not None:
            # Look up antibiotic in merged DOT data (now in wide format)
            antibiotic_name = var_str.strip()
            abx_rows = dot_df[dot_df['antibiotic'] == antibiotic_name]
            if not abx_rows.empty:
                # Get the all_sites rate from wide format
                all_sites_value = f"{abx_rows.iloc[0]['all_sites_dot_per_1000_pd']:.2f}"
        else:
            # Standard aggregation
            all_sites_value = aggregate_table1_row(variable, site_data, total_n)

        # Get individual site values
        site_values = {}
        for site in sorted(site_dfs.keys()):
            if site in site_data:
                site_values[site] = site_data[site]['Value']
            else:
                site_values[site] = ""

        # Calculate total n_missing
        total_missing = ""
        if site_data:
            missing_vals = [data['n_missing'] for data in site_data.values()]
            if any(m != "" and not pd.isna(m) for m in missing_vals):
                total_missing = sum(parse_simple_number(m) or 0 for m in missing_vals)
                if total_missing > 0:
                    total_missing = int(total_missing)
                else:
                    total_missing = ""

        # Build result row
        result_row = OrderedDict([
            ('Category', category),
            ('Variable', variable),
            ('All_Sites', all_sites_value),
        ])

        # Add individual site columns
        for site in sorted(site_dfs.keys()):
            result_row[site] = site_values.get(site, "")

        result_row['n_missing_total'] = total_missing
        result_row['Notes'] = notes

        result_rows.append(result_row)

    # Handle antibiotic list - get top 15 from merged data
    if dot_df is not None:
        # Find the index where antibiotics list starts
        antibiotic_list_idx = None
        for idx, row_dict in enumerate(result_rows):
            if 'Common antibiotics prescribed' in str(row_dict.get('Variable', '')):
                antibiotic_list_idx = idx + 1
                break

        if antibiotic_list_idx is not None:
            # Remove old antibiotic rows (find next empty row)
            end_idx = antibiotic_list_idx
            for idx in range(antibiotic_list_idx, len(result_rows)):
                if result_rows[idx]['Variable'] == 'DOT per 1000 patient days (all antibiotics in excel)':
                    end_idx = idx
                    break

            # Get top 15 from merged data (now in wide format)
            top15_abx = dot_df.nlargest(15, 'all_sites_dot_per_1000_pd')

            # Build new antibiotic rows
            new_abx_rows = []
            for _, abx_row in top15_abx.iterrows():
                antibiotic = abx_row['antibiotic']
                combined_rate = abx_row['all_sites_dot_per_1000_pd']

                # Get site-specific rates from wide format
                site_values = {}
                for site in sorted(site_dfs.keys()):
                    # Get rate from wide format dot_df
                    rate_col = f'{site}_dot_per_1000_pd'
                    if rate_col in abx_row:
                        site_values[site] = f"{abx_row[rate_col]:.2f}"
                    else:
                        site_values[site] = "0.00"

                abx_result_row = OrderedDict([
                    ('Category', 'Antibiotics'),
                    ('Variable', f'  {antibiotic}'),
                    ('All_Sites', f"{combined_rate:.2f}"),
                ])
                for site in sorted(site_dfs.keys()):
                    abx_result_row[site] = site_values.get(site, "0.00")
                abx_result_row['n_missing_total'] = ""
                abx_result_row['Notes'] = ""

                new_abx_rows.append(abx_result_row)

            # Replace old rows with new
            result_rows = result_rows[:antibiotic_list_idx] + new_abx_rows + result_rows[end_idx:]

    # Create DataFrame
    result_df = pd.DataFrame(result_rows)

    # Save
    output_path = os.path.join(output_dir, "table1_summary.csv")
    result_df.to_csv(output_path, index=False)
    print(f"    ✓ Saved {output_path} ({len(result_df)} rows)")
    return result_df


def merge_json_files(sites, filename, output_dir):
    """Merge JSON files from all sites."""
    print(f"  Merging {filename}...")
    merged_data = {}

    for site in sites:
        file_path = os.path.join(site, RESULTS_FOLDER, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                merged_data[site.lower()] = data
        else:
            print(f"    Warning: {file_path} not found, skipping.")

    if merged_data:
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(merged_data, f, indent=2)
        print(f"    ✓ Saved {output_path} ({len(merged_data)} sites)")
        return merged_data
    else:
        print(f"    ✗ No data to merge for {filename}")
        return None


def create_combined_asc_by_year_plot(df, output_dir):
    """Create combined plot for ASC by year across all sites."""
    print("  Creating combined ASC by year plot...")

    if df is None or df.empty:
        print("    ✗ No data available for ASC by year plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each site
    for site in df['site'].unique():
        site_data = df[df['site'] == site]
        ax.plot(site_data['year'], site_data['mean_asc'],
                marker='o', linewidth=2, markersize=8, label=site.upper())

        # Add confidence intervals
        ax.fill_between(site_data['year'],
                        site_data['lower_ci_asc'],
                        site_data['upper_ci_asc'],
                        alpha=0.2)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Mean ASC Score', fontsize=12)
    ax.set_title('Antibiotic Spectrum Coverage (ASC) by Year - Multi-Site Comparison',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, 'asc_by_year_plot.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved {output_path}")


def create_combined_asc_by_window_plot(df, output_dir):
    """Create combined plot for ASC by window across all sites."""
    print("  Creating combined ASC by window plot...")

    if df is None or df.empty:
        print("    ✗ No data available for ASC by window plot")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot each site
    for site in df['site'].unique():
        site_data = df[df['site'] == site]
        ax.plot(site_data['window_num'], site_data['mean_asc'],
                marker='o', linewidth=2, markersize=6, label=site.upper())

        # Add confidence intervals
        ax.fill_between(site_data['window_num'],
                        site_data['lower_ci_asc'],
                        site_data['upper_ci_asc'],
                        alpha=0.2)

    ax.set_xlabel('Window Number (Days)', fontsize=12)
    ax.set_ylabel('Mean ASC Score', fontsize=12)
    ax.set_title('Antibiotic Spectrum Coverage (ASC) by Time Window - Multi-Site Comparison',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, 'asc_by_window_plot.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved {output_path}")


def main():
    """Main function to merge all site data."""
    print("=" * 70)
    print("VACCIA Multi-Site Data Merger")
    print("=" * 70)

    # Discover sites
    print("\n1. Discovering site folders...")
    sites = discover_sites()
    if not sites:
        print("  ✗ No site folders found!")
        return
    print(f"  ✓ Found {len(sites)} sites: {', '.join(sites)}")

    # Create output directory
    output_dir = os.path.join(OUTPUT_FOLDER, RESULTS_FOLDER)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n2. Created output directory: {output_dir}")

    # Merge CSV files with site column
    print("\n3. Merging CSV files (with site column)...")
    merged_data = {}
    for csv_file in CSV_FILES_WITH_SITE:
        df = merge_csv_with_site_column(sites, csv_file, output_dir)
        if df is not None:
            merged_data[csv_file] = df

    # Merge DOT files with wide format
    print("\n4. Merging DOT files (wide format)...")
    dot_cohort_df = merge_dot_cohort_wide(sites, output_dir)
    if dot_cohort_df is not None:
        merged_data['dot_cohort_level.csv'] = dot_cohort_df

    dot_antibiotic_df = merge_dot_antibiotic_wide(sites, output_dir)
    if dot_antibiotic_df is not None:
        merged_data['dot_antibiotic_level.csv'] = dot_antibiotic_df

    # Merge table1_summary.csv with special aggregation
    print("\n5. Merging table1_summary.csv with aggregation...")
    table1_df = merge_table1_summary(sites, output_dir)
    if table1_df is not None:
        merged_data['table1_summary.csv'] = table1_df

    # Merge JSON files
    print("\n6. Merging JSON files...")
    for json_file in JSON_FILES:
        merge_json_files(sites, json_file, output_dir)

    # Create combined plots
    print("\n7. Creating combined plots...")
    if 'asc_by_year_summary.csv' in merged_data:
        create_combined_asc_by_year_plot(merged_data['asc_by_year_summary.csv'], output_dir)

    if 'daily_asc_summary.csv' in merged_data:
        create_combined_asc_by_window_plot(merged_data['daily_asc_summary.csv'], output_dir)

    print("\n" + "=" * 70)
    print("✓ Merge complete! All results saved to:", output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
