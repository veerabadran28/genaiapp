# sustainable_finance_dashboard/data_processing.py

import pandas as pd
import streamlit as st
import os

# --- Configuration ---
DATA_DIR = "data"
PCAF_FILE_NAME = os.path.join(DATA_DIR, "group_client_coverage_dec24.xlsx")
LLM_FILE_NAME = os.path.join(DATA_DIR, "llm_generated.csv")
SFF_FILE_NAME = os.path.join(DATA_DIR, "Mar PP list_cF.xlsx")

def robust_normalize_string_series(series: pd.Series) -> pd.Series:
    """
    Robustly normalizes a pandas Series for string matching:
    1. Fills NaN with empty string.
    2. Converts to string.
    3. Removes '.0' if it's a float string (e.g., '123.0' -> '123').
    4. Strips whitespace.
    5. Converts to lowercase.
    6. Converts truly empty strings or original NaNs (now empty strings) to None
       to prevent them from matching each other during joins if that's not desired.
       If matching empty strings is desired, this last step can be removed.
       For key matching, None is usually safer for "missing" keys.
    """
    if series.empty:
        return series
    
    normalized_series = series.fillna('').astype(str) \
        .str.replace(r'\.0$', '', regex=True) \
        .str.strip().str.lower()
    
    # After normalization, if a value is an empty string, set it to None
    # This helps joins treat these as 'missing' rather than matching all empty strings.
    normalized_series = normalized_series.replace('', None)
    normalized_series = normalized_series.replace('nan', None) # Case where 'nan' string might appear
    return normalized_series

@st.cache_data
def load_pcaf_data():
    """Loads and provides basic info for PCAF dataset."""
    try:
        df = pd.read_excel(PCAF_FILE_NAME)
        # Minimal cleaning on load - specific columns will be normalized before joins
        return df
    except FileNotFoundError:
        st.error(f"Error: PCAF data file '{PCAF_FILE_NAME}' not found. Ensure it's in '{DATA_DIR}'.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading PCAF data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_llm_generated_data():
    """Loads and provides basic info for LLM generated dataset."""
    try:
        df = pd.read_csv(LLM_FILE_NAME)
        # Normalize key columns if needed, or do it right before join
        df['greenRevenuePercent'] = pd.to_numeric(df['greenRevenuePercent'], errors='coerce')
        if 'totalRevenue' in df.columns and df['totalRevenue'].dtype == 'object':
            df['totalRevenue'] = df['totalRevenue'].astype(str).str.replace(',', '', regex=False)
            df['totalRevenue'] = pd.to_numeric(df['totalRevenue'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Error: LLM data file '{LLM_FILE_NAME}' not found. Ensure it's in '{DATA_DIR}'.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading LLM data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_sff_data():
    """Loads and provides basic info for SFF dataset."""
    try:
        df = pd.read_excel(SFF_FILE_NAME)
        return df
    except FileNotFoundError:
        st.error(f"Error: SFF data file '{SFF_FILE_NAME}' not found. Ensure it's in '{DATA_DIR}'.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading SFF data: {e}")
        return pd.DataFrame()

def create_green_revenue_dataset(pcaf_df, llm_df):
    """
    Prepares the GREEN_REVENUE dataset with robust normalization for join keys.
    """
    if pcaf_df.empty or llm_df.empty:
        st.warning("PCAF or LLM data is empty for GREEN_REVENUE creation.")
        return pd.DataFrame()

    pcaf_unique_subset_cols = [
        'cob_date', 'productype', 'legal_entity', 'counterparty_id',
        'counterparty_name', 'parent_id', 'group_id', 'group_name',
        'bic_code', 'country_code'
    ]
    if not all(col in pcaf_df.columns for col in pcaf_unique_subset_cols):
        missing_cols = [col for col in pcaf_unique_subset_cols if col not in pcaf_df.columns]
        st.error(f"PCAF data missing required columns for unique selection: {missing_cols}")
        return pd.DataFrame()
    pcaf_selected_df = pcaf_df[pcaf_unique_subset_cols].drop_duplicates().copy() # Use .copy()

    # Robust normalization for join keys
    pcaf_selected_df['join_key_pcaf'] = robust_normalize_string_series(pcaf_selected_df['counterparty_name'])
    
    llm_df_copy = llm_df.copy()
    llm_df_copy['join_key_llm'] = robust_normalize_string_series(llm_df_copy['companyName'])

    # Filter out rows where join key became None, as they won't match
    pcaf_selected_df.dropna(subset=['join_key_pcaf'], inplace=True)
    llm_df_copy.dropna(subset=['join_key_llm'], inplace=True)

    if pcaf_selected_df.empty or llm_df_copy.empty:
        st.warning("One of the datasets became empty after normalizing and dropping null join keys for GREEN_REVENUE creation.")
        return pd.DataFrame()

    green_revenue_df = pd.merge(
        pcaf_selected_df,
        llm_df_copy,
        left_on='join_key_pcaf',
        right_on='join_key_llm',
        how='inner' # Inner join ensures only matching records are kept
    )

    if green_revenue_df.empty:
        st.warning("Join for GREEN_REVENUE resulted in an empty dataframe. "
                   "Check 'counterparty_name' (PCAF) and 'companyName' (LLM) for matchable values.")
        # For debugging:
        # st.write("Sample PCAF join keys (first 10 non-null):", pcaf_selected_df['join_key_pcaf'].dropna().unique()[:10])
        # st.write("Sample LLM join keys (first 10 non-null):", llm_df_copy['join_key_llm'].dropna().unique()[:10])
        return pd.DataFrame()

    final_attributes = [
        'cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name',
        'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code',
        'year', 'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources'
    ]
    actual_cols = [col for col in final_attributes if col in green_revenue_df.columns]
    green_revenue_df = green_revenue_df[actual_cols].copy()

    green_revenue_df['pure_play_flag'] = 'N'
    green_revenue_df['greenRevenuePercent'] = pd.to_numeric(green_revenue_df['greenRevenuePercent'], errors='coerce')
    green_revenue_df.loc[green_revenue_df['greenRevenuePercent'] >= 50, 'pure_play_flag'] = 'Y'
    # If GR% is NaN, flag remains 'N'. If 'N/A' is desired for NaN GR%:
    # green_revenue_df.loc[green_revenue_df['greenRevenuePercent'].isna(), 'pure_play_flag'] = 'N/A'
    
    return green_revenue_df

def compare_datasets_for_overlap(green_revenue_df, sff_df, filter_pure_play_flag_value):
    """
    Compares GREEN_REVENUE (filtered) with SFF_DATA with robust normalization for join keys.
    Primary join key: counterparty_id (GR) vs SDS (SFF).
    """
    if green_revenue_df.empty:
        st.warning(f"GREEN_REVENUE dataset is empty for comparison with flag '{filter_pure_play_flag_value}'.")
        return pd.DataFrame(), sff_df.copy() if sff_df is not None and not sff_df.empty else pd.DataFrame(), pd.DataFrame()

    gr_filtered = green_revenue_df[green_revenue_df['pure_play_flag'] == filter_pure_play_flag_value].copy()

    if gr_filtered.empty:
        st.info(f"No companies in GREEN_REVENUE for pure_play_flag = '{filter_pure_play_flag_value}'.")
        return pd.DataFrame(), sff_df.copy() if sff_df is not None and not sff_df.empty else pd.DataFrame(), pd.DataFrame()

    if sff_df is None or sff_df.empty:
        st.warning("SFF_DATA is empty for comparison.")
        # All gr_filtered are "Un-Identified"
        gr_unidentified_cols = [
            'cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name',
            'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code', 'year',
            'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag'
        ]
        return pd.DataFrame(), pd.DataFrame(), gr_filtered[[col for col in gr_unidentified_cols if col in gr_filtered.columns]]

    # Check for essential join key columns
    if 'counterparty_id' not in gr_filtered.columns:
        st.error("Critical column 'counterparty_id' missing in filtered GREEN_REVENUE.")
        return pd.DataFrame(), sff_df.copy(), pd.DataFrame()
    if 'SDS' not in sff_df.columns:
        st.error("Critical column 'SDS' missing in SFF_DATA.")
        return pd.DataFrame(), pd.DataFrame(), gr_filtered # All GR become unidentified


    # Robust normalization for join keys
    gr_filtered['norm_key_gr'] = robust_normalize_string_series(gr_filtered['counterparty_id'])
    
    sff_df_copy = sff_df.copy()
    sff_df_copy['norm_key_sff'] = robust_normalize_string_series(sff_df_copy['SDS'])

    # Filter out rows where join key became None after normalization
    gr_filtered.dropna(subset=['norm_key_gr'], inplace=True)
    sff_df_copy.dropna(subset=['norm_key_sff'], inplace=True)

    if gr_filtered.empty or sff_df_copy.empty:
        st.warning("One or both datasets became empty after normalizing and dropping null join keys for comparison. This may result in zero overlaps.")
        # If gr_filtered is not empty, it becomes unidentified. If sff_df_copy is not empty, it becomes identified.
        gr_unidentified_cols = [
            'cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name',
            'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code', 'year',
            'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag'
        ]
        sff_identified_cols = sff_df.columns.tolist()

        return (pd.DataFrame(), 
                sff_df_copy[[col for col in sff_identified_cols if col in sff_df_copy.columns]] if not sff_df_copy.empty else pd.DataFrame(), 
                gr_filtered[[col for col in gr_unidentified_cols if col in gr_filtered.columns]] if not gr_filtered.empty else pd.DataFrame())


    merged_df = pd.merge(
        gr_filtered,
        sff_df_copy,
        left_on='norm_key_gr',
        right_on='norm_key_sff',
        how='outer',
        suffixes=('_gr', '_sff'),
        indicator=True
    )
    
    # Debug: Show merge indicator counts
    # st.write(f"Debug - Merge indicator counts for flag '{filter_pure_play_flag_value}':\n{merged_df['_merge'].value_counts(dropna=False)}")
    # st.write(f"Debug - Sample norm_key_gr (GR): {gr_filtered['norm_key_gr'].dropna().unique()[:5]}")
    # st.write(f"Debug - Sample norm_key_sff (SFF): {sff_df_copy['norm_key_sff'].dropna().unique()[:5]}")


    # --- Overlap ---
    overlap_records = merged_df[merged_df['_merge'] == 'both']
    overlap_df_cols_gr_suffixed = [
        'cob_date_gr', 'productype_gr', 'legal_entity_gr', 'counterparty_id_gr', 'counterparty_name_gr',
        'parent_id_gr', 'group_id_gr', 'group_name_gr', 'bic_code_gr', 'country_code_gr', 'year_gr',
        'totalRevenue_gr', 'greenRevenuePercent_gr', 'justification_gr', 'dataSources_gr', 'pure_play_flag_gr'
    ]
    overlap_df = overlap_records[[col for col in overlap_df_cols_gr_suffixed if col in overlap_records.columns]].copy()
    overlap_df.columns = [col.replace('_gr', '') for col in overlap_df.columns]

    # --- Identified (SFF only) ---
    identified_sff_only_records = merged_df[merged_df['_merge'] == 'right_only']
    sff_original_cols = sff_df.columns.tolist() # Use original SFF columns
    identified_sff_only_df = identified_sff_only_records[[col for col in sff_original_cols if col in identified_sff_only_records.columns]].copy()

    # --- Un-Identified (GR only) ---
    unidentified_gr_only_records = merged_df[merged_df['_merge'] == 'left_only']
    # Columns should be from the GR side of the merge (suffixed with _gr)
    unidentified_gr_only_df_cols_gr_suffixed = [
         'cob_date_gr', 'productype_gr', 'legal_entity_gr', 'counterparty_id_gr', 'counterparty_name_gr',
        'parent_id_gr', 'group_id_gr', 'group_name_gr', 'bic_code_gr', 'country_code_gr', 'year_gr',
        'totalRevenue_gr', 'greenRevenuePercent_gr', 'justification_gr', 'dataSources_gr', 'pure_play_flag_gr'
    ]
    unidentified_gr_only_df = unidentified_gr_only_records[[col for col in unidentified_gr_only_df_cols_gr_suffixed if col in unidentified_gr_only_records.columns]].copy()
    unidentified_gr_only_df.columns = [col.replace('_gr', '') for col in unidentified_gr_only_df.columns]
    
    return overlap_df, identified_sff_only_df, unidentified_gr_only_df
