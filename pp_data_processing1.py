# sustainable_finance_dashboard/data_processing.py

import pandas as pd
import streamlit as st
import os

# --- Configuration ---
DATA_DIR = "data"
PCAF_FILE_NAME = os.path.join(DATA_DIR, "group_client_coverage_dec24.xlsx")
LLM_FILE_NAME = os.path.join(DATA_DIR, "llm_generated.csv") # Updated to CSV
SFF_FILE_NAME = os.path.join(DATA_DIR, "Mar PP list_cF.xlsx")

@st.cache_data
def load_pcaf_data():
    """Loads the PCAF dataset from the specified Excel file."""
    try:
        df = pd.read_excel(PCAF_FILE_NAME)
        expected_cols = ['cob_date', 'productype', 'legal_entity', 'counterparty_id',
                         'counterparty_name', 'parent_id', 'group_id', 'group_name',
                         'bic_code', 'naics_code', 'country_code']
        if not all(col in df.columns for col in expected_cols):
            st.error(f"PCAF data ('{PCAF_FILE_NAME}') is missing one or more expected columns. Expected: {expected_cols}. Found: {df.columns.tolist()}")
            return pd.DataFrame()
        return df
    except FileNotFoundError:
        st.error(f"Error: The PCAF data file '{PCAF_FILE_NAME}' was not found. Ensure it is in the '{DATA_DIR}' subdirectory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading PCAF data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_llm_generated_data():
    """Loads the LLM generated dataset from the specified CSV file."""
    try:
        df = pd.read_csv(LLM_FILE_NAME) # Changed to read_csv
        expected_cols = ['companyName', 'year', 'totalRevenue',
                         'greenRevenuePercent', 'justification', 'dataSources']
        if not all(col in df.columns for col in expected_cols):
            st.error(f"LLM generated data ('{LLM_FILE_NAME}') is missing one or more expected columns. Expected: {expected_cols}. Found: {df.columns.tolist()}")
            return pd.DataFrame()
        df['greenRevenuePercent'] = pd.to_numeric(df['greenRevenuePercent'], errors='coerce')
        # Handle potential issues with totalRevenue if it's read as string with commas
        if 'totalRevenue' in df.columns and df['totalRevenue'].dtype == 'object':
            df['totalRevenue'] = df['totalRevenue'].astype(str).str.replace(',', '', regex=False)
            df['totalRevenue'] = pd.to_numeric(df['totalRevenue'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Error: The LLM generated data file '{LLM_FILE_NAME}' was not found. Ensure it is in the '{DATA_DIR}' subdirectory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading LLM data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_sff_data():
    """Loads the SFF dataset from the specified Excel file."""
    try:
        df = pd.read_excel(SFF_FILE_NAME)
        expected_cols = ['Pureplay Status', 'SDS', 'Alt SDS', 'Client Name', 'Themes',
                         'Sub Theme', 'TLN', 'SLN', 'CSID', 'additional CSID', 'BIC']
        if not all(col in df.columns for col in expected_cols):
            st.error(f"SFF data ('{SFF_FILE_NAME}') is missing one or more expected columns. Expected: {expected_cols}. Found: {df.columns.tolist()}")
            return pd.DataFrame()
        return df
    except FileNotFoundError:
        st.error(f"Error: The SFF data file '{SFF_FILE_NAME}' was not found. Ensure it is in the '{DATA_DIR}' subdirectory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading SFF data: {e}")
        return pd.DataFrame()

def create_green_revenue_dataset(pcaf_df, llm_df):
    """
    Prepares the GREEN_REVENUE dataset.
    """
    if pcaf_df.empty or llm_df.empty:
        st.warning("PCAF or LLM data is empty. Cannot create GREEN_REVENUE dataset.")
        return pd.DataFrame()

    pcaf_unique_subset_cols = [
        'cob_date', 'productype', 'legal_entity', 'counterparty_id',
        'counterparty_name', 'parent_id', 'group_id', 'group_name',
        'bic_code', 'country_code'
    ]
    if not all(col in pcaf_df.columns for col in pcaf_unique_subset_cols):
        missing_pcaf_cols = [col for col in pcaf_unique_subset_cols if col not in pcaf_df.columns]
        st.error(f"One or more columns for PCAF unique selection not found in PCAF data: {missing_pcaf_cols}")
        return pd.DataFrame()
    pcaf_selected_df = pcaf_df[pcaf_unique_subset_cols].drop_duplicates()

    pcaf_selected_df['join_key_pcaf'] = pcaf_selected_df['counterparty_name'].astype(str).str.strip().str.lower()
    llm_df_copy = llm_df.copy()
    llm_df_copy['join_key_llm'] = llm_df_copy['companyName'].astype(str).str.strip().str.lower()

    green_revenue_df = pd.merge(
        pcaf_selected_df,
        llm_df_copy,
        left_on='join_key_pcaf',
        right_on='join_key_llm',
        how='inner'
    )

    if green_revenue_df.empty:
        st.warning("Join between PCAF unique data and LLM data resulted in an empty dataframe. "
                   "Check join keys (PCAF 'counterparty_name' vs LLM 'companyName') and data content.")
        return pd.DataFrame()

    final_green_revenue_attributes = [
        'cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name',
        'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code',
        'year', 'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources'
    ]
    cols_to_select = [col for col in final_green_revenue_attributes if col in green_revenue_df.columns]
    if len(cols_to_select) != len(final_green_revenue_attributes):
        missing_cols = set(final_green_revenue_attributes) - set(cols_to_select)
        st.warning(f"Some columns expected in GREEN_REVENUE were not found after merge: {missing_cols}. Using available columns.")
    green_revenue_df = green_revenue_df[cols_to_select].copy()

    green_revenue_df['pure_play_flag'] = 'N' # Default
    green_revenue_df['greenRevenuePercent'] = pd.to_numeric(green_revenue_df['greenRevenuePercent'], errors='coerce')
    green_revenue_df.loc[green_revenue_df['greenRevenuePercent'] >= 50, 'pure_play_flag'] = 'Y'
    # For <50%, they remain 'N'. If GR% is NaN, it also remains 'N' (or could be 'N/A' if explicitly handled).
    # As per Task 4 using pure_play_flag="N", this setup is fine.
    # If NaN greenRevenuePercent should not be 'N', then add:
    # green_revenue_df.loc[green_revenue_df['greenRevenuePercent'].isna(), 'pure_play_flag'] = 'N/A'

    return green_revenue_df

def compare_datasets_for_overlap(green_revenue_df, sff_df, filter_pure_play_flag_value):
    """
    Compares GREEN_REVENUE (filtered by a specific pure_play_flag value) with SFF_DATA.
    Matching primarily on counterparty_id (SDS in SFF).

    Args:
        green_revenue_df (pd.DataFrame): The full GREEN_REVENUE dataset.
        sff_df (pd.DataFrame): The SFF_DATA dataset.
        filter_pure_play_flag_value (str): The value to filter 'pure_play_flag' on (e.g., "Y" or "N").

    Returns:
        tuple: (overlap_df, identified_sff_only_df, unidentified_gr_only_df)
    """
    if green_revenue_df.empty:
        st.warning("GREEN_REVENUE dataframe is empty for comparison.")
        sff_cols = sff_df.columns if not sff_df.empty else []
        return pd.DataFrame(), pd.DataFrame(columns=sff_cols) if not sff_df.empty else pd.DataFrame(), pd.DataFrame()

    # Filter GREEN_REVENUE
    gr_filtered = green_revenue_df[green_revenue_df['pure_play_flag'] == filter_pure_play_flag_value].copy()
    filter_desc = f"Pure Play Flag = '{filter_pure_play_flag_value}'"

    if gr_filtered.empty:
        st.info(f"No companies in GREEN_REVENUE match the filter: {filter_desc}.")
        return pd.DataFrame(), sff_df.copy() if not sff_df.empty else pd.DataFrame(), pd.DataFrame()

    if sff_df.empty:
        st.warning("SFF_DATA is empty. Comparison will primarily show Un-Identified Clients from the filtered GREEN_REVENUE.")
        gr_cols_for_unidentified = [
            'cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name',
            'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code', 'year',
            'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag'
        ]
        # Ensure all requested columns exist in gr_filtered
        existing_cols = [col for col in gr_cols_for_unidentified if col in gr_filtered.columns]
        return pd.DataFrame(), pd.DataFrame(), gr_filtered[existing_cols]


    if 'counterparty_id' not in gr_filtered.columns:
        st.error("Column 'counterparty_id' not found in filtered GREEN_REVENUE. Cannot perform comparison.")
        return pd.DataFrame(), sff_df.copy() if not sff_df.empty else pd.DataFrame(), pd.DataFrame() # Return SFF as identified
    if 'SDS' not in sff_df.columns:
        st.error("Column 'SDS' not found in SFF_DATA. Cannot perform comparison.")
        # Return GR_filtered as unidentified
        gr_cols_for_unidentified = [
            'cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name',
            'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code', 'year',
            'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag'
        ]
        existing_cols = [col for col in gr_cols_for_unidentified if col in gr_filtered.columns]
        return pd.DataFrame(), pd.DataFrame(), gr_filtered[existing_cols]


    gr_filtered['norm_key_gr'] = gr_filtered['counterparty_id'].astype(str).str.strip().str.lower()
    sff_df_copy = sff_df.copy()
    sff_df_copy['norm_key_sff'] = sff_df_copy['SDS'].astype(str).str.strip().str.lower()

    merged_df = pd.merge(
        gr_filtered,
        sff_df_copy,
        left_on='norm_key_gr',
        right_on='norm_key_sff',
        how='outer',
        suffixes=('_gr', '_sff'),
        indicator=True
    )

    overlap_records = merged_df[merged_df['_merge'] == 'both']
    overlap_df_cols_gr = [
        'cob_date_gr', 'productype_gr', 'legal_entity_gr', 'counterparty_id_gr', 'counterparty_name_gr',
        'parent_id_gr', 'group_id_gr', 'group_name_gr', 'bic_code_gr', 'country_code_gr', 'year_gr',
        'totalRevenue_gr', 'greenRevenuePercent_gr', 'justification_gr', 'dataSources_gr', 'pure_play_flag_gr'
    ]
    # Ensure columns exist before selection and renaming
    overlap_df = overlap_records[[col for col in overlap_df_cols_gr if col in overlap_records.columns]].copy()
    overlap_df.columns = [col.replace('_gr', '') for col in overlap_df.columns]


    identified_sff_only_records = merged_df[merged_df['_merge'] == 'right_only']
    sff_original_cols = [
        'Pureplay Status', 'SDS', 'Alt SDS', 'Client Name', 'Themes',
        'Sub Theme', 'TLN', 'SLN', 'CSID', 'additional CSID', 'BIC'
    ]
    identified_sff_only_df = identified_sff_only_records[[col for col in sff_original_cols if col in identified_sff_only_records.columns]].copy()


    unidentified_gr_only_records = merged_df[merged_df['_merge'] == 'left_only']
    # Use the same column list as for overlap_df, but from the GR side of the left_only merge
    unidentified_gr_only_df_cols_gr = [
         'cob_date_gr', 'productype_gr', 'legal_entity_gr', 'counterparty_id_gr', 'counterparty_name_gr',
        'parent_id_gr', 'group_id_gr', 'group_name_gr', 'bic_code_gr', 'country_code_gr', 'year_gr',
        'totalRevenue_gr', 'greenRevenuePercent_gr', 'justification_gr', 'dataSources_gr', 'pure_play_flag_gr'
    ]
    unidentified_gr_only_df = unidentified_gr_only_records[[col for col in unidentified_gr_only_df_cols_gr if col in unidentified_gr_only_records.columns]].copy()
    unidentified_gr_only_df.columns = [col.replace('_gr', '') for col in unidentified_gr_only_df.columns]


    return overlap_df, identified_sff_only_df, unidentified_gr_only_df
