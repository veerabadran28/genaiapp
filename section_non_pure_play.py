# section_non_pure_play.py
# Displays analysis for companies classified as non-pure play in GREEN_REVENUE (<50%)

import streamlit as st
import pandas as pd
from data_loader import preprocess_join_column, preprocess_sff_keys # For key preprocessing

# Helper function to display data tables (re-used from section_pure_play, ideally in a utils.py)
def display_table(df, header):
    st.markdown(f"**{header} (Count: {len(df)})**")
    if not df.empty:
        st.dataframe(df)
    else:
        st.info("No data to display for this category.")

def display_non_pure_play_analysis(green_revenue_df: pd.DataFrame, sff_data_original: pd.DataFrame):
    st.header("Companies Classified as Non-Pure Play in GREEN REVENUE (<50%)")

    if green_revenue_df.empty or sff_data_original.empty:
        st.warning("Required data (GREEN_REVENUE or SFF_DATA) is not available or empty. Cannot perform analysis.")
        return

    # --- Data Preparation ---
    # 1. Filter GREEN_REVENUE for pure_play_flag="N"
    green_revenue_n = green_revenue_df[green_revenue_df["pure_play_flag"] == "N"].copy()
    if green_revenue_n.empty:
        st.info("No companies found in GREEN_REVENUE dataset with pure_play_flag = 'N'.")
        # Display empty tabs structure or a message
        tab1, tab2, tab3 = st.tabs(["Overlap", "Identified Clients (SFF only)", "Un-Identified Clients (GR only)"])
        with tab1: display_table(pd.DataFrame(), "Companies in both GREEN_REVENUE (N) and SFF_DATA")
        with tab2: display_table(pd.DataFrame(), "Companies in SFF_DATA but not in GREEN_REVENUE (N)")
        with tab3: display_table(pd.DataFrame(), "Companies in GREEN_REVENUE (N) but not in SFF_DATA")
        return

    # 2. Preprocess SFF_DATA keys
    sff_data_processed = preprocess_sff_keys(sff_data_original.copy())

    # 3. Prepare join keys for comparison
    if 'counterparty_name' in green_revenue_n.columns:
        green_revenue_n["join_key_gr_name"] = preprocess_join_column(green_revenue_n["counterparty_name"])
    else:
        st.error("Error: 'counterparty_name' not found in GREEN_REVENUE (N) data. Cannot perform comparison.")
        return
    
    if 'join_key_sff_name' not in sff_data_processed.columns:
        st.error("Error: 'join_key_sff_name' not found in processed SFF_DATA. Ensure 'Client Name' exists and is processed.")
        return

    # --- Perform Comparisons (Merge/Join) ---
    comparison_df = pd.merge(
        green_revenue_n,
        sff_data_processed,
        left_on="join_key_gr_name",
        right_on="join_key_sff_name",
        how="outer",
        suffixes=('_gr', '_sff')
    )

    # --- Categorize Companies ---
    overlap_df = comparison_df[comparison_df['join_key_gr_name'].notna() & comparison_df['join_key_sff_name'].notna()].copy()
    sff_only_df = comparison_df[comparison_df['join_key_gr_name'].isna() & comparison_df['join_key_sff_name'].notna()].copy()
    gr_n_only_df = comparison_df[comparison_df['join_key_sff_name'].isna() & comparison_df['join_key_gr_name'].notna()].copy()

    # --- Define Columns for Display ---
    cols_green_revenue = [
        'cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name',
        'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code', 'year',
        'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag'
    ]
    cols_sff = [
        'Pureplay Status', 'SDS', 'Alt SDS', 'Client Name', 'Themes', 'Sub Theme',
        'TLN', 'SLN', 'CSID', 'additional CSID', 'BIC'
    ]

    # For Overlap: select from green_revenue_n
    overlap_display = green_revenue_n[green_revenue_n['join_key_gr_name'].isin(overlap_df['join_key_gr_name'].dropna())]
    overlap_display = overlap_display[cols_green_revenue].drop_duplicates()

    # For SFF Only: select from sff_data_processed
    sff_only_display = sff_data_processed[sff_data_processed['join_key_sff_name'].isin(sff_only_df['join_key_sff_name'].dropna())]
    sff_only_display = sff_only_display[cols_sff].drop_duplicates()

    # For GR_N Only: select from green_revenue_n
    gr_n_only_display = green_revenue_n[green_revenue_n['join_key_gr_name'].isin(gr_n_only_df['join_key_gr_name'].dropna())]
    gr_n_only_display = gr_n_only_display[cols_green_revenue].drop_duplicates()

    # --- Display in Tabs ---
    tab1, tab2, tab3 = st.tabs(["Overlap", "Identified Clients (SFF only)", "Un-Identified Clients (GR_N only)"])

    with tab1:
        display_table(overlap_display, "Companies in both GREEN_REVENUE (N) and SFF_DATA")

    with tab2:
        display_table(sff_only_display, "Companies in SFF_DATA but not in GREEN_REVENUE (N)")

    with tab3:
        display_table(gr_n_only_display, "Companies in GREEN_REVENUE (N) but not in SFF_DATA")

    st.markdown("---_Additional metrics and visualizations for this section can be added here._---")

if __name__ == '__main__':
    st.info("Testing section_non_pure_play.py module independently.")
    # Create dummy data for testing
    dummy_green_revenue = pd.DataFrame({
        'counterparty_name': ['Company A', 'Company B', 'Company C', 'Company D', 'Company G'],
        'greenRevenuePercent': [60, 20, 40, 80, 10],
        'pure_play_flag': ['Y', 'N', 'N', 'Y', 'N'],
        'cob_date': [None]*5, 'productype': [None]*5, 'legal_entity': [None]*5, 
        'counterparty_id': ['A1','B1','C1','D1', 'G1'], 'parent_id': [None]*5, 'group_id': [None]*5, 
        'group_name': [None]*5, 'bic_code': [None]*5, 'country_code': [None]*5, 'year': [2023]*5,
        'totalRevenue': [1000]*5, 'justification': ['Test']*5, 'dataSources': ['Test']*5
    })
    dummy_sff = pd.DataFrame({
        'Client Name': ['Company C', 'Company E', 'Company B', 'Company F', 'Company H'],
        'SDS': ['C1_sff', 'E1_sff', 'B1_sff', 'F1_sff', 'H1_sff'],
        'Pureplay Status': ['Active']*5, 'Alt SDS': [None]*5, 'Themes': [None]*5, 
        'Sub Theme': [None]*5, 'TLN': [None]*5, 'SLN': [None]*5, 'CSID': [None]*5, 
        'additional CSID': [None]*5, 'BIC': [None]*5
    })

    display_non_pure_play_analysis(dummy_green_revenue, dummy_sff)

