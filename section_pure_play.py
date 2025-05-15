# section_pure_play.py
# Displays analysis for companies classified as pure play in GREEN_REVENUE (>=50%)

import streamlit as st
import pandas as pd
from data_loader import preprocess_join_column, preprocess_sff_keys # For key preprocessing

# Helper function to display data tables (placeholder for python-slickgrid)
def display_table(df, header):
    st.markdown(f"**{header} (Count: {len(df)})**")
    if not df.empty:
        # Using st.dataframe as a standard display. python-slickgrid is a JS library.
        # For advanced tables, consider streamlit-aggrid if available and suitable.
        st.dataframe(df)
    else:
        st.info("No data to display for this category.")

def display_pure_play_analysis(green_revenue_df: pd.DataFrame, sff_data_original: pd.DataFrame):
    st.header("Companies Classified as Pure Play in GREEN REVENUE (>=50%)")

    if green_revenue_df.empty or sff_data_original.empty:
        st.warning("Required data (GREEN_REVENUE or SFF_DATA) is not available or empty. Cannot perform analysis.")
        return

    # --- Data Preparation ---
    # 1. Filter GREEN_REVENUE for pure_play_flag="Y"
    green_revenue_y = green_revenue_df[green_revenue_df["pure_play_flag"] == "Y"].copy()
    if green_revenue_y.empty:
        st.info("No companies found in GREEN_REVENUE dataset with pure_play_flag = 'Y'.")
        # Display empty tabs structure or a message
        tab1, tab2, tab3 = st.tabs(["Overlap", "Identified Clients (SFF only)", "Un-Identified Clients (GR only)"])
        with tab1: display_table(pd.DataFrame(), "Companies in both GREEN_REVENUE (Y) and SFF_DATA")
        with tab2: display_table(pd.DataFrame(), "Companies in SFF_DATA but not in GREEN_REVENUE (Y)")
        with tab3: display_table(pd.DataFrame(), "Companies in GREEN_REVENUE (Y) but not in SFF_DATA")
        return

    # 2. Preprocess SFF_DATA keys if not already done (it should be, but good to ensure)
    # Assuming sff_data_original is the raw SFF data, we need to process its keys.
    # If it's already processed, this step might be redundant but safe.
    sff_data_processed = preprocess_sff_keys(sff_data_original.copy())

    # 3. Prepare join keys for comparison
    # GREEN_REVENUE uses 'counterparty_name'. SFF_DATA uses 'Client Name' (processed to 'join_key_sff_name').
    # We need a consistent join key. Let's use preprocessed counterparty_name from GREEN_REVENUE.
    if 'counterparty_name' in green_revenue_y.columns:
        green_revenue_y["join_key_gr_name"] = preprocess_join_column(green_revenue_y["counterparty_name"])
    else:
        st.error("Error: 'counterparty_name' not found in GREEN_REVENUE (Y) data. Cannot perform comparison.")
        return
    
    if 'join_key_sff_name' not in sff_data_processed.columns:
        st.error("Error: 'join_key_sff_name' not found in processed SFF_DATA. Ensure 'Client Name' exists and is processed.")
        return

    # --- Perform Comparisons (Merge/Join) ---
    # Using an outer join to find all categories: overlap, GR_Y only, SFF only
    comparison_df = pd.merge(
        green_revenue_y,
        sff_data_processed,
        left_on="join_key_gr_name",
        right_on="join_key_sff_name",
        how="outer",
        suffixes=('_gr', '_sff') # Suffixes to distinguish columns from both dataframes after merge
    )

    # --- Categorize Companies ---
    # Overlap: Companies present in both (join_key_gr_name is not null AND join_key_sff_name is not null)
    # This means the merge found a match on the keys.
    overlap_df = comparison_df[comparison_df['join_key_gr_name'].notna() & comparison_df['join_key_sff_name'].notna()].copy()

    # Identified Clients (SFF_DATA only, not in GREEN_REVENUE_Y)
    # These are rows where join_key_gr_name is null (meaning no match from green_revenue_y)
    # but join_key_sff_name is not null (meaning they came from sff_data_processed).
    sff_only_df = comparison_df[comparison_df['join_key_gr_name'].isna() & comparison_df['join_key_sff_name'].notna()].copy()

    # Un-Identified Clients (GREEN_REVENUE_Y only, not in SFF_DATA)
    # These are rows where join_key_sff_name is null (meaning no match from sff_data_processed)
    # but join_key_gr_name is not null (meaning they came from green_revenue_y).
    gr_y_only_df = comparison_df[comparison_df['join_key_sff_name'].isna() & comparison_df['join_key_gr_name'].notna()].copy()

    # --- Define Columns for Display (as per requirements) ---
    cols_green_revenue = [
        'cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name',
        'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code', 'year',
        'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag'
    ]
    # Adjust column names if suffixes were added and they are the ones to be displayed
    # For overlap_df, columns from green_revenue_y will have _gr suffix if there were conflicts.
    # However, the requirement lists original names. We should select from original green_revenue_y columns for these.
    # Let's select the required columns from the original dataframes based on the join keys found.

    # For Overlap: select from green_revenue_y using its join keys present in overlap_df
    overlap_display = green_revenue_y[green_revenue_y['join_key_gr_name'].isin(overlap_df['join_key_gr_name'].dropna())]
    overlap_display = overlap_display[cols_green_revenue].drop_duplicates()

    # For SFF Only: select from sff_data_processed using its join keys present in sff_only_df
    cols_sff = [
        'Pureplay Status', 'SDS', 'Alt SDS', 'Client Name', 'Themes', 'Sub Theme',
        'TLN', 'SLN', 'CSID', 'additional CSID', 'BIC'
    ]
    sff_only_display = sff_data_processed[sff_data_processed['join_key_sff_name'].isin(sff_only_df['join_key_sff_name'].dropna())]
    sff_only_display = sff_only_display[cols_sff].drop_duplicates()

    # For GR_Y Only: select from green_revenue_y using its join keys present in gr_y_only_df
    gr_y_only_display = green_revenue_y[green_revenue_y['join_key_gr_name'].isin(gr_y_only_df['join_key_gr_name'].dropna())]
    gr_y_only_display = gr_y_only_display[cols_green_revenue].drop_duplicates()

    # --- Display in Tabs ---
    tab1, tab2, tab3 = st.tabs(["Overlap", "Identified Clients (SFF only)", "Un-Identified Clients (GR_Y only)"])

    with tab1:
        display_table(overlap_display, "Companies in both GREEN_REVENUE (Y) and SFF_DATA")

    with tab2:
        display_table(sff_only_display, "Companies in SFF_DATA but not in GREEN_REVENUE (Y)")

    with tab3:
        display_table(gr_y_only_display, "Companies in GREEN_REVENUE (Y) but not in SFF_DATA")

    # Placeholder for other metrics, charts, diagrams for this section
    st.markdown("---_Additional metrics and visualizations for this section can be added here._---")

if __name__ == '__main__':
    st.info("Testing section_pure_play.py module independently.")
    # Create dummy data for testing
    dummy_green_revenue = pd.DataFrame({
        'counterparty_name': ['Company A', 'Company B', 'Company C', 'Company D'],
        'greenRevenuePercent': [60, 70, 40, 80],
        'pure_play_flag': ['Y', 'Y', 'N', 'Y'],
        'cob_date': [None]*4, 'productype': [None]*4, 'legal_entity': [None]*4, 
        'counterparty_id': ['A1','B1','C1','D1'], 'parent_id': [None]*4, 'group_id': [None]*4, 
        'group_name': [None]*4, 'bic_code': [None]*4, 'country_code': [None]*4, 'year': [2023]*4,
        'totalRevenue': [1000]*4, 'justification': ['Test']*4, 'dataSources': ['Test']*4
    })
    dummy_sff = pd.DataFrame({
        'Client Name': ['Company A', 'Company E', 'Company B', 'Company F'],
        'SDS': ['A1_sff', 'E1_sff', 'B1_sff', 'F1_sff'],
        'Pureplay Status': ['Active']*4, 'Alt SDS': [None]*4, 'Themes': [None]*4, 
        'Sub Theme': [None]*4, 'TLN': [None]*4, 'SLN': [None]*4, 'CSID': [None]*4, 
        'additional CSID': [None]*4, 'BIC': [None]*4
    })

    display_pure_play_analysis(dummy_green_revenue, dummy_sff)

