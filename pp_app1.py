# sustainable_finance_dashboard/app.py

import streamlit as st
import pandas as pd
from data_processing import (
    load_pcaf_data, load_llm_generated_data, load_sff_data,
    create_green_revenue_dataset, compare_datasets_for_overlap
)
from visualizations import (
    plot_green_revenue_distribution, plot_pure_play_flag_distribution,
    plot_revenue_by_attribute, plot_custom_venn_diagram,
    plot_comparison_metrics
)

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Sustainable Finance Dashboard v3", page_icon="🌍")

# --- Custom CSS ---
st.markdown("""
<style>
    /* Add modern styling */
    body {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem; padding-left: 3rem; padding-right: 3rem;}
    .stMetric value {font-size: 2rem !important; color: #2E8B57;} /* SeaGreen */
    .stMetric label {font-size: 0.9rem !important; color: #4A4A4A;}
    h1, h2, h3 {color: #004085;} /* Dark Blue */
    .stTabs [data-baseweb="tab-list"] {gap: 28px; border-bottom: 2px solid #004085;}
    .stTabs [data-baseweb="tab"] {height: 55px; white-space: pre-wrap; background-color: #E9ECEF;
                                 border-radius: 6px 6px 0px 0px; padding: 12px 18px; font-weight: 500;}
    .stTabs [aria-selected="true"] {background-color: #004085; color: white; border-bottom: 2px solid #004085;}
    .stDataFrame {border: 1px solid #dee2e6; border-radius: 0.25rem;}
    .stButton>button {border-radius: 20px; border: 2px solid #004085; background-color: white; color: #004085;
                      padding: 10px 24px; font-weight: bold;}
    .stButton>button:hover {border-color: #007bff; background-color: #007bff; color: white;}
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
with st.spinner("🌍 Loading and Verifying Datasets... This may take a moment."):
    pcaf_df_raw = load_pcaf_data()
    llm_df_raw = load_llm_generated_data() # Handles CSV
    sff_df_raw = load_sff_data()

st.sidebar.header("📊 Data Load Summary")
st.sidebar.metric("PCAF Records (Raw)", f"{len(pcaf_df_raw):,}" if pcaf_df_raw is not None and not pcaf_df_raw.empty else "0")
st.sidebar.metric("LLM Records (Raw)", f"{len(llm_df_raw):,}" if llm_df_raw is not None and not llm_df_raw.empty else "0")
st.sidebar.metric("SFF Records (Raw)", f"{len(sff_df_raw):,}" if sff_df_raw is not None and not sff_df_raw.empty else "0")

# --- GREEN_REVENUE Dataset Creation ---
green_revenue_df = pd.DataFrame()
if (pcaf_df_raw is not None and not pcaf_df_raw.empty and
    llm_df_raw is not None and not llm_df_raw.empty):
    with st.spinner("🌿 Processing GREEN_REVENUE dataset..."):
        green_revenue_df = create_green_revenue_dataset(pcaf_df_raw, llm_df_raw)
    if not green_revenue_df.empty:
        st.sidebar.metric("GREEN_REVENUE Records", f"{len(green_revenue_df):,}")
        pure_play_count = green_revenue_df[green_revenue_df['pure_play_flag'] == 'Y'].shape[0]
        st.sidebar.metric("Pure Play ('Y') in GR", f"{pure_play_count:,}")
        not_pure_play_count = green_revenue_df[green_revenue_df['pure_play_flag'] == 'N'].shape[0]
        st.sidebar.metric("Not Pure Play ('N') in GR", f"{not_pure_play_count:,}")
    else:
        st.sidebar.warning("GREEN_REVENUE dataset is empty. Check data and join logic.")
else:
    st.sidebar.error("Core data (PCAF/LLM) missing. GREEN_REVENUE not created.")

# --- Main Dashboard ---
st.title("🌍 Sustainable Finance Intelligence Dashboard")
st.markdown("Analyzing Green Revenue, Pure Play Classifications, and Company Overlaps")
st.markdown("---")

# --- Section 1: Overall Metrics & Statistics ---
st.header("📈 Overall GREEN_REVENUE Insights")
if not green_revenue_df.empty:
    col1a, col1b = st.columns([0.6, 0.4]) # Adjust column ratios
    with col1a:
        plot_green_revenue_distribution(green_revenue_df, column='greenRevenuePercent')
    with col1b:
        plot_pure_play_flag_distribution(green_revenue_df, column='pure_play_flag')

    st.subheader("💰 Revenue Breakdown (Top 10 from GREEN_REVENUE)")
    gr_df_vis = green_revenue_df.copy()
    gr_df_vis['totalRevenue'] = pd.to_numeric(gr_df_vis['totalRevenue'], errors='coerce')
    col2a, col2b = st.columns(2)
    with col2a:
        plot_revenue_by_attribute(gr_df_vis, attribute_col='country_code', top_n=10, title_prefix="Total Revenue")
    with col2b:
        plot_revenue_by_attribute(gr_df_vis, attribute_col='productype', top_n=10, title_prefix="Total Revenue")
    st.markdown("---")

    # Venn Diagram
    st.subheader("🌐 Company Set Overlaps (Interpreted from Image)")
    st.markdown("""
    - **PCAF (Blue):** All unique companies from the PCAF dataset.
    - **LLM Green Revenue >=50% (Orange):** Unique companies from LLM data with Green Revenue $\ge 50\%$.
    - **SFF Pure Play List (Green):** All unique companies from the SFF dataset.
    """)
    if (pcaf_df_raw is not None and not pcaf_df_raw.empty and
        llm_df_raw is not None and not llm_df_raw.empty and
        sff_df_raw is not None and not sff_df_raw.empty):
        pcaf_companies = set(pcaf_df_raw['counterparty_name'].astype(str).str.strip().str.lower().unique())
        llm_copy_for_venn = llm_df_raw.copy()
        llm_copy_for_venn['greenRevenuePercent'] = pd.to_numeric(llm_copy_for_venn['greenRevenuePercent'], errors='coerce')
        llm_pp_companies = set(llm_copy_for_venn[llm_copy_for_venn['greenRevenuePercent'] >= 50]['companyName'].astype(str).str.strip().str.lower().unique())
        sff_companies = set(sff_df_raw['Client Name'].astype(str).str.strip().str.lower().unique())

        plot_custom_venn_diagram(
            set1_size=len(pcaf_companies), set2_size=len(llm_pp_companies), set3_size=len(sff_companies),
            set12_overlap=len(pcaf_companies.intersection(llm_pp_companies)),
            set13_overlap=len(pcaf_companies.intersection(sff_companies)),
            set23_overlap=len(llm_pp_companies.intersection(sff_companies)),
            set123_overlap=len(pcaf_companies.intersection(llm_pp_companies).intersection(sff_companies)),
            set_labels=('PCAF Companies', 'LLM (GR ≥50%)', 'SFF Pure Play List'),
            title="Company Data Overlaps (Interpreted)"
        )
    else:
        st.info("One or more raw datasets are missing for Venn diagram generation.")
    st.markdown("---")
else:
    st.warning("GREEN_REVENUE dataset is empty. Most insights cannot be displayed.")

# --- Comparison Sections Helper ---
def display_comparison_section_tabs(section_title_main, df_overlap, df_identified, df_unidentified, section_key_suffix):
    st.subheader(section_title_main)
    plot_comparison_metrics(len(df_overlap), len(df_identified), len(df_unidentified), section_title_main)

    tab_cols_gr = [
        'cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name',
        'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code', 'year',
        'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag'
    ]
    tab_cols_sff = [
        'Pureplay Status', 'SDS', 'Alt SDS', 'Client Name', 'Themes',
        'Sub Theme', 'TLN', 'SLN', 'CSID', 'additional CSID', 'BIC'
    ]

    # Ensure dataframes have columns before trying to select
    overlap_display_df = df_overlap[[col for col in tab_cols_gr if col in df_overlap.columns]] if not df_overlap.empty else pd.DataFrame(columns=tab_cols_gr)
    identified_display_df = df_identified[[col for col in tab_cols_sff if col in df_identified.columns]] if not df_identified.empty else pd.DataFrame(columns=tab_cols_sff)
    unidentified_display_df = df_unidentified[[col for col in tab_cols_gr if col in df_unidentified.columns]] if not df_unidentified.empty else pd.DataFrame(columns=tab_cols_gr)


    tab1, tab2, tab3 = st.tabs([
        f"🔗 Overlap ({len(overlap_display_df):,})",
        f"🎯 Identified (SFF only) ({len(identified_display_df):,})",
        f"❓ Un-Identified (GR only) ({len(unidentified_display_df):,})"
    ])

    with tab1:
        st.markdown(f"###### Companies in both filtered GREEN_REVENUE and SFF_DATA")
        st.dataframe(overlap_display_df, height=350, use_container_width=True)
    with tab2:
        st.markdown(f"###### Companies in SFF_DATA but not in filtered GREEN_REVENUE")
        st.dataframe(identified_display_df, height=350, use_container_width=True)
    with tab3:
        st.markdown(f"###### Companies in filtered GREEN_REVENUE but not in SFF_DATA")
        st.dataframe(unidentified_display_df, height=350, use_container_width=True)
    st.markdown("---")

# --- Section 2: Pure Play (>=50%) Comparison ---
st.header("✅ Pure Play in GREEN_REVENUE (GR% >= 50%) vs SFF Data")
if (not green_revenue_df.empty and sff_df_raw is not None and not sff_df_raw.empty):
    overlap_pp, identified_pp, unidentified_pp = compare_datasets_for_overlap(
        green_revenue_df, sff_df_raw, filter_pure_play_flag_value="Y"
    )
    display_comparison_section_tabs(
        "Comparison: GR (Pure Play Flag = 'Y') vs SFF",
        overlap_pp, identified_pp, unidentified_pp, "ppY"
    )
elif green_revenue_df.empty:
    st.warning("GREEN_REVENUE dataset is empty. Cannot perform this comparison.")
elif sff_df_raw is None or sff_df_raw.empty:
    st.warning("SFF_DATA is empty. Full comparison for Pure Play ('Y') is limited.")
    if not green_revenue_df.empty:
        gr_filtered_pp_y = green_revenue_df[green_revenue_df['pure_play_flag'] == 'Y']
        st.markdown("###### Companies in GREEN_REVENUE (Flag='Y') - SFF Data was empty:")
        st.dataframe(gr_filtered_pp_y, height=300, use_container_width=True)
st.markdown("---")

# --- Section 3: Not Pure Play (<50%, Flag='N') Comparison ---
st.header("❌ Not Pure Play in GREEN_REVENUE (GR% < 50%, Flag = 'N') vs SFF Data")
if (not green_revenue_df.empty and sff_df_raw is not None and not sff_df_raw.empty):
    # Using filter_pure_play_flag_value="N" as per clarified Task 4
    overlap_npp, identified_npp, unidentified_npp = compare_datasets_for_overlap(
        green_revenue_df, sff_df_raw, filter_pure_play_flag_value="N"
    )
    display_comparison_section_tabs(
        "Comparison: GR (Pure Play Flag = 'N') vs SFF",
        overlap_npp, identified_npp, unidentified_npp, "ppN"
    )
elif green_revenue_df.empty:
    st.warning("GREEN_REVENUE dataset is empty. Cannot perform this comparison.")
elif sff_df_raw is None or sff_df_raw.empty:
    st.warning("SFF_DATA is empty. Full comparison for Not Pure Play ('N') is limited.")
    if not green_revenue_df.empty:
        gr_filtered_pp_n = green_revenue_df[green_revenue_df['pure_play_flag'] == 'N']
        st.markdown("###### Companies in GREEN_REVENUE (Flag='N') - SFF Data was empty:")
        st.dataframe(gr_filtered_pp_n, height=300, use_container_width=True)
st.markdown("---")


# --- Data Viewer Expander ---
with st.expander("🔍 Explore Raw & Processed Data Samples (First 5 Rows)"):
    if pcaf_df_raw is not None and not pcaf_df_raw.empty:
        st.subheader("PCAF Data (Raw)")
        st.dataframe(pcaf_df_raw.head())
    if llm_df_raw is not None and not llm_df_raw.empty:
        st.subheader("LLM Generated Data (Raw CSV)")
        st.dataframe(llm_df_raw.head())
    if sff_df_raw is not None and not sff_df_raw.empty:
        st.subheader("SFF Data (Raw)")
        st.dataframe(sff_df_raw.head())
    if not green_revenue_df.empty:
        st.subheader("GREEN_REVENUE Dataset (Processed)")
        st.dataframe(green_revenue_df.head())

st.sidebar.markdown("---")
st.sidebar.info(f"Sustainable Finance Dashboard | Version 3.0\nLast Refreshed: {pd.Timestamp.now(tz='America/New_York').strftime('%b %d, %Y %H:%M %Z')}")
