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
st.set_page_config(layout="wide", page_title="Sustainable Finance Dashboard v4", page_icon="🌿")

# --- Custom CSS ---
st.markdown("""
<style>
    body {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem; padding-left: 3rem; padding-right: 3rem;}
    .stMetric value {font-size: 2rem !important; color: #1E88E5;} /* Brighter Blue */
    .stMetric label {font-size: 0.9rem !important; color: #555;}
    h1 {color: #0D47A1;} /* Darker Blue for main title */
    h2, h3 {color: #1565C0;} /* Medium Blue for headers */
    .stTabs [data-baseweb="tab-list"] {gap: 28px; border-bottom: 3px solid #1565C0;}
    .stTabs [data-baseweb="tab"] {height: 55px; white-space: pre-wrap; background-color: #F5F5F5;
                                 border-radius: 8px 8px 0px 0px; padding: 12px 20px; font-weight: 500; color: #333;}
    .stTabs [aria-selected="true"] {background-color: #1565C0; color: white; border-bottom: 3px solid #0D47A1;}
    .stDataFrame {border: 1px solid #CFD8DC; border-radius: 0.3rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
with st.spinner("🌍 Loading and Verifying Datasets..."):
    pcaf_df_raw = load_pcaf_data()
    llm_df_raw = load_llm_generated_data()
    sff_df_raw = load_sff_data()

st.sidebar.header("📊 Data Load Summary")
st.sidebar.metric("PCAF Records", f"{len(pcaf_df_raw):,}" if pcaf_df_raw is not None and not pcaf_df_raw.empty else "0")
st.sidebar.metric("LLM Records", f"{len(llm_df_raw):,}" if llm_df_raw is not None and not llm_df_raw.empty else "0")
st.sidebar.metric("SFF Records", f"{len(sff_df_raw):,}" if sff_df_raw is not None and not sff_df_raw.empty else "0")

# --- GREEN_REVENUE Dataset Creation ---
green_revenue_df = pd.DataFrame()
if (pcaf_df_raw is not None and not pcaf_df_raw.empty and
    llm_df_raw is not None and not llm_df_raw.empty):
    with st.spinner("🌿 Processing GREEN_REVENUE dataset..."):
        green_revenue_df = create_green_revenue_dataset(pcaf_df_raw, llm_df_raw)
    if not green_revenue_df.empty:
        st.sidebar.metric("GREEN_REVENUE Records", f"{len(green_revenue_df):,}")
        st.sidebar.metric("GR: Pure Play ('Y')", f"{green_revenue_df[green_revenue_df['pure_play_flag'] == 'Y'].shape[0]:,}")
        st.sidebar.metric("GR: Not Pure Play ('N')", f"{green_revenue_df[green_revenue_df['pure_play_flag'] == 'N'].shape[0]:,}")
        # Optional: Count 'N/A' if you introduce it
        # st.sidebar.metric("GR: Flag 'N/A'", f"{green_revenue_df[green_revenue_df['pure_play_flag'] == 'N/A'].shape[0]:,}")
    else:
        st.sidebar.warning("GREEN_REVENUE empty. Check data/joins.")
else:
    st.sidebar.error("PCAF/LLM data missing. GREEN_REVENUE not created.")

# --- Main Dashboard ---
st.title("🌍 Sustainable Finance Intelligence Dashboard")
st.markdown("Analyzing Green Revenue, Pure Play Classifications, and Company Overlaps")
st.markdown("---")

# --- Section 1: Overall Metrics & Statistics ---
st.header("📈 Overall GREEN_REVENUE Insights")
if not green_revenue_df.empty:
    col1a, col1b = st.columns([0.6, 0.4])
    with col1a:
        plot_green_revenue_distribution(green_revenue_df, column='greenRevenuePercent')
    with col1b:
        plot_pure_play_flag_distribution(green_revenue_df, column='pure_play_flag')

    st.subheader("💰 Revenue Breakdown (Top 10 from GREEN_REVENUE)")
    gr_df_vis = green_revenue_df.copy()
    gr_df_vis['totalRevenue'] = pd.to_numeric(gr_df_vis['totalRevenue'], errors='coerce')
    col2a, col2b = st.columns(2)
    with col2a:
        plot_revenue_by_attribute(gr_df_vis, attribute_col='country_code', top_n=10)
    with col2b:
        plot_revenue_by_attribute(gr_df_vis, attribute_col='productype', top_n=10)
    st.markdown("---")

    st.subheader("🌐 Company Set Overlaps (Interpreted)")
    st.markdown("""
    - **PCAF (Blue):** All unique companies (by name) from the PCAF dataset.
    - **LLM Green Revenue >=50% (Orange):** Unique companies (by name) from LLM data with Green Revenue $\ge 50\%$.
    - **SFF Pure Play List (Green):** All unique companies (by name) from the SFF dataset.
    """)
    if (pcaf_df_raw is not None and not pcaf_df_raw.empty and
        llm_df_raw is not None and not llm_df_raw.empty and
        sff_df_raw is not None and not sff_df_raw.empty):
        
        pcaf_companies = set(robust_normalize_string_series(pcaf_df_raw['counterparty_name']).dropna().unique())
        
        llm_copy_for_venn = llm_df_raw.copy()
        llm_copy_for_venn['greenRevenuePercent'] = pd.to_numeric(llm_copy_for_venn['greenRevenuePercent'], errors='coerce')
        llm_pp_companies_series = robust_normalize_string_series(llm_copy_for_venn[llm_copy_for_venn['greenRevenuePercent'] >= 50]['companyName'])
        llm_pp_companies = set(llm_pp_companies_series.dropna().unique())
        
        sff_companies_series = robust_normalize_string_series(sff_df_raw['Client Name'])
        sff_companies = set(sff_companies_series.dropna().unique())

        plot_custom_venn_diagram(
            set1_size=len(pcaf_companies), set2_size=len(llm_pp_companies), set3_size=len(sff_companies),
            set12_overlap=len(pcaf_companies.intersection(llm_pp_companies)),
            set13_overlap=len(pcaf_companies.intersection(sff_companies)),
            set23_overlap=len(llm_pp_companies.intersection(sff_companies)),
            set123_overlap=len(pcaf_companies.intersection(llm_pp_companies).intersection(sff_companies)),
            set_labels=('PCAF Companies', 'LLM (GR ≥50%)', 'SFF Pure Play List'),
            title="Company Data Overlaps (Normalized Names)"
        )
    else:
        st.info("Raw data missing for Venn diagram.")
    st.markdown("---")
else:
    st.warning("GREEN_REVENUE dataset is empty. Most insights cannot be displayed.")

# --- Comparison Sections Helper ---
def display_comparison_section_tabs(section_title_main, df_overlap, df_identified, df_unidentified):
    st.subheader(section_title_main)
    plot_comparison_metrics(len(df_overlap), len(df_identified), len(df_unidentified), section_title_main)

    tab_cols_gr = ['cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name', 'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code', 'year', 'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag']
    tab_cols_sff = ['Pureplay Status', 'SDS', 'Alt SDS', 'Client Name', 'Themes', 'Sub Theme', 'TLN', 'SLN', 'CSID', 'additional CSID', 'BIC']

    overlap_display_df = df_overlap[[col for col in tab_cols_gr if col in df_overlap.columns]] if df_overlap is not None and not df_overlap.empty else pd.DataFrame(columns=tab_cols_gr)
    identified_display_df = df_identified[[col for col in tab_cols_sff if col in df_identified.columns]] if df_identified is not None and not df_identified.empty else pd.DataFrame(columns=tab_cols_sff)
    unidentified_display_df = df_unidentified[[col for col in tab_cols_gr if col in df_unidentified.columns]] if df_unidentified is not None and not df_unidentified.empty else pd.DataFrame(columns=tab_cols_gr)

    tab1, tab2, tab3 = st.tabs([f"🔗 Overlap ({len(overlap_display_df):,})", f"🎯 Identified (SFF only) ({len(identified_display_df):,})", f"❓ Un-Identified (GR only) ({len(unidentified_display_df):,})"])
    with tab1:
        st.markdown("###### Companies in both filtered GREEN_REVENUE and SFF_DATA (matched on Counterparty ID/SDS)")
        st.dataframe(overlap_display_df, height=350, use_container_width=True)
    with tab2:
        st.markdown("###### Companies in SFF_DATA but not in filtered GREEN_REVENUE")
        st.dataframe(identified_display_df, height=350, use_container_width=True)
    with tab3:
        st.markdown("###### Companies in filtered GREEN_REVENUE but not in SFF_DATA")
        st.dataframe(unidentified_display_df, height=350, use_container_width=True)
    st.markdown("---")

# --- Section 2: Pure Play (>=50%) Comparison ---
st.header("✅ Pure Play in GREEN_REVENUE (GR% >= 50%) vs SFF Data")
if not green_revenue_df.empty and (sff_df_raw is not None and not sff_df_raw.empty):
    overlap_pp, identified_pp, unidentified_pp = compare_datasets_for_overlap(green_revenue_df, sff_df_raw, filter_pure_play_flag_value="Y")
    display_comparison_section_tabs("Comparison: GR (Pure Play Flag = 'Y') vs SFF", overlap_pp, identified_pp, unidentified_pp)
elif green_revenue_df.empty:
    st.warning("GREEN_REVENUE empty. Cannot compare for Pure Play ('Y').")
elif sff_df_raw is None or sff_df_raw.empty:
    st.warning("SFF_DATA empty. Comparison for Pure Play ('Y') limited.")
    # ... (code to show only GR side if SFF is missing)
st.markdown("---")

# --- Section 3: Not Pure Play (<50%, Flag='N') Comparison ---
st.header("❌ Not Pure Play in GREEN_REVENUE (GR% < 50%, Flag = 'N') vs SFF Data")
if not green_revenue_df.empty and (sff_df_raw is not None and not sff_df_raw.empty):
    overlap_npp, identified_npp, unidentified_npp = compare_datasets_for_overlap(green_revenue_df, sff_df_raw, filter_pure_play_flag_value="N")
    display_comparison_section_tabs("Comparison: GR (Pure Play Flag = 'N') vs SFF", overlap_npp, identified_npp, unidentified_npp)
elif green_revenue_df.empty:
    st.warning("GREEN_REVENUE empty. Cannot compare for Not Pure Play ('N').")
elif sff_df_raw is None or sff_df_raw.empty:
    st.warning("SFF_DATA empty. Comparison for Not Pure Play ('N') limited.")
    # ... (code to show only GR side if SFF is missing)
st.markdown("---")

with st.expander("🔍 Explore Raw & Processed Data Samples (First 5 Rows)"):
    # ... (same as previous version)

st.sidebar.markdown("---")
st.sidebar.info(f"Sustainable Finance Dashboard | Version 4.0\nLast Refreshed: {pd.Timestamp.now(tz='America/New_York').strftime('%b %d, %Y %H:%M %Z')}")
