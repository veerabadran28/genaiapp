import streamlit as st
from data_loader import load_and_preprocess_data, create_green_revenue_dataset
from analytics import compare_datasets, calculate_venn_stats
from visualization import plot_venn_diagram, plot_green_revenue_distribution, plot_country_analysis
from components.tables import display_data_table

# Page configuration
st.set_page_config(
    page_title="Sustainable Finance Dashboard",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .main {background-color: #f5f5f5;}
        .stMetric {background-color: white; border-radius: 10px; padding: 15px;}
        .stMetric label {font-size: 1rem; color: #666;}
        .stMetric div {font-size: 1.5rem; color: #333;}
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache all datasets."""
    pcaf_data, llm_data, sff_data = load_and_preprocess_data()
    green_revenue = create_green_revenue_dataset(pcaf_data, llm_data)
    return green_revenue, sff_data

def main():
    """Main application function."""
    st.title("🌍 Sustainable Finance Dashboard")
    st.markdown("""
        This dashboard provides insights into green revenue classification and comparison 
        between LLM-generated data and the Sustainable Finance Framework (SFF) data.
    """)
    
    # Load data
    green_revenue, sff_data = load_data()
    
    # Overview metrics
    st.header("📊 Overview Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Companies", len(green_revenue))
    with col2:
        st.metric("Pure Play Companies (≥50%)", 
                 len(green_revenue[green_revenue['pure_play_flag'] == 'Y']))
    with col3:
        st.metric("SFF Pure Play Companies", len(sff_data))
    with col4:
        overlap = len(set(green_revenue[green_revenue['pure_play_flag'] == 'Y']['counterparty_id']) & 
                    set(sff_data['sds']))
        st.metric("Overlap Between Datasets", overlap)
    
    # Venn diagram section
    st.header("🔵 Venn Diagram Analysis")
    venn_stats = calculate_venn_stats(green_revenue, sff_data)
    plot_venn_diagram(venn_stats)
    
    # Additional visualizations
    st.header("📈 Green Revenue Analysis")
    plot_green_revenue_distribution(green_revenue)
    plot_country_analysis(green_revenue)
    
    # Pure Play comparison section
    st.header("🔍 Pure Play Comparison (≥50%)")
    pure_play_overlap, pure_play_sff_only, pure_play_green_only = compare_datasets(
        green_revenue, sff_data, pure_play_filter='Y'
    )
    
    tab1, tab2, tab3 = st.tabs(["Overlap", "Identified Clients (SFF Only)", "Un-Identified Clients (Green Only)"])
    
    with tab1:
        display_data_table(
            pure_play_overlap,
            columns=[
                'cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name',
                'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code', 'year',
                'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag'
            ],
            title="Companies in both GREEN_REVENUE and SFF_DATA",
            key="pure_play_overlap"
        )
    
    with tab2:
        display_data_table(
            pure_play_sff_only,
            columns=[
                'Pureplay Status', 'sds', 'Alt SDS', 'client_name', 'Themes',
                'Sub Theme', 'TLN', 'SLN', 'CSID', 'additional CSID', 'bic'
            ],
            title="Companies in SFF_DATA but not in GREEN_REVENUE",
            key="pure_play_sff_only"
        )
    
    with tab3:
        display_data_table(
            pure_play_green_only,
            columns=[
                'cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name',
                'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code', 'year',
                'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag'
            ],
            title="Companies in GREEN_REVENUE but not in SFF_DATA",
            key="pure_play_green_only"
        )
    
    # Non Pure Play comparison section
    st.header("🔍 Non Pure Play Comparison (<50%)")
    non_pure_overlap, non_pure_sff_only, non_pure_green_only = compare_datasets(
        green_revenue, sff_data, pure_play_filter='N'
    )
    
    tab4, tab5, tab6 = st.tabs(["Overlap", "Identified Clients (SFF Only)", "Un-Identified Clients (Green Only)"])
    
    with tab4:
        display_data_table(
            non_pure_overlap,
            columns=[
                'cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name',
                'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code', 'year',
                'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag'
            ],
            title="Non-Pure Play Companies in both GREEN_REVENUE and SFF_DATA",
            key="non_pure_overlap"
        )
    
    with tab5:
        display_data_table(
            non_pure_sff_only,
            columns=[
                'Pureplay Status', 'sds', 'Alt SDS', 'client_name', 'Themes',
                'Sub Theme', 'TLN', 'SLN', 'CSID', 'additional CSID', 'bic'
            ],
            title="Companies in SFF_DATA but not in GREEN_REVENUE (Non-Pure Play)",
            key="non_pure_sff_only"
        )
    
    with tab6:
        display_data_table(
            non_pure_green_only,
            columns=[
                'cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name',
                'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code', 'year',
                'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag'
            ],
            title="Non-Pure Play Companies in GREEN_REVENUE but not in SFF_DATA",
            key="non_pure_green_only"
        )

if __name__ == "__main__":
    main()
