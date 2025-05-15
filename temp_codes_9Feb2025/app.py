import streamlit as st
from data_loader import load_and_preprocess_data, create_green_revenue_dataset
from analytics import calculate_venn_stats, compare_datasets, get_industry_analysis
from visualization import plot_three_circle_venn, plot_green_revenue_distribution, plot_country_analysis
from components.metrics import display_key_metrics, display_comparison_metrics
from components.tables import display_data_table

# Page config
st.set_page_config(
    page_title="Sustainable Finance Dashboard",
    layout="wide"
)

@st.cache_data
def load_data():
    pcaf_data, llm_data, sff_data = load_and_preprocess_data()
    green_revenue = create_green_revenue_dataset(pcaf_data, llm_data)
    return green_revenue, sff_data

def main():
    st.title("🌍 Sustainable Finance Dashboard")
    
    # Load data
    green_revenue, sff_data = load_data()
    venn_stats = calculate_venn_stats(green_revenue, sff_data)
    
    # Overview metrics
    st.header("📊 Overview Metrics")
    display_key_metrics(venn_stats)
    
    # Venn diagram
    st.header("🔵 Venn Diagram Analysis")
    plot_three_circle_venn(venn_stats)
    
    # Visualizations
    st.header("📈 Green Revenue Analysis")
    plot_green_revenue_distribution(green_revenue)
    plot_country_analysis(green_revenue)
    
    # Pure Play comparison
    st.header("🔍 Pure Play Comparison (≥50%)")
    overlap, sff_only, green_only = compare_datasets(green_revenue, sff_data, 'Y')
    display_comparison_metrics(len(overlap), len(sff_only), len(green_only), "Pure Play")
    
    tab1, tab2, tab3 = st.tabs(["Overlap", "SFF Only", "Green Only"])
    with tab1:
        display_data_table(
            overlap,
            ['counterparty_name', 'greenRevenuePercent', 'country_code', 'bic_code'],
            "Companies in both datasets"
        )
    with tab2:
        display_data_table(
            sff_only,
            ['client_name', 'Themes', 'Sub Theme', 'bic'],
            "Companies only in SFF"
        )
    with tab3:
        display_data_table(
            green_only,
            ['counterparty_name', 'greenRevenuePercent', 'country_code'],
            "Pure Play companies not in SFF"
        )
    
    # Industry analysis
    st.header("🏭 Industry Analysis")
    industry_stats = get_industry_analysis(green_revenue)
    st.dataframe(industry_stats, use_container_width=True)

if __name__ == "__main__":
    main()
