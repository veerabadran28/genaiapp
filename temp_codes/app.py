# app.py
import streamlit as st
import os
from utils.data_processor import DataProcessor
from components.metrics import MetricsSection
from components.pure_play_section import PurePlaySection
from components.non_pure_play_section import NonPurePlaySection

# Set page configuration
st.set_page_config(
    page_title="Sustainable Finance Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    .dashboard-title {
        text-align: center;
        font-weight: bold;
        font-size: 36px;
        margin-bottom: 30px;
        color: #1E3D58;
    }
    .section-header {
        background-color: #1E3D58;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # App title
    st.markdown('<div class="dashboard-title">Sustainable Finance Dashboard</div>', unsafe_allow_html=True)
    
    # Initialize data processor
    data_processor = DataProcessor(
        pcaf_file="data/group_client_coverage_dec24.xlsx",
        llm_file="data/llm_generated.csv",
        sff_file="data/Mar PP list_vF.xlsx"
    )
    
    # Process data
    data_processor.process_data()
    
    # Create sections
    metrics_section = MetricsSection(data_processor)
    pure_play_section = PurePlaySection(data_processor)
    non_pure_play_section = NonPurePlaySection(data_processor)
    
    # Add sidebar with information
    with st.sidebar:
        st.title("Dataset Information")
        st.markdown("### PCAF Dataset")
        st.info(f"Records: {data_processor.pcaf_data_count:,}")
        
        st.markdown("### LLM Generated Dataset")
        st.info(f"Records: {data_processor.llm_data_count:,}")
        
        st.markdown("### SFF Dataset")
        st.info(f"Records: {data_processor.sff_data_count:,}")
        
        st.markdown("### Green Revenue Dataset")
        st.info(f"Records: {data_processor.green_revenue_count:,}")
        
        st.markdown("### Pure Play Companies")
        st.success(f"Records: {data_processor.pure_play_count:,}")
        
        st.markdown("### Non-Pure Play Companies")
        st.error(f"Records: {data_processor.non_pure_play_count:,}")
    
    # Display sections
    metrics_section.display()
    pure_play_section.display()
    non_pure_play_section.display()

if __name__ == "__main__":
    main()
