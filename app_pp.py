# app.py
# Main Streamlit application file

import streamlit as st
import pandas as pd

# Import functions from other modules
from data_loader import load_all_data, preprocess_data, create_green_revenue_dataset
from section_general_metrics import display_general_metrics
from section_pure_play import display_pure_play_analysis
from section_non_pure_play import display_non_pure_play_analysis

# Page Configuration
st.set_page_config(layout="wide", page_title="Sustainable Finance Dashboard")

# --- Main Application Flow ---
def main():
    st.title("Sustainable Finance Insights Dashboard")
    st.markdown("""
    This dashboard provides an analysis of company data based on PCAF, LLM-generated insights, 
    and Sustainable Finance Framework (SFF) pure play lists.
    """)

    # --- Data Loading and Preprocessing ---
    st.sidebar.header("Data Loading Options")
    # In a real scenario, you might have file uploaders or fixed paths
    # For this structure, we assume data files are in a 'data' subdirectory
    # and their names are fixed as per the requirements.

    # Placeholder for data loading status
    data_load_state = st.text("Loading data...")

    try:
        pcaf_data, llm_generated_data, sff_data = load_all_data(
            "data/group_client_coverage_dec24.xlsx",
            "data/llm_generated.csv",
            "data/Mar PP list_vF.xlsx"
        )
        
        # Preprocess join keys (as per requirements)
        pcaf_data_processed, llm_generated_data_processed = preprocess_data(pcaf_data, llm_generated_data)
        
        # Create GREEN_REVENUE dataset
        green_revenue_df = create_green_revenue_dataset(pcaf_data_processed, llm_generated_data_processed)
        
        data_load_state.text("Data loaded successfully!")
        st.sidebar.success("All datasets loaded and processed.")

        # Display some basic info about loaded data (optional)
        if st.sidebar.checkbox("Show Raw Data Summary"):
            st.subheader("Raw Data Summary")
            st.write("PCAF Dataframe head:", pcaf_data.head())
            st.write("LLM Generated Dataframe head:", llm_generated_data.head())
            st.write("SFF Dataframe head:", sff_data.head())
            st.write("GREEN_REVENUE Dataframe head:", green_revenue_df.head())

    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}. Please ensure data files are in the 'data' folder.")
        st.warning("The dashboard requires 'group_client_coverage_dec24.xlsx', 'llm_generated.csv', and 'Mar PP list_vF.xlsx' in a 'data' subfolder to function.")
        data_load_state.text("Data loading failed.")
        return # Stop execution if data loading fails
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading or processing: {e}")
        data_load_state.text("Data loading/processing failed.")
        return

    # --- Dashboard Sections --- #
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the dashboard section",
        ["General Metrics & Statistics", 
         "Pure Play Analysis (Green Revenue >= 50%)", 
         "Non-Pure Play Analysis (Green Revenue < 50%)"]
    )

    if app_mode == "General Metrics & Statistics":
        display_general_metrics(pcaf_data, llm_generated_data, sff_data, green_revenue_df)
    elif app_mode == "Pure Play Analysis (Green Revenue >= 50%)":
        display_pure_play_analysis(green_revenue_df, sff_data)
    elif app_mode == "Non-Pure Play Analysis (Green Revenue < 50%)":
        display_non_pure_play_analysis(green_revenue_df, sff_data)

    st.sidebar.info("Dashboard developed by Manus AI.")

if __name__ == "__main__":
    main()

