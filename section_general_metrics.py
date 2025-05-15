# section_general_metrics.py
# Displays general metrics and statistics, including the Venn diagram.

import streamlit as st
import pandas as pd
from matplotlib_venn import venn3 #, venn3_circles # venn3_circles is for outlines only
import matplotlib.pyplot as plt

# Make sure this function signature matches how it's called in app.py
def display_general_metrics(pcaf_data_processed: pd.DataFrame,
                            llm_generated_data_processed: pd.DataFrame,
                            sff_data_processed: pd.DataFrame, # Expecting SFF data already processed with join keys
                            green_revenue_df: pd.DataFrame):
    st.header("General Metrics & Statistics")

    st.subheader("Company Overlap Analysis (Venn Diagram)")
    st.markdown("""
    This Venn diagram illustrates the overlap between companies from three key datasets:
    1.  **PCAF Data**: Unique companies from the PCAF dataset (`group_client_coverage_dec24.xlsx`).
    2.  **LLM Pure Play**: Unique companies from the LLM-generated dataset (`llm_generated.csv`) identified as "Pure Play" (i.e., `greenRevenuePercent >= 50%`).
    3.  **SFF Data**: Unique Pure Play companies from the Sustainable Finance Framework dataset (`Mar PP list_vF.xlsx`).
    
    Company matching is based on preprocessed company names/identifiers.
    """)

    try:
        # Set 1: PCAF Companies
        if 'join_key_pcaf' not in pcaf_data_processed.columns:
            st.error("Venn Diagram Error: `join_key_pcaf` not found in processed PCAF data.")
            return
        set_pcaf = set(pcaf_data_processed['join_key_pcaf'].dropna().unique())

        # Set 2: LLM Pure Play Companies (greenRevenuePercent >= 50%)
        if 'join_key_llm' not in llm_generated_data_processed.columns or \
           'greenRevenuePercent' not in llm_generated_data_processed.columns:
            st.error("Venn Diagram Error: Required columns (`join_key_llm`, `greenRevenuePercent`) for LLM Pure Play set not found.")
            return
        
        # Ensure 'greenRevenuePercent' is numeric for comparison
        llm_df_copy = llm_generated_data_processed.copy()
        llm_df_copy['greenRevenuePercent_numeric'] = pd.to_numeric(llm_df_copy['greenRevenuePercent'], errors='coerce')
        
        llm_pure_play_df = llm_df_copy[
            llm_df_copy['greenRevenuePercent_numeric'] >= 50
        ]
        set_llm_pure_play = set(llm_pure_play_df['join_key_llm'].dropna().unique())

        # Set 3: SFF Companies
        if 'join_key_sff_name' not in sff_data_processed.columns:
            st.error("Venn Diagram Error: `join_key_sff_name` not found in processed SFF data.")
            st.info("Ensure SFF data is processed using `preprocess_sff_keys` from `data_loader.py` before being passed to this function.")
            return
        set_sff = set(sff_data_processed['join_key_sff_name'].dropna().unique())

        # Calculate Venn diagram segment sizes
        # Order for venn3: (Abc, aBc, ABc, abC, AbC, aBC, ABC)
        # A = set_pcaf, B = set_llm_pure_play, C = set_sff
        s100 = len(set_pcaf - set_llm_pure_play - set_sff)  # PCAF only
        s010 = len(set_llm_pure_play - set_pcaf - set_sff)  # LLM_PP only
        s110 = len((set_pcaf & set_llm_pure_play) - set_sff) # PCAF and LLM_PP, not SFF
        s001 = len(set_sff - set_pcaf - set_llm_pure_play)  # SFF only
        s101 = len((set_pcaf & set_sff) - set_llm_pure_play) # PCAF and SFF, not LLM_PP
        s011 = len((set_llm_pure_play & set_sff) - set_pcaf) # LLM_PP and SFF, not PCAF
        s111 = len(set_pcaf & set_llm_pure_play & set_sff)   # All three (PCAF, LLM_PP, SFF)

        fig, ax = plt.subplots(figsize=(12, 8)) # Adjusted size for better readability
        v = venn3(subsets=(s100, s010, s110, s001, s101, s011, s111),
                  set_labels=('PCAF Data', 'LLM Pure Play (>=50%)', 'SFF Data'),
                  ax=ax)
        
        # Customizing labels for specific segments to match user's image interpretation
        # The user's image labels point to specific intersections involving PCAF data primarily.

        # Label for (PCAF and LLM_PP, not SFF) - region s110 (Id '110')
        if v.get_label_by_id('110'): 
            v.get_label_by_id('110').set_text(f'{s110}')
            # ax.text for descriptive label might be better if default is too cluttered

        # Label for (PCAF and SFF, not LLM_PP) - region s101 (Id '101')
        if v.get_label_by_id('101'): 
            v.get_label_by_id('101').set_text(f'{s101}')

        # Label for (LLM_PP and SFF, not PCAF) - region s011 (Id '011')
        if v.get_label_by_id('011'):
            v.get_label_by_id('011').set_text(f'{s011}')
            
        # Label for the intersection of all three - region s111 (Id '111')
        if v.get_label_by_id('111'): 
            v.get_label_by_id('111').set_text(f'{s111}')

        plt.title("Company Overlap: PCAF, LLM Pure Play, and SFF Datasets", fontsize=16)
        st.pyplot(fig)

        # Displaying the counts for the regions described in the user's Venn diagram image:
        st.markdown("#### Key Overlap Segments (Interpreted from User's Diagram Image):")
        st.markdown(f"- **Companies in PCAF & LLM Pure Play (>=50%) but NOT in SFF Data:** `{s110}`")
        st.markdown(f"- **Companies in PCAF & SFF Data but NOT in LLM Pure Play (>=50%):** `{s101}`")
        st.markdown(f"- **Companies common to LLM Pure Play (>=50%) & SFF Data:** `{s011 + s111}`")
        st.markdown(f"  - Common to LLM Pure Play (>=50%), SFF Data, AND PCAF Data: `{s111}`")
        st.markdown(f"  - Common to LLM Pure Play (>=50%) & SFF Data, but NOT in PCAF Data: `{s011}`")

    except Exception as e:
        st.error(f"An error occurred while generating the Venn diagram: {e}")
        st.exception(e) # Provides full traceback for debugging in Streamlit

    st.subheader("Other Potential Statistics and Charts")
    st.markdown("*(This section is a placeholder for additional general metrics, charts, and diagrams. Specific visualizations can be developed based on further analysis needs.)*")
    
    # Example: Basic counts from input dataframes
    st.write(f"Total unique company identifiers in processed PCAF Data: `{len(set_pcaf) if 'set_pcaf' in locals() else 'Error'}`")
    st.write(f"Total unique company identifiers in LLM Pure Play (>=50%) list: `{len(set_llm_pure_play) if 'set_llm_pure_play' in locals() else 'Error'}`")
    st.write(f"Total unique company identifiers in processed SFF Data: `{len(set_sff) if 'set_sff' in locals() else 'Error'}`")
    st.write(f"Total companies in the derived GREEN_REVENUE dataset: `{len(green_revenue_df)}`")
    if 'pure_play_flag' in green_revenue_df.columns:
        st.write(f"  - Companies in GREEN_REVENUE with pure_play_flag='Y': `{len(green_revenue_df[green_revenue_df['pure_play_flag']=='Y'])}`")
        st.write(f"  - Companies in GREEN_REVENUE with pure_play_flag='N': `{len(green_revenue_df[green_revenue_df['pure_play_flag']=='N'])}`")

    # Further ideas for charts:
    # - Distribution of 'greenRevenuePercent' from llm_generated_data_processed.
    # - Bar chart of companies by 'country_code' in pcaf_data_processed.
    # - Table summary of 'Themes' in sff_data_processed.

if __name__ == '__main__':
    # This section can be used for testing this module independently
    # You would need to create dummy DataFrames for pcaf_data_processed, 
    # llm_generated_data_processed, sff_data_processed, and green_revenue_df.
    st.info("Testing section_general_metrics.py module independently.")
    # Example dummy data (replace with more comprehensive data for real testing)
    dummy_pcaf = pd.DataFrame({'join_key_pcaf': ['a', 'b', 'c', 'd', 'e', 'f']})
    dummy_llm = pd.DataFrame({
        'join_key_llm': ['a', 'b', 'g', 'h', 'd', 'f'], 
        'greenRevenuePercent': [60, 40, 70, 80, 55, 20]
    })
    dummy_sff = pd.DataFrame({'join_key_sff_name': ['c', 'd', 'e', 'i', 'j', 'f']})
    dummy_green_revenue = pd.DataFrame({
        'companyName': ['a', 'd', 'f'], 
        'pure_play_flag': ['Y', 'Y', 'N']
    })

    display_general_metrics(dummy_pcaf, dummy_llm, dummy_sff, dummy_green_revenue)

