# data_loader.py
# Functions for loading and preprocessing data

import pandas as pd
import streamlit as st # For error/warning messages if needed, though primarily for data ops here

# --- Configuration for join key preprocessing ---
# As per requirements: "For any column used in a join, it must be converted to string, 
# have leading/trailing whitespace removed, and converted to a consistent case (lowercase) on both sides of the join"

def preprocess_join_column(series):
    """Applies required preprocessing to a join column (Series)."""
    return series.astype(str).str.strip().str.lower()

# --- Data Loading Functions ---

def load_pcaf_data(file_path):
    """Loads PCAF_DATA from the specified Excel file."""
    try:
        # Attributes: cob_date, productype, legal_entity, counterparty_id, counterparty_name, 
        # parent_id, group_id, group_name, bic_code, naics_code, country_code
        df = pd.read_excel(file_path)
        # Requirement: Select unique (cob_date, productype, legal_entity, counterparty_id, 
        # counterparty_name, parent_id, group_id, group_name, bic_code, country_code) from PCAF_DATA.
        # This implies dropping duplicates based on these columns if there are any.
        # The problem statement says "Select unique (...) from PCAF_DATA. Join this with LLM_GENERATED"
        # This suggests the uniqueness should be applied *before* the join.
        unique_cols = ['cob_date', 'productype', 'legal_entity', 'counterparty_id', 
                         'counterparty_name', 'parent_id', 'group_id', 'group_name', 
                         'bic_code', 'country_code'] # 'naics_code' is listed as an attribute but not in unique list in task 1.
                                                    # Assuming it should be included if it's part of the core identity.
                                                    # For now, sticking to the explicitly listed unique columns for selection.
        
        # Check if all unique_cols are present
        missing_cols = [col for col in unique_cols if col not in df.columns]
        if missing_cols:
            st.warning(f"PCAF Data: Missing columns required for unique selection: {missing_cols}. Proceeding with available columns.")
            # Filter unique_cols to only those present in df
            unique_cols = [col for col in unique_cols if col in df.columns]

        if unique_cols: # Proceed if there are columns to make it unique by
             df = df.drop_duplicates(subset=unique_cols)
        else:
            st.warning("PCAF Data: No columns available to define uniqueness as specified. Using raw data.")

        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please ensure it is in the 'data' directory.")
        raise
    except Exception as e:
        st.error(f"Error loading PCAF data from {file_path}: {e}")
        raise

def load_llm_generated_data(file_path):
    """Loads LLM_GENERATED data from the specified CSV file."""
    try:
        # Attributes: companyName, year, totalRevenue, greenRevenuePercent, justification, dataSources
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please ensure it is in the 'data' directory.")
        raise
    except Exception as e:
        st.error(f"Error loading LLM Generated data from {file_path}: {e}")
        raise

def load_sff_data(file_path):
    """Loads SFF_DATA from the specified Excel file."""
    try:
        # Attributes: Pureplay Status, SDS, Alt SDS, Client Name, Themes, Sub Theme, TLN, SLN, CSID, additional CSID, BIC
        df = pd.read_excel(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File not found: {file_path}. Please ensure it is in the 'data' directory.")
        raise
    except Exception as e:
        st.error(f"Error loading SFF data from {file_path}: {e}")
        raise

def load_all_data(pcaf_file, llm_file, sff_file):
    """Loads all three datasets."""
    pcaf_data = load_pcaf_data(pcaf_file)
    llm_generated_data = load_llm_generated_data(llm_file)
    sff_data = load_sff_data(sff_file)
    return pcaf_data, llm_generated_data, sff_data

# --- Data Preprocessing and Dataset Creation ---

def preprocess_data(pcaf_df, llm_df):
    """Preprocesses join keys for PCAF and LLM dataframes."""
    # Make copies to avoid SettingWithCopyWarning if these DFs are used elsewhere
    pcaf_df_processed = pcaf_df.copy()
    llm_df_processed = llm_df.copy()

    # Preprocess join keys as per requirements
    # Join on trim(lower(PCAF_DATA.counterparty_name)) and trim(lower(LLM_GENERATED.companyName))
    if 'counterparty_name' in pcaf_df_processed.columns:
        pcaf_df_processed['join_key_pcaf'] = preprocess_join_column(pcaf_df_processed['counterparty_name'])
    else:
        st.error("Critical error: 'counterparty_name' not found in PCAF data. Cannot create join key.")
        raise KeyError("'counterparty_name' not found in PCAF_DATA")

    if 'companyName' in llm_df_processed.columns:
        llm_df_processed['join_key_llm'] = preprocess_join_column(llm_df_processed['companyName'])
    else:
        st.error("Critical error: 'companyName' not found in LLM_GENERATED data. Cannot create join key.")
        raise KeyError("'companyName' not found in LLM_GENERATED_DATA")
        
    return pcaf_df_processed, llm_df_processed

def create_green_revenue_dataset(pcaf_df_processed, llm_df_processed):
    """Creates the GREEN_REVENUE dataset by joining PCAF and LLM data and adding pure_play_flag."""
    
    # Perform the join
    # Attributes for GREEN_REVENUE: cob_date, productype, legal_entity, counterparty_id, 
    # counterparty_name, parent_id, group_id, group_name, bic_code, country_code, 
    # year, totalRevenue, greenRevenuePercent, justification, dataSources.
    
    green_revenue_df = pd.merge(
        pcaf_df_processed, 
        llm_df_processed, 
        left_on='join_key_pcaf', 
        right_on='join_key_llm',
        how='inner' # Assuming inner join based on the context of finding matches
    )

    # Create "pure_play_flag"
    # Value as "Y" if greenRevenuePercent >=50.
    if 'greenRevenuePercent' in green_revenue_df.columns:
        # Ensure greenRevenuePercent is numeric, coercing errors to NaN
        green_revenue_df['greenRevenuePercent'] = pd.to_numeric(green_revenue_df['greenRevenuePercent'], errors='coerce')
        green_revenue_df['pure_play_flag'] = green_revenue_df['greenRevenuePercent'].apply(lambda x: 'Y' if pd.notnull(x) and x >= 50 else 'N')
    else:
        st.error("Critical error: 'greenRevenuePercent' not found in the merged dataset. Cannot create 'pure_play_flag'.")
        # Add a default 'N' flag or handle as appropriate
        green_revenue_df['pure_play_flag'] = 'N' 

    # Select and rename columns to match the GREEN_REVENUE specification
    # PCAF columns: cob_date, productype, legal_entity, counterparty_id, counterparty_name, parent_id, group_id, group_name, bic_code, country_code
    # LLM columns: year, totalRevenue, greenRevenuePercent, justification, dataSources
    
    # Ensure all required columns exist before selection to prevent KeyErrors
    required_pcaf_cols = ['cob_date', 'productype', 'legal_entity', 'counterparty_id', 
                            'counterparty_name', 'parent_id', 'group_id', 'group_name', 
                            'bic_code', 'country_code']
    required_llm_cols = ['year', 'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources']
    
    final_green_revenue_cols = required_pcaf_cols + required_llm_cols + ['pure_play_flag']
    
    # Check for missing columns in the merged dataframe
    missing_final_cols = [col for col in final_green_revenue_cols if col not in green_revenue_df.columns]
    if missing_final_cols:
        st.warning(f"GREEN_REVENUE dataset: The following expected columns are missing after merge: {missing_final_cols}. They will not be in the final dataset.")
        # Filter final_green_revenue_cols to only those present
        final_green_revenue_cols = [col for col in final_green_revenue_cols if col in green_revenue_df.columns]

    if not final_green_revenue_cols:
        st.error("GREEN_REVENUE dataset: No columns available for the final dataset after checking for existence. Returning an empty DataFrame.")
        return pd.DataFrame()

    green_revenue_df = green_revenue_df[final_green_revenue_cols]
    
    # Clean up join key columns if they are not needed in the final output
    # green_revenue_df = green_revenue_df.drop(columns=['join_key_pcaf', 'join_key_llm'], errors='ignore')
    # The requirements list specific columns for GREEN_REVENUE, join keys are not among them.

    return green_revenue_df

# --- Helper for SFF Data Key Preprocessing (for comparisons later) ---
def preprocess_sff_keys(sff_df):
    """Preprocesses join/comparison keys for SFF_DATA."""
    sff_df_processed = sff_df.copy()
    # Key Mappings:
    # SFF_DATA."Client Name" corresponds to GREEN_REVENUE.counterparty_name.
    # SFF_DATA.SDS corresponds to GREEN_REVENUE.counterparty_id.
    # SFF_DATA.BIC corresponds to GREEN_REVENUE.bic_code.

    if 'Client Name' in sff_df_processed.columns:
        sff_df_processed['join_key_sff_name'] = preprocess_join_column(sff_df_processed['Client Name'])
    else:
        st.warning("'Client Name' not found in SFF_DATA. Comparison on name might fail.")

    if 'SDS' in sff_df_processed.columns:
        sff_df_processed['join_key_sff_sds'] = preprocess_join_column(sff_df_processed['SDS'])
    else:
        st.warning("'SDS' not found in SFF_DATA. Comparison on SDS/counterparty_id might fail.")

    if 'BIC' in sff_df_processed.columns:
        sff_df_processed['join_key_sff_bic'] = preprocess_join_column(sff_df_processed['BIC'])
    else:
        st.warning("'BIC' not found in SFF_DATA. Comparison on BIC might fail.")
        
    return sff_df_processed

if __name__ == '__main__':
    # This part is for testing the module independently
    # Create dummy data files in a 'data' subdirectory for testing
    # e.g., data/dummy_pcaf.xlsx, data/dummy_llm.csv, data/dummy_sff.xlsx
    
    st.info("Testing data_loader.py module...")
    try:
        # Make sure to create these dummy files in a ./data/ subfolder relative to this script if running directly
        # For example:
        # pd.DataFrame({'counterparty_name': ['Company A', 'Company B'], 'cob_date': [None, None], ...}).to_excel('data/dummy_pcaf.xlsx', index=False)
        # pd.DataFrame({'companyName': ['Company A', 'Company C'], 'greenRevenuePercent': [60, 40]}).to_csv('data/dummy_llm.csv', index=False)
        # pd.DataFrame({'Client Name': ['Company A', 'Company D']}).to_excel('data/dummy_sff.xlsx', index=False)

        pcaf, llm, sff = load_all_data(
            "data/group_client_coverage_dec24.xlsx", 
            "data/llm_generated.csv", 
            "data/Mar PP list_vF.xlsx"
        )
        st.write("PCAF Data:", pcaf.head() if pcaf is not None else "Not loaded")
        st.write("LLM Data:", llm.head() if llm is not None else "Not loaded")
        st.write("SFF Data:", sff.head() if sff is not None else "Not loaded")

        if pcaf is not None and llm is not None:
            pcaf_p, llm_p = preprocess_data(pcaf, llm)
            st.write("Processed PCAF Data (with join_key_pcaf):", pcaf_p.head())
            st.write("Processed LLM Data (with join_key_llm):", llm_p.head())
            
            green_revenue = create_green_revenue_dataset(pcaf_p, llm_p)
            st.write("GREEN_REVENUE Dataset:", green_revenue.head())
            st.write("GREEN_REVENUE Info:", green_revenue.info())

        if sff is not None:
            sff_p = preprocess_sff_keys(sff)
            st.write("Processed SFF Data (with join keys):", sff_p.head())
            
    except Exception as e:
        st.error(f"Error during module test: {e}")

