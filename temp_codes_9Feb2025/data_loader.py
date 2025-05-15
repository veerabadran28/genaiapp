import pandas as pd
import numpy as np
from typing import Tuple

def clean_join_column(col: pd.Series) -> pd.Series:
    """Clean join columns by converting to string, stripping whitespace, and lowercasing."""
    return col.astype(str).str.strip().str.lower()

def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess all datasets with proper join column cleaning.
    
    Returns:
        Tuple containing (pcaf_data, llm_data, sff_data)
    """
    # Load PCAF data
    pcaf_data = pd.read_excel("data/group_client_coverage_dec24.xlsx")
    pcaf_data = pcaf_data[[
        'cob_date', 'productype', 'legal_entity', 'counterparty_id', 
        'counterparty_name', 'parent_id', 'group_id', 'group_name', 
        'bic_code', 'country_code'
    ]].drop_duplicates()
    
    # Load LLM generated data
    llm_data = pd.read_csv("data/llm_generated.csv")
    
    # Load SFF data
    sff_data = pd.read_excel("data/Mar PP list_vF.xlsx")
    sff_data = sff_data.rename(columns={
        'Client Name': 'client_name',
        'SDS': 'sds',
        'BIC': 'bic'
    })
    
    return pcaf_data, llm_data, sff_data

def create_green_revenue_dataset(pcaf_data: pd.DataFrame, llm_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create the GREEN_REVENUE dataset by joining PCAF and LLM data.
    
    Args:
        pcaf_data: Processed PCAF dataset
        llm_data: Processed LLM dataset
        
    Returns:
        Merged GREEN_REVENUE dataset with pure_play_flag
    """
    # Clean join columns
    pcaf_data['join_key'] = clean_join_column(pcaf_data['counterparty_name'])
    llm_data['join_key'] = clean_join_column(llm_data['companyName'])
    
    # Perform the merge
    green_revenue = pd.merge(
        pcaf_data,
        llm_data,
        left_on='join_key',
        right_on='join_key',
        how='inner'
    ).drop(columns=['join_key', 'companyName'])
    
    # Add pure_play_flag
    green_revenue['pure_play_flag'] = np.where(
        green_revenue['greenRevenuePercent'] >= 50, 'Y', 'N'
    )
    
    return green_revenue
