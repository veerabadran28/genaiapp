from typing import Tuple, Dict, List
import pandas as pd
import numpy as np

def clean_join_column(col: pd.Series) -> pd.Series:
    """Standardize join columns."""
    return col.astype(str).str.strip().str.lower()

def compare_datasets(
    green_revenue: pd.DataFrame, 
    sff_data: pd.DataFrame,
    pure_play_filter: str = 'Y'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare datasets based on pure play flag.
    
    Returns:
        Tuple of (overlap, sff_only, green_only) DataFrames
    """
    if pure_play_filter not in ['Y', 'N']:
        raise ValueError("pure_play_filter must be 'Y' or 'N'")
    
    filtered_green = green_revenue[green_revenue['pure_play_flag'] == pure_play_filter].copy()
    sff_data = sff_data.copy()
    
    filtered_green.loc[:, 'join_key'] = clean_join_column(filtered_green['counterparty_id'])
    sff_data.loc[:, 'join_key'] = clean_join_column(sff_data['sds'])
    
    overlap = pd.merge(
        filtered_green,
        sff_data,
        on='join_key',
        how='inner'
    )
    
    sff_only = sff_data[~sff_data['join_key'].isin(filtered_green['join_key'])]
    green_only = filtered_green[~filtered_green['join_key'].isin(sff_data['join_key'])]
    
    return overlap, sff_only, green_only

def calculate_venn_stats(
    green_revenue: pd.DataFrame, 
    sff_data: pd.DataFrame
) -> Dict[str, int]:
    """
    Calculate statistics for Venn diagram visualization.
    """
    green_revenue = green_revenue.copy()
    sff_data = sff_data.copy()
    
    green_revenue.loc[:, 'join_key'] = clean_join_column(green_revenue['counterparty_id'])
    sff_data.loc[:, 'join_key'] = clean_join_column(sff_data['sds'])
    
    pure_play = green_revenue[green_revenue['pure_play_flag'] == 'Y']
    pure_play_ids = set(pure_play['join_key'])
    
    not_pure_but_30 = green_revenue[
        (green_revenue['pure_play_flag'] == 'N') & 
        (green_revenue['greenRevenuePercent'] >= 30)
    ]
    not_pure_but_30_ids = set(not_pure_but_30['join_key'])
    
    sff_ids = set(sff_data['join_key'])
    
    stats = {
        'pure_play_count': len(pure_play),
        'sff_data_count': len(sff_data),
        'not_pure_but_30_count': len(not_pure_but_30),
        'pure_play_in_sff': len(pure_play_ids & sff_ids),
        'pure_play_not_in_sff': len(pure_play_ids - sff_ids),
        'sff_not_in_pure_play': len(sff_ids - pure_play_ids),
        'not_pure_but_30_in_sff': len(not_pure_but_30_ids & sff_ids)
    }
    
    return stats

def get_industry_analysis(green_revenue: pd.DataFrame) -> pd.DataFrame:
    """Analyze green revenue by industry sector."""
    naics_mapping = {
        '11': 'Agriculture', '21': 'Mining', '22': 'Utilities',
        '23': 'Construction', '31-33': 'Manufacturing', '42': 'Wholesale',
        '44-45': 'Retail', '48-49': 'Transport', '51': 'Information',
        '52': 'Finance', '53': 'Real Estate', '54': 'Professional',
        '56': 'Admin', '61': 'Education', '62': 'Healthcare',
        '71': 'Arts', '72': 'Hospitality', '81': 'Services'
    }
    
    analysis_df = green_revenue.copy()
    analysis_df['naics_prefix'] = analysis_df['naics_code'].astype(str).str[:2]
    analysis_df['industry_sector'] = analysis_df['naics_prefix'].map(naics_mapping)
    
    industry_stats = analysis_df.groupby('industry_sector').agg(
        company_count=('counterparty_id', 'nunique'),
        avg_green_revenue=('greenRevenuePercent', 'mean'),
        pure_play_count=('pure_play_flag', lambda x: (x == 'Y').sum())
    ).reset_index()
    
    industry_stats['pure_play_pct'] = (
        industry_stats['pure_play_count'] / industry_stats['company_count'] * 100
    )
    
    return industry_stats.sort_values('avg_green_revenue', ascending=False)
