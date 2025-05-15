from typing import Tuple, Dict, List
import pandas as pd
import numpy as np

def clean_join_column(col: pd.Series) -> pd.Series:
    """Clean join columns by converting to string, stripping whitespace, and lowercasing."""
    return col.astype(str).str.strip().str.lower()

def compare_datasets(
    green_revenue: pd.DataFrame, 
    sff_data: pd.DataFrame,
    pure_play_filter: str = 'Y'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare GREEN_REVENUE and SFF_DATA datasets based on pure play flag.
    
    Args:
        green_revenue: GREEN_REVENUE dataset
        sff_data: SFF dataset
        pure_play_filter: 'Y' or 'N' to filter pure play companies
        
    Returns:
        Tuple containing:
        - overlap: Companies in both datasets
        - in_sff_only: Companies only in SFF_DATA
        - in_green_only: Companies only in GREEN_REVENUE
    """
    # Filter green_revenue based on pure_play_flag
    filtered_green = green_revenue[green_revenue['pure_play_flag'] == pure_play_filter].copy()
    
    # Clean join columns
    filtered_green.loc[:, 'join_key'] = clean_join_column(filtered_green['counterparty_id'])
    sff_data = sff_data.copy()
    sff_data.loc[:, 'join_key'] = clean_join_column(sff_data['sds'])
    
    # Find overlaps
    overlap = pd.merge(
        filtered_green,
        sff_data,
        left_on='join_key',
        right_on='join_key',
        how='inner'
    )
    
    # Find in SFF only
    in_sff_mask = ~sff_data['join_key'].isin(filtered_green['join_key'])
    in_sff_only = sff_data[in_sff_mask].copy()
    
    # Find in Green only
    in_green_mask = ~filtered_green['join_key'].isin(sff_data['join_key'])
    in_green_only = filtered_green[in_green_mask].copy()
    
    return overlap, in_sff_only, in_green_only

def calculate_venn_stats(
    green_revenue: pd.DataFrame, 
    sff_data: pd.DataFrame
) -> Dict[str, int]:
    """
    Calculate statistics for Venn diagram.
    
    Args:
        green_revenue: GREEN_REVENUE dataset
        sff_data: SFF dataset
        
    Returns:
        Dictionary with counts for each Venn diagram section containing:
        - pure_play_count: Number of pure play companies (≥50%)
        - sff_data_count: Number of companies in SFF data
        - pure_play_in_sff: Pure play companies also in SFF
        - pure_play_not_in_sff: Pure play companies not in SFF
        - sff_not_in_pure_play: SFF companies not pure play
        - not_pure_but_30_count: Companies with 30-49% green revenue
        - not_pure_but_30_in_sff: Companies with 30-49% in SFF
    """
    # Clean join columns
    green_revenue = green_revenue.copy()
    green_revenue.loc[:, 'join_key'] = clean_join_column(green_revenue['counterparty_id'])
    sff_data = sff_data.copy()
    sff_data.loc[:, 'join_key'] = clean_join_column(sff_data['sds'])
    
    # Pure play companies (≥50%)
    pure_play = green_revenue[green_revenue['pure_play_flag'] == 'Y']
    pure_play_ids = set(pure_play['join_key'])
    
    # Not pure play but ≥30%
    not_pure_but_30 = green_revenue[
        (green_revenue['pure_play_flag'] == 'N') & 
        (green_revenue['greenRevenuePercent'] >= 30)
    ]
    not_pure_but_30_ids = set(not_pure_but_30['join_key'])
    
    # SFF data IDs
    sff_ids = set(sff_data['join_key'])
    
    # Calculate counts
    stats = {
        'pure_play_count': len(pure_play),
        'sff_data_count': len(sff_data),
        'pure_play_in_sff': len(pure_play_ids & sff_ids),
        'pure_play_not_in_sff': len(pure_play_ids - sff_ids),
        'sff_not_in_pure_play': len(sff_ids - pure_play_ids),
        'not_pure_but_30_count': len(not_pure_but_30),
        'not_pure_but_30_in_sff': len(not_pure_but_30_ids & sff_ids)
    }
    
    return stats

def get_industry_analysis(
    green_revenue: pd.DataFrame,
    sff_data: pd.DataFrame,
    naics_mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Analyze green revenue by industry sector.
    
    Args:
        green_revenue: GREEN_REVENUE dataset
        sff_data: SFF dataset
        naics_mapping: Dictionary mapping NAICS codes to industry names
        
    Returns:
        DataFrame with industry analysis
    """
    # Create a copy to avoid SettingWithCopyWarning
    analysis_df = green_revenue.copy()
    
    # Map NAICS codes to industry sectors
    analysis_df['industry_sector'] = analysis_df['naics_code'].astype(str).str[:2].map(naics_mapping)
    
    # Group by industry and calculate metrics
    industry_stats = analysis_df.groupby('industry_sector').agg(
        company_count=('counterparty_id', 'nunique'),
        avg_green_revenue=('greenRevenuePercent', 'mean'),
        pure_play_count=('pure_play_flag', lambda x: (x == 'Y').sum())
    ).reset_index()
    
    # Calculate pure play percentage
    industry_stats['pure_play_pct'] = (industry_stats['pure_play_count'] / 
                                      industry_stats['company_count'] * 100)
    
    return industry_stats.sort_values('avg_green_revenue', ascending=False)
