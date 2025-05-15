from typing import Tuple, Dict
import pandas as pd

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
        Tuple of (overlap, in_sff_only, in_green_only) DataFrames
    """
    # Filter green_revenue based on pure_play_flag
    filtered_green = green_revenue[green_revenue['pure_play_flag'] == pure_play_filter]
    
    # Clean join columns
    filtered_green['join_key'] = filtered_green['counterparty_id'].astype(str).str.strip().str.lower()
    sff_data['join_key'] = sff_data['sds'].astype(str).str.strip().str.lower()
    
    # Find overlaps
    overlap = pd.merge(
        filtered_green,
        sff_data,
        left_on='join_key',
        right_on='join_key',
        how='inner'
    )
    
    # Find in SFF only
    in_sff_only = sff_data[~sff_data['join_key'].isin(filtered_green['join_key'])]
    
    # Find in Green only
    in_green_only = filtered_green[~filtered_green['join_key'].isin(sff_data['join_key'])]
    
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
        Dictionary with counts for each Venn diagram section
    """
    # Clean join columns
    green_revenue['join_key'] = clean_join_column(green_revenue['counterparty_id'])
    sff_data['join_key'] = clean_join_column(sff_data['sds'])
    
    # Pure play companies (>=50%)
    pure_play = green_revenue[green_revenue['pure_play_flag'] == 'Y']
    
    # Not pure play but >=30%
    not_pure_but_30 = green_revenue[
        (green_revenue['pure_play_flag'] == 'N') & 
        (green_revenue['greenRevenuePercent'] >= 30)
    ]
    
    # Calculate counts
    stats = {
        'pure_play_count': len(pure_play),
        'sff_data_count': len(sff_data),
        'pure_play_in_sff': len(set(pure_play['join_key']) & set(sff_data['join_key'])),
        'pure_play_not_in_sff': len(set(pure_play['join_key']) - len(set(pure_play['join_key']) & set(sff_data['join_key'])),
        'sff_not_in_pure_play': len(set(sff_data['join_key']) - set(pure_play['join_key'])),
        'not_pure_but_30_count': len(not_pure_but_30),
        'not_pure_but_30_in_sff': len(set(not_pure_but_30['join_key']) & set(sff_data['join_key']))
    }
    
    return stats
