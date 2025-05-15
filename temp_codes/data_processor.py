# utils/data_processor.py
import pandas as pd
import numpy as np
import os
import warnings

class DataProcessor:
    """
    Class responsible for processing and joining the datasets
    """
    
    def __init__(self, pcaf_file, llm_file, sff_file):
        """
        Initialize the DataProcessor with file paths
        
        Args:
            pcaf_file (str): Path to PCAF dataset
            llm_file (str): Path to LLM generated dataset
            sff_file (str): Path to SFF dataset
        """
        self.pcaf_file = pcaf_file
        self.llm_file = llm_file
        self.sff_file = sff_file
        
        # Initialize dataframes
        self.pcaf_data = None
        self.llm_data = None
        self.sff_data = None
        self.green_revenue = None
        
        # Initialize record counts
        self.pcaf_data_count = 0
        self.llm_data_count = 0
        self.sff_data_count = 0
        self.green_revenue_count = 0
        self.pure_play_count = 0
        self.non_pure_play_count = 0
        
        # Initialize overlaps and sets
        self.pure_play_overlap = None
        self.pure_play_identified = None
        self.pure_play_unidentified = None
        self.non_pure_play_overlap = None
        self.non_pure_play_identified = None
        self.non_pure_play_unidentified = None
        
        # Suppress warnings
        warnings.filterwarnings('ignore')
    
    def load_data(self):
        """
        Load the datasets from files
        """
        # Load PCAF data
        self.pcaf_data = pd.read_excel(self.pcaf_file)
        self.pcaf_data_count = len(self.pcaf_data)
        
        # Load LLM generated data
        self.llm_data = pd.read_csv(self.llm_file)
        self.llm_data_count = len(self.llm_data)
        
        # Load SFF data
        self.sff_data = pd.read_excel(self.sff_file)
        self.sff_data_count = len(self.sff_data)
    
    def preprocess_data(self):
        """
        Preprocess the datasets - cleanup and standardize
        """
        # Preprocess PCAF data - select unique records
        self.pcaf_data = self.pcaf_data.drop_duplicates(subset=[
            'cob_date', 'productype', 'legal_entity', 'counterparty_id',
            'counterparty_name', 'parent_id', 'group_id', 'group_name',
            'bic_code', 'country_code'
        ])
        
        # Apply string cleaning for join fields
        self.pcaf_data['counterparty_name_clean'] = self.pcaf_data['counterparty_name'].astype(str).str.strip().str.lower()
        self.llm_data['companyName_clean'] = self.llm_data['companyName'].astype(str).str.strip().str.lower()
        self.sff_data['Client_Name_clean'] = self.sff_data['Client Name'].astype(str).str.strip().str.lower()
        
        # Convert numeric fields if needed
        if 'greenRevenuePercent' in self.llm_data.columns:
            self.llm_data['greenRevenuePercent'] = pd.to_numeric(self.llm_data['greenRevenuePercent'], errors='coerce')
    
    def create_green_revenue(self):
        """
        Create the GREEN_REVENUE dataset by joining PCAF_DATA and LLM_GENERATED
        """
        # Join PCAF data with LLM data
        self.green_revenue = pd.merge(
            self.pcaf_data,
            self.llm_data,
            left_on='counterparty_name_clean',
            right_on='companyName_clean',
            how='inner'
        )
        
        # Select required columns
        self.green_revenue = self.green_revenue[[
            'cob_date', 'productype', 'legal_entity', 'counterparty_id',
            'counterparty_name', 'parent_id', 'group_id', 'group_name',
            'bic_code', 'country_code', 'year', 'totalRevenue',
            'greenRevenuePercent', 'justification', 'dataSources'
        ]]
        
        # Add pure_play_flag
        self.green_revenue['pure_play_flag'] = np.where(
            self.green_revenue['greenRevenuePercent'] >= 50, 'Y', 'N'
        )
        
        # Update counts
        self.green_revenue_count = len(self.green_revenue)
        self.pure_play_count = len(self.green_revenue[self.green_revenue['pure_play_flag'] == 'Y'])
        self.non_pure_play_count = len(self.green_revenue[self.green_revenue['pure_play_flag'] == 'N'])
    
    def create_comparison_sets(self):
        """
        Create comparison sets for analysis
        """
        # Filter pure play and non-pure play companies
        pure_play = self.green_revenue[self.green_revenue['pure_play_flag'] == 'Y']
        non_pure_play = self.green_revenue[self.green_revenue['pure_play_flag'] == 'N']
        
        # Create sets for pure play analysis
        # Match on different fields per requirements
        pure_play_overlap_by_name = pd.merge(
            pure_play,
            self.sff_data,
            left_on='counterparty_name_clean',
            right_on='Client_Name_clean',
            how='inner'
        )
        
        pure_play_overlap_by_id = pd.merge(
            pure_play,
            self.sff_data,
            left_on='counterparty_id',
            right_on='SDS',
            how='inner'
        )
        
        pure_play_overlap_by_bic = pd.merge(
            pure_play,
            self.sff_data,
            left_on='bic_code',
            right_on='BIC',
            how='inner'
        )
        
        # Combine overlaps (removing duplicates)
        self.pure_play_overlap = pd.concat([
            pure_play_overlap_by_name,
            pure_play_overlap_by_id,
            pure_play_overlap_by_bic
        ]).drop_duplicates(subset=[
            'counterparty_id', 'counterparty_name'
        ])
        
        # Companies in SFF but not in pure play GREEN_REVENUE
        pure_play_ids = set(pure_play['counterparty_id'].dropna())
        pure_play_names = set(pure_play['counterparty_name_clean'].dropna())
        pure_play_bics = set(pure_play['bic_code'].dropna())
        
        # Companies in SFF but not in GREEN_REVENUE
        self.pure_play_identified = self.sff_data[
            (~self.sff_data['SDS'].isin(pure_play_ids)) &
            (~self.sff_data['Client_Name_clean'].isin(pure_play_names)) &
            (~self.sff_data['BIC'].isin(pure_play_bics))
        ]
        
        # Companies in GREEN_REVENUE but not in SFF
        sff_ids = set(self.sff_data['SDS'].dropna())
        sff_names = set(self.sff_data['Client_Name_clean'].dropna())
        sff_bics = set(self.sff_data['BIC'].dropna())
        
        self.pure_play_unidentified = pure_play[
            (~pure_play['counterparty_id'].isin(sff_ids)) &
            (~pure_play['counterparty_name_clean'].isin(sff_names)) &
            (~pure_play['bic_code'].isin(sff_bics))
        ]
        
        # Create sets for non-pure play analysis
        # Match on different fields per requirements
        non_pure_play_overlap_by_name = pd.merge(
            non_pure_play,
            self.sff_data,
            left_on='counterparty_name_clean',
            right_on='Client_Name_clean',
            how='inner'
        )
        
        non_pure_play_overlap_by_id = pd.merge(
            non_pure_play,
            self.sff_data,
            left_on='counterparty_id',
            right_on='SDS',
            how='inner'
        )
        
        non_pure_play_overlap_by_bic = pd.merge(
            non_pure_play,
            self.sff_data,
            left_on='bic_code',
            right_on='BIC',
            how='inner'
        )
        
        # Combine overlaps (removing duplicates)
        self.non_pure_play_overlap = pd.concat([
            non_pure_play_overlap_by_name,
            non_pure_play_overlap_by_id,
            non_pure_play_overlap_by_bic
        ]).drop_duplicates(subset=[
            'counterparty_id', 'counterparty_name'
        ])
        
        # Companies in SFF but not in non-pure play GREEN_REVENUE
        non_pure_play_ids = set(non_pure_play['counterparty_id'].dropna())
        non_pure_play_names = set(non_pure_play['counterparty_name_clean'].dropna())
        non_pure_play_bics = set(non_pure_play['bic_code'].dropna())
        
        # Companies in SFF but not in non-pure play GREEN_REVENUE
        self.non_pure_play_identified = self.sff_data[
            (~self.sff_data['SDS'].isin(non_pure_play_ids)) &
            (~self.sff_data['Client_Name_clean'].isin(non_pure_play_names)) &
            (~self.sff_data['BIC'].isin(non_pure_play_bics))
        ]
        
        # Companies in non-pure play GREEN_REVENUE but not in SFF
        self.non_pure_play_unidentified = non_pure_play[
            (~non_pure_play['counterparty_id'].isin(sff_ids)) &
            (~non_pure_play['counterparty_name_clean'].isin(sff_names)) &
            (~non_pure_play['bic_code'].isin(sff_bics))
        ]
    
    def get_dataset_info(self):
        """
        Get dataset information for Venn diagram
        
        Returns:
            dict: Counts for various dataset overlaps
        """
        # Count the unique companies in each dataset
        pcaf_companies = set(self.pcaf_data['counterparty_name_clean'].dropna())
        llm_companies = set(self.llm_data['companyName_clean'].dropna())
        sff_companies = set(self.sff_data['Client_Name_clean'].dropna())
        
        # Calculate overlaps
        pcaf_only = len(pcaf_companies - (llm_companies | sff_companies))
        llm_only = len(llm_companies - (pcaf_companies | sff_companies))
        sff_only = len(sff_companies - (pcaf_companies | llm_companies))
        
        pcaf_llm = len((pcaf_companies & llm_companies) - sff_companies)
        pcaf_sff = len((pcaf_companies & sff_companies) - llm_companies)
        llm_sff = len((llm_companies & sff_companies) - pcaf_companies)
        
        all_three = len(pcaf_companies & llm_companies & sff_companies)
        
        return {
            'pcaf_only': pcaf_only,
            'llm_only': llm_only,
            'sff_only': sff_only,
            'pcaf_llm': pcaf_llm,
            'pcaf_sff': pcaf_sff,
            'llm_sff': llm_sff,
            'all_three': all_three,
            'pcaf_total': len(pcaf_companies),
            'llm_total': len(llm_companies),
            'sff_total': len(sff_companies)
        }
    
    def process_data(self):
        """
        Process all data in sequence
        """
        self.load_data()
        self.preprocess_data()
        self.create_green_revenue()
        self.create_comparison_sets()
