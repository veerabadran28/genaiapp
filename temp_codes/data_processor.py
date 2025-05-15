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
        try:
            # Load PCAF data
            self.pcaf_data = pd.read_excel(self.pcaf_file)
            self.pcaf_data_count = len(self.pcaf_data)
            
            # Load LLM generated data
            self.llm_data = pd.read_csv(self.llm_file)
            self.llm_data_count = len(self.llm_data)
            
            # Load SFF data
            self.sff_data = pd.read_excel(self.sff_file)
            self.sff_data_count = len(self.sff_data)
            
            print(f"Data loaded successfully: PCAF ({self.pcaf_data_count}), LLM ({self.llm_data_count}), SFF ({self.sff_data_count})")
        except Exception as e:
            print(f"Error loading data: {e}")
            # Initialize with empty dataframes
            self.pcaf_data = pd.DataFrame()
            self.llm_data = pd.DataFrame() 
            self.sff_data = pd.DataFrame()
    
    def preprocess_data(self):
        """
        Preprocess the datasets - cleanup and standardize
        """
        try:
            # Handle missing columns and ensure they exist
            required_pcaf_columns = ['cob_date', 'productype', 'legal_entity', 'counterparty_id',
                'counterparty_name', 'parent_id', 'group_id', 'group_name',
                'bic_code', 'country_code']
            
            # Add missing columns with default values if needed
            for col in required_pcaf_columns:
                if col not in self.pcaf_data.columns:
                    self.pcaf_data[col] = np.nan
            
            # Preprocess PCAF data - select unique records
            self.pcaf_data = self.pcaf_data.drop_duplicates(subset=[
                'cob_date', 'productype', 'legal_entity', 'counterparty_id',
                'counterparty_name', 'parent_id', 'group_id', 'group_name',
                'bic_code', 'country_code'
            ])
            
            # Make sure the necessary columns exist in each dataframe
            if 'counterparty_name' not in self.pcaf_data.columns:
                self.pcaf_data['counterparty_name'] = ''
                
            if 'companyName' not in self.llm_data.columns:
                self.llm_data['companyName'] = ''
                
            if 'Client Name' not in self.sff_data.columns:
                self.sff_data['Client Name'] = ''
            
            # Apply string cleaning for join fields
            self.pcaf_data['counterparty_name_clean'] = self.pcaf_data['counterparty_name'].astype(str).str.strip().str.lower()
            self.llm_data['companyName_clean'] = self.llm_data['companyName'].astype(str).str.strip().str.lower()
            self.sff_data['Client_Name_clean'] = self.sff_data['Client Name'].astype(str).str.strip().str.lower()
            
            # For the updated join requirement: clean counterparty_id and SDS
            if 'counterparty_id' in self.pcaf_data.columns:
                self.pcaf_data['counterparty_id_clean'] = self.pcaf_data['counterparty_id'].astype(str).str.strip().str.lower()
            else:
                self.pcaf_data['counterparty_id_clean'] = ''
                
            if 'SDS' in self.sff_data.columns:
                self.sff_data['SDS_clean'] = self.sff_data['SDS'].astype(str).str.strip().str.lower()
            else:
                self.sff_data['SDS_clean'] = ''
            
            # Ensure green_revenue is initialized
            self.green_revenue = pd.DataFrame()
            
            # Convert numeric fields if needed
            if 'greenRevenuePercent' in self.llm_data.columns:
                self.llm_data['greenRevenuePercent'] = pd.to_numeric(self.llm_data['greenRevenuePercent'], errors='coerce')
                
            print("Data preprocessing completed")
        except Exception as e:
            print(f"Error preprocessing data: {e}")
    
    def create_green_revenue(self):
        """
        Create the GREEN_REVENUE dataset by joining PCAF_DATA and LLM_GENERATED
        """
        try:
            # Make sure the required columns exist
            if 'counterparty_name_clean' not in self.pcaf_data.columns:
                self.pcaf_data['counterparty_name_clean'] = self.pcaf_data['counterparty_name'].astype(str).str.strip().str.lower()
                
            if 'companyName_clean' not in self.llm_data.columns:
                self.llm_data['companyName_clean'] = self.llm_data['companyName'].astype(str).str.strip().str.lower()
            
            # Join PCAF data with LLM data
            self.green_revenue = pd.merge(
                self.pcaf_data,
                self.llm_data,
                left_on='counterparty_name_clean',
                right_on='companyName_clean',
                how='inner'
            )
            
            print(f"GREEN_REVENUE created with {len(self.green_revenue)} records")
            
            # If no matches found, create an empty DataFrame with the required columns
            if len(self.green_revenue) == 0:
                self.green_revenue = pd.DataFrame(columns=[
                    'cob_date', 'productype', 'legal_entity', 'counterparty_id',
                    'counterparty_name', 'parent_id', 'group_id', 'group_name',
                    'bic_code', 'country_code', 'year', 'totalRevenue',
                    'greenRevenuePercent', 'justification', 'dataSources',
                    'counterparty_name_clean', 'companyName_clean', 'counterparty_id_clean'
                ])
                print("Warning: No matches found when joining PCAF and LLM data")
            else:
                # Select required columns
                try:
                    # Keep only necessary columns
                    required_columns = [
                        'cob_date', 'productype', 'legal_entity', 'counterparty_id',
                        'counterparty_name', 'parent_id', 'group_id', 'group_name',
                        'bic_code', 'country_code', 'year', 'totalRevenue',
                        'greenRevenuePercent', 'justification', 'dataSources'
                    ]
                    
                    # Create clean version of counterparty_id for joining with SDS
                    self.green_revenue['counterparty_id_clean'] = self.green_revenue['counterparty_id'].astype(str).str.strip().str.lower()
                    
                    # Add missing columns with default values if needed
                    for col in required_columns:
                        if col not in self.green_revenue.columns:
                            self.green_revenue[col] = np.nan
                    
                    # Select only required columns plus the clean joining columns
                    self.green_revenue = self.green_revenue[required_columns + ['counterparty_id_clean', 'counterparty_name_clean']]
                    
                except KeyError as e:
                    print(f"Warning: Missing columns in green_revenue: {e}")
                    
                    # Create missing columns with default values
                    for col in [
                        'cob_date', 'productype', 'legal_entity', 'counterparty_id',
                        'counterparty_name', 'parent_id', 'group_id', 'group_name',
                        'bic_code', 'country_code', 'year', 'totalRevenue',
                        'greenRevenuePercent', 'justification', 'dataSources'
                    ]:
                        if col not in self.green_revenue.columns:
                            self.green_revenue[col] = np.nan
            
            # Add pure_play_flag
            self.green_revenue['pure_play_flag'] = np.where(
                self.green_revenue['greenRevenuePercent'] >= 50, 'Y', 'N'
            )
            
            # Ensure counterparty_id_clean column exists for later use
            if 'counterparty_id_clean' not in self.green_revenue.columns:
                self.green_revenue['counterparty_id_clean'] = self.green_revenue['counterparty_id'].astype(str).str.strip().str.lower()
            
            # Update counts
            self.green_revenue_count = len(self.green_revenue)
            self.pure_play_count = len(self.green_revenue[self.green_revenue['pure_play_flag'] == 'Y'])
            self.non_pure_play_count = len(self.green_revenue[self.green_revenue['pure_play_flag'] == 'N'])
            
            print(f"GREEN_REVENUE: Total={self.green_revenue_count}, Pure Play={self.pure_play_count}, Non-Pure Play={self.non_pure_play_count}")
        
        except Exception as e:
            print(f"Error creating GREEN_REVENUE: {e}")
            # Initialize with empty dataframe and counts
            self.green_revenue = pd.DataFrame()
            self.green_revenue_count = 0
            self.pure_play_count = 0
            self.non_pure_play_count = 0
    
    def create_comparison_sets(self):
        """
        Create comparison sets for analysis using SDS and counterparty_id as the joining fields
        """
        try:
            # Ensure we have data to work with
            if len(self.green_revenue) == 0:
                print("Warning: GREEN_REVENUE dataset is empty, creating empty comparison sets")
                self.pure_play_overlap = pd.DataFrame()
                self.pure_play_identified = pd.DataFrame()
                self.pure_play_unidentified = pd.DataFrame()
                self.non_pure_play_overlap = pd.DataFrame()
                self.non_pure_play_identified = pd.DataFrame()
                self.non_pure_play_unidentified = pd.DataFrame()
                return
                
            # Ensure the pure_play_flag column exists
            if 'pure_play_flag' not in self.green_revenue.columns:
                self.green_revenue['pure_play_flag'] = np.where(
                    self.green_revenue['greenRevenuePercent'] >= 50, 'Y', 'N'
                )
            
            # Filter pure play and non-pure play companies
            pure_play = self.green_revenue[self.green_revenue['pure_play_flag'] == 'Y']
            non_pure_play = self.green_revenue[self.green_revenue['pure_play_flag'] == 'N']
            
            # Ensure both dataframes have the necessary columns for joining
            # Convert counterparty_id and SDS to string, strip whitespace, and convert to lowercase
            if 'counterparty_id' in pure_play.columns:
                pure_play['counterparty_id_clean'] = pure_play['counterparty_id'].astype(str).str.strip().str.lower()
            else:
                pure_play['counterparty_id_clean'] = ''
                
            if 'SDS' in self.sff_data.columns:
                self.sff_data['SDS_clean'] = self.sff_data['SDS'].astype(str).str.strip().str.lower()
            else:
                self.sff_data['SDS_clean'] = ''
            
            print("Starting comparison set creation")
            
            # Initialize with empty dataframes in case merges fail
            self.pure_play_overlap = pd.DataFrame()
            
            # Create sets for pure play analysis - using counterparty_id and SDS as the primary join condition
            try:
                # Match on counterparty_id and SDS per updated requirements
                self.pure_play_overlap = pd.merge(
                    pure_play,
                    self.sff_data,
                    left_on='counterparty_id_clean',
                    right_on='SDS_clean',
                    how='inner'
                )
                
                print(f"Pure play overlap count: {len(self.pure_play_overlap)}")
            except Exception as e:
                print(f"Error in counterparty_id-SDS merge for pure play: {e}")
                self.pure_play_overlap = pd.DataFrame()
            
            # Companies in SFF but not in pure play GREEN_REVENUE - with error handling
            try:
                pure_play_ids = set(pure_play['counterparty_id_clean'].dropna())
                
                # Companies in SFF but not in GREEN_REVENUE
                self.pure_play_identified = self.sff_data[
                    ~self.sff_data['SDS_clean'].isin(pure_play_ids)
                ]
                
                print(f"Pure play identified count: {len(self.pure_play_identified)}")
            except Exception as e:
                print(f"Error creating pure_play_identified: {e}")
                self.pure_play_identified = pd.DataFrame()
            
            # Companies in GREEN_REVENUE but not in SFF - with error handling
            try:
                sff_ids = set(self.sff_data['SDS_clean'].dropna())
                
                self.pure_play_unidentified = pure_play[
                    ~pure_play['counterparty_id_clean'].isin(sff_ids)
                ]
                
                print(f"Pure play unidentified count: {len(self.pure_play_unidentified)}")
            except Exception as e:
                print(f"Error creating pure_play_unidentified: {e}")
                self.pure_play_unidentified = pd.DataFrame()
            
            # For non-pure play companies
            # Ensure non_pure_play has the necessary column for joining
            if 'counterparty_id' in non_pure_play.columns:
                non_pure_play['counterparty_id_clean'] = non_pure_play['counterparty_id'].astype(str).str.strip().str.lower()
            else:
                non_pure_play['counterparty_id_clean'] = ''
            
            # Initialize with empty dataframe in case merge fails
            self.non_pure_play_overlap = pd.DataFrame()
            
            # Create sets for non-pure play analysis - using counterparty_id and SDS as the primary join condition
            try:
                # Match on counterparty_id and SDS per updated requirements
                self.non_pure_play_overlap = pd.merge(
                    non_pure_play,
                    self.sff_data,
                    left_on='counterparty_id_clean',
                    right_on='SDS_clean',
                    how='inner'
                )
                
                print(f"Non-pure play overlap count: {len(self.non_pure_play_overlap)}")
            except Exception as e:
                print(f"Error in counterparty_id-SDS merge for non-pure play: {e}")
                self.non_pure_play_overlap = pd.DataFrame()
            
            # Companies in SFF but not in non-pure play GREEN_REVENUE - with error handling
            try:
                non_pure_play_ids = set(non_pure_play['counterparty_id_clean'].dropna())
                
                # Companies in SFF but not in non-pure play GREEN_REVENUE
                self.non_pure_play_identified = self.sff_data[
                    ~self.sff_data['SDS_clean'].isin(non_pure_play_ids)
                ]
                
                print(f"Non-pure play identified count: {len(self.non_pure_play_identified)}")
            except Exception as e:
                print(f"Error creating non_pure_play_identified: {e}")
                self.non_pure_play_identified = pd.DataFrame()
            
            # Companies in non-pure play GREEN_REVENUE but not in SFF - with error handling
            try:
                sff_ids = set(self.sff_data['SDS_clean'].dropna())
                
                self.non_pure_play_unidentified = non_pure_play[
                    ~non_pure_play['counterparty_id_clean'].isin(sff_ids)
                ]
                
                print(f"Non-pure play unidentified count: {len(self.non_pure_play_unidentified)}")
            except Exception as e:
                print(f"Error creating non_pure_play_unidentified: {e}")
                self.non_pure_play_unidentified = pd.DataFrame()
            
            print("Comparison sets created successfully")
        
        except Exception as e:
            print(f"Error in create_comparison_sets: {e}")
            # Initialize with empty dataframes
            self.pure_play_overlap = pd.DataFrame()
            self.pure_play_identified = pd.DataFrame()
            self.pure_play_unidentified = pd.DataFrame()
            self.non_pure_play_overlap = pd.DataFrame()
            self.non_pure_play_identified = pd.DataFrame()
            self.non_pure_play_unidentified = pd.DataFrame()
    
    def get_dataset_info(self):
        """
        Get dataset information for Venn diagram
        
        Returns:
            dict: Counts for various dataset overlaps
        """
        try:
            # Ensure columns exist
            if 'counterparty_name_clean' not in self.pcaf_data.columns:
                self.pcaf_data['counterparty_name_clean'] = self.pcaf_data['counterparty_name'].astype(str).str.strip().str.lower()
                
            if 'companyName_clean' not in self.llm_data.columns:
                self.llm_data['companyName_clean'] = self.llm_data['companyName'].astype(str).str.strip().str.lower()
                
            if 'Client_Name_clean' not in self.sff_data.columns:
                self.sff_data['Client_Name_clean'] = self.sff_data['Client Name'].astype(str).str.strip().str.lower()
            
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
        except Exception as e:
            print(f"Error in get_dataset_info: {e}")
            # Return default values
            return {
                'pcaf_only': 0,
                'llm_only': 0,
                'sff_only': 0,
                'pcaf_llm': 0,
                'pcaf_sff': 0,
                'llm_sff': 0,
                'all_three': 0,
                'pcaf_total': 0,
                'llm_total': 0,
                'sff_total': 0
            }
    
    def process_data(self):
        """
        Process all data in sequence
        """
        self.load_data()
        self.preprocess_data()
        self.create_green_revenue()
        self.create_comparison_sets()
