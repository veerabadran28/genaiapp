import streamlit as st
from datetime import datetime
import os
import json
import sys
from typing import Dict, Optional, List  # Add this line for type hints

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Update imports to use relative paths
from services.bedrock_service import BedrockService
from src.services.s3_service import S3Service
from ui.summary_analysis import SummaryAnalysisUI
from ui.document_assistant import DocumentAssistantUI
from ui.admin_interface import AdminUI
from src.ui.dashboard import DashboardUI
from ui.report_ui import ReportUI

# UI Configuration
MAIN_UI_CONFIG = """
<style>
div.block-container {padding-top: 1rem !important;}
div.stImage {margin: 0 !important; padding: 0 !important;}
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}

.header {
    background-color: #0051A2;
    padding: 1rem;
    border-radius: 0;
    margin: 3 -4rem 2rem -4rem;
    padding: 1rem 4rem;
}

.search-container {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}

.search-input {
    flex: 1;
    padding: 0.5rem;
    border: none;
    border-radius: 4px;
}

.search-button {
    background: #003D82;
    color: white;
    border: none;
    padding: 0.5rem 2rem;
    border-radius: 4px;
    cursor: pointer;
}

.stButton>button {
    background-color: #0051A2;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    cursor: pointer;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #f8f9fa;
    padding: 2rem 1rem;
}

[data-testid="stSidebar"] .block-container {
    margin-top: 1rem;
}
</style>
"""

def get_drivers_dir() -> str:
    """Get the drivers directory path."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    drivers_dir = os.path.join(base_path, 'drivers')
    
    # Create directory if it doesn't exist
    if not os.path.exists(drivers_dir):
        os.makedirs(drivers_dir)
    
    return drivers_dir

def get_driver_file_path(year: str, period: str) -> str:
    """Get the complete path for the driver file."""
    drivers_dir = get_drivers_dir()
    return os.path.join(drivers_dir, f"{year}_{period}_application_drivers.txt")

def load_driver_values_from_file(file_path: str) -> Optional[Dict]:
    """Load driver values from file if it exists."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        st.error(f"Error loading driver values: {str(e)}")
        return None

def save_driver_values_to_file(values: Dict, file_path: str) -> bool:
    """Save driver values to file by overwriting existing content."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(values, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving driver values: {str(e)}")
        return False

def generate_driver_values(bedrock_service, year: str, period: str, force_refresh: bool = False) -> Optional[Dict]:
    """Generate driver values using the bedrock service."""
    try:
        # Get file path
        file_path = get_driver_file_path(year, period)
        
        # Check for existing file unless force refresh is requested
        if not force_refresh:
            existing_values = load_driver_values_from_file(file_path)
            if existing_values:
                return existing_values

        driver_prompt = f"""
        Generate financial period information for the year {year}, period {period}, with these rules:
        1. Banks publish quarterly results with a delay (current quarter results available in next quarter)
        2. Half year results follow the same pattern (H1 results fully available in Q3)
        3. Due to reporting delays:
        - Current quarter results become available in next quarter
        - H1 results become available in Q3
        4. Use 2-digit year format (e.g., 24 for 2024)
        5. Calculate days based on input year (check if leap year)
        6. Format periods as follows:
        - Half years: H124, H223
        - Full year: FY23
        - Quarters: Q224, Q124
        
        Attribute 1: "Previous Half Year Period". Logic: Previous Half Year Period for which the results are out.
        Attribute 2: "Prior Half Year Period". Logic: Same previous Half Year Period of previous year.
        Attribute 3: "Prior Year End Period". Logic: Prior Year End Period.
        Attribute 4: "Current Quarter End Period". Logic: Most recent quarter end period for which results are available (previous quarter).
        Attribute 5: "Prior Quarter End Period". Logic: Second Most recent quarter end period for which results are available.
        Attribute 6: "Prior Quarter End Period Previous Year". Logic: Same as Prior Quarter End Period but for previous year.
        Attribute 7: "Total Days".
        Attribute 8: "Half Year Period Days".                
        Attribute 9: "Q1 Days".
        Attribute 10: "Q2 Days".
        Attribute 11: "Q3 Days".
        Attribute 12: "Q4 Days".
        Attribute 13: "Previous Year Total Days".
        Attribute 14: "Previous Year Half Year Period Days".
        Attribute 15: "Q1 Days of previous year".
        Attribute 16: "Q2 Days of previous year".
        Attribute 17: "Q3 Days of previous year".
        Attribute 18: "Q4 Days of previous year".
        Attribute 19: "Half Year End Results Period". Logic: First two letters of "Previous Half Year Period". Example: H2.
        Attribute 20: "Quarter End Results Period". Logic: First two letters of "Current Quarter End Period". Example: Q4.
        Attribute 21: "Quarter End Results Period Year". Logic: Last two letters of "Current Quarter End Period" in YYYY format. Example: 2023.
        Attribute 22: "Prior Quarter End Results Period". Logic: First two letters of "Prior Quarter End Period". Example: Q4.
        Attribute 23: "Prior Quarter End Results Period Year". Logic: Last two letters of "Prior Quarter End Period" in YYYY format. Example: 2023.
        Attribute 24: "Prior 1 Quarter". Logic: Previous quarter to "Prior Quarter End Period". Logic: If "Prior Quarter End Period" is Q124, then one quarter prior to that is Q423.
        Attribute 25: "Prior 2 Quarter". Logic: Previous quarter to "Prior 1 Quarter"
        Attribute 26: "Prior 3 Quarter". Logic: Previous quarter to "Prior 2 Quarter"
        Attribute 27: "Prior 4 Quarter". Logic: Previous quarter to "Prior 3 Quarter"
        Attribute 28: "Prior 5 Quarter". Logic: Previous quarter to "Prior 4 Quarter"
        Attribute 29: "Prior 6 Quarter". Logic: Previous quarter to "Prior 5 Quarter"
        
        Output exactly this format with strictly no additional text:
        
        [Previous Half Year Period]|[Prior Half Year Period]|[Prior Year End Period]|[Current Quarter End Period]|[Prior Quarter End Period]|[Prior Quarter End Period Previous Year]|[Total Days]|[Half Year Period Days]|[Q1 Days]|[Q2 Days]|[Q3 Days]|[Q4 Days]|[Previous Year Total Days]|[Previous Year Half Year Period Days]|[Q1 Days of previous year]|[Q2 Days of previous year]|[Q3 Days of previous year]|[Q4 Days of previous year]|[Half Year End Results Period]|[Quarter End Results Period]|[Quarter End Results Period Year]|[Prior Quarter End Results Period]|[Prior Quarter End Results Period Year]|[Prior 1 Quarter]|[Prior 2 Quarter]|[Prior 3 Quarter]|[Prior 4 Quarter]|[Prior 5 Quarter]|[Prior 6 Quarter]
        """
        
        response = bedrock_service.invoke_model_simple(driver_prompt)
        if response:
            values = response.strip().split('|')
            if len(values) == 29:
                result = {
                    "Half Year Period": values[0],
                    "Prior Half Year Period": values[1],
                    "Prior Year End Period": values[2],
                    "Current Quarter End Period": values[3],
                    "Prior Quarter End Period": values[4],
                    "Prior Quarter End Period Previous Year": values[5],
                    "Total Days": values[6],
                    "Half Year Period Days": values[7],                            
                    "Q1 Days": values[8],
                    "Q2 Days": values[9],
                    "Q3 Days": values[10],
                    "Q4 Days": values[11],
                    "Previous Year Total Days": values[12],
                    "Previous Year Half Year Period Days": values[13],
                    "Q1 Days of previous year": values[14],
                    "Q2 Days of previous year": values[15],
                    "Q3 Days of previous year": values[16],
                    "Q4 Days of previous year": values[17],
                    "Half Year End Results Period": values[18],
                    "Quarter End Results Period": values[19],
                    "Quarter End Results Period Year": values[20],
                    "Prior Quarter End Results Period": values[21],
                    "Prior Quarter End Results Period Year": values[22],
                    "Prior 1 Quarter": values[23],
                    "Prior 2 Quarter": values[24],
                    "Prior 3 Quarter": values[25],
                    "Prior 4 Quarter": values[26],
                    "Prior 5 Quarter": values[27],
                    "Prior 6 Quarter": values[28]
                }
                
                # Save by overwriting existing file
                if save_driver_values_to_file(result, file_path):
                    return result
                
        return None
    except Exception as e:
        st.error(f"Error generating driver values: {str(e)}")
        return st.session_state.get('driver_values', None)

def display_app_drivers(common_config: Dict, bedrock_service):
    """Display application drivers with refresh button."""
    with st.expander("Application Drivers"):
        # Create filters and refresh button
        s3_service = S3Service(common_config['s3_config'])
        available_years = s3_service.get_distinct_values("year")
        periods = ['Q1', 'Q2', 'Q3', 'Q4']

        # Find indices for current selections
        year_index = available_years.index(st.session_state.driver_year) if st.session_state.driver_year in available_years else 0
        period_index = periods.index(st.session_state.driver_period) if st.session_state.driver_period in periods else 0

        # Create columns for filters and refresh button
        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            year = st.selectbox(
                "Select Year:",
                options=available_years,
                key="year_filter",
                index=year_index
            )
            if year != st.session_state.driver_year:
                st.session_state.driver_year = year

        with col2:
            period = st.selectbox(
                "Select Period:",
                options=periods,
                key="period_filter",
                index=period_index
            )
            if period != st.session_state.driver_period:
                st.session_state.driver_period = period

        with col3:
            if st.button("🔄 Refresh", key="refresh_drivers"):
                with st.spinner("Refreshing driver values..."):
                    new_values = generate_driver_values(
                        bedrock_service,
                        st.session_state.driver_year,
                        st.session_state.driver_period,
                        force_refresh=True
                    )
                    if new_values:
                        st.session_state.driver_values = new_values
                        st.rerun()

        # Display driver values in form
        if 'driver_values' in st.session_state and st.session_state.driver_values:
            with st.form("application_drivers_form"):
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.text_input(
                        "Half Year Period",
                        value=st.session_state.driver_values["Half Year Period"],
                        disabled=True,
                        key="driver_half_year"
                    )
                with col2:
                    st.text_input(
                        "Prior Half Year Period",
                        value=st.session_state.driver_values["Prior Half Year Period"],
                        disabled=True,
                        key="driver_prior_half_year"
                    )
                with col3:
                    st.text_input(
                        "Prior Year End Period",
                        value=st.session_state.driver_values["Prior Year End Period"],
                        disabled=True,
                        key="driver_prior_year_end"
                    )
                with col4:
                    st.text_input(
                        "Current Quarter End Period",
                        value=st.session_state.driver_values["Current Quarter End Period"],
                        disabled=True,
                        key="driver_current_quarter"
                    )
                with col5:
                    st.text_input(
                        "Prior Quarter End Period",
                        value=st.session_state.driver_values["Prior Quarter End Period"],
                        disabled=True,
                        key="driver_prior_quarter"
                    )
                with col6:
                    st.text_input(
                        "Prior Quarter End Period Previous Year",
                        value=st.session_state.driver_values["Prior Quarter End Period Previous Year"],
                        disabled=True,
                        key="driver_prior_quarter_prev_year"
                    )

                col1, col2, col3, col4, col5, col6 = st.columns(6)                        
                with col1:
                    st.text_input(
                        "Total Days",
                        value=st.session_state.driver_values["Total Days"],
                        disabled=True,
                        key="driver_total_days"
                    )
                with col2:
                    st.text_input(
                        "Half Year Period Days",
                        value=st.session_state.driver_values["Half Year Period Days"],
                        disabled=True,
                        key="driver_half_year_days"
                    )
                with col3:
                    st.text_input(
                        "Q1 Days",
                        value=st.session_state.driver_values["Q1 Days"],
                        disabled=True,
                        key="driver_q1_days"
                    )
                with col4:
                    st.text_input(
                        "Q2 Days",
                        value=st.session_state.driver_values["Q2 Days"],
                        disabled=True,
                        key="driver_q2_days"
                    )                      
                with col5:
                    st.text_input(
                        "Q3 Days",
                        value=st.session_state.driver_values["Q3 Days"],
                        disabled=True,
                        key="driver_q3_days"
                    )
                with col6:
                    st.text_input(
                        "Q4 Days",
                        value=st.session_state.driver_values["Q4 Days"],
                        disabled=True,
                        key="driver_q4_days"
                    )
                
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.text_input(
                        "Previous Year Total Days",
                        value=st.session_state.driver_values["Previous Year Total Days"],
                        disabled=True,
                        key="driver_previous_year_total_days"
                    )                                          
                with col2:
                    st.text_input(
                        "Previous Year Half Year Period Days",
                        value=st.session_state.driver_values["Previous Year Half Year Period Days"],
                        disabled=True,
                        key="driver_prev_half_year_days"
                    )
                with col3:
                    st.text_input(
                        "Q1 Days of previous year",
                        value=st.session_state.driver_values["Q1 Days of previous year"],
                        disabled=True,
                        key="driver_q1_days_prev_year"
                    )
                with col4:
                    st.text_input(
                        "Q2 Days of previous year",
                        value=st.session_state.driver_values["Q2 Days of previous year"],
                        disabled=True,
                        key="driver_q2_days_prev_year"
                    )                      
                with col5:
                    st.text_input(
                        "Q3 Days of previous year",
                        value=st.session_state.driver_values["Q3 Days of previous year"],
                        disabled=True,
                        key="driver_q3_days_prev_year"
                    )
                with col6:
                    st.text_input(
                        "Q4 Days of previous year",
                        value=st.session_state.driver_values["Q4 Days of previous year"],
                        disabled=True,
                        key="driver_q4_days_prev_year"
                    )
                
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.text_input(
                        "Half Year End Results Period",
                        value=st.session_state.driver_values["Half Year End Results Period"],
                        disabled=True,
                        key="driver_half_year_end_results_period"
                    )
                with col2:
                    st.text_input(
                        "Quarter End Results Period",
                        value=st.session_state.driver_values["Quarter End Results Period"],
                        disabled=True,
                        key="driver_quarter_end_results_period"
                    )
                with col3:
                    st.text_input(
                        "Quarter End Results Period Year",
                        value=st.session_state.driver_values["Quarter End Results Period Year"],
                        disabled=True,
                        key="driver_quarter_end_results_period_year"
                    )
                with col4:
                    st.text_input(
                        "Prior Quarter End Results Period",
                        value=st.session_state.driver_values["Prior Quarter End Results Period"],
                        disabled=True,
                        key="driver_prior_quarter_end_results_period"
                    )
                with col5:
                    st.text_input(
                        "Prior Quarter End Results Period Year",
                        value=st.session_state.driver_values["Prior Quarter End Results Period Year"],
                        disabled=True,
                        key="driver_prior_quarter_end_results_period_year"
                    )
                with col6:
                    st.text_input(
                        "Prior 1 Quarter",
                        value=st.session_state.driver_values["Prior 1 Quarter"],
                        disabled=True,
                        key="driver_prior_1_quarter"
                    )
                    
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.text_input(
                        "Prior 2 Quarter",
                        value=st.session_state.driver_values["Prior 2 Quarter"],
                        disabled=True,
                        key="driver_prior_2_quarter"
                    )
                with col2:
                    st.text_input(
                        "Prior 3 Quarter",
                        value=st.session_state.driver_values["Prior 3 Quarter"],
                        disabled=True,
                        key="driver_prior_3_quarter"
                    )                       
                with col3:
                    st.text_input(
                        "Prior 4 Quarter",
                        value=st.session_state.driver_values["Prior 4 Quarter"],
                        disabled=True,
                        key="driver_prior_4_quarter"
                    )
                with col4:
                    st.text_input(
                        "Prior 5 Quarter",
                        value=st.session_state.driver_values["Prior 5 Quarter"],
                        disabled=True,
                        key="driver_prior_5_quarter"
                    )
                with col5:
                    st.text_input(
                        "Prior 6 Quarter",
                        value=st.session_state.driver_values["Prior 6 Quarter"],
                        disabled=True,
                        key="driver_prior_6_quarter"
                    )
                
                # Submit button (disabled since it's read-only)
                st.form_submit_button("", disabled=True)

def load_configs():
    """Load common, bank-specific, and tab-specific configurations."""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_dir = os.path.join(base_dir, 'config')
        
        # Load common config
        common_config_path = os.path.join(config_dir, 'common_config.json')
        if not os.path.exists(common_config_path):
            raise FileNotFoundError("Common config file not found")
            
        with open(common_config_path, 'r', encoding='utf-8') as f:
            common_config = json.load(f)
        
        # Load bank configs
        bank_configs = {}
        banks_dir = os.path.join(config_dir, 'banks')
        
        # Default placeholder config for banks
        placeholder_bank_config = {
            "bank_info": {
                "name": "",
                "logo_url": "",
                "implementation_status": "pending"
            }
        }
        
        # Create placeholder configs for all banks in working_summary_tabs
        working_summary_tabs = common_config['tabs_config']['working_summary_tabs']
        for bank_name in working_summary_tabs:
            bank_id = bank_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            bank_configs[bank_id] = placeholder_bank_config.copy()
            bank_configs[bank_id]['bank_info']['name'] = bank_name
        
        # Load actual bank configs where available
        for file in os.listdir(banks_dir):
            if file.endswith('_config.json'):
                bank_id = file.replace('_config.json', '')
                config_path = os.path.join(banks_dir, file)
                
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            bank_configs[bank_id] = json.loads(content)
                            bank_configs[bank_id]['bank_info']['implementation_status'] = 'active'
                except Exception as e:
                    st.warning(f"Could not load config for {bank_id}: {str(e)}")
                    continue

        # Load tab configs
        tab_configs = {}
        tabs_dir = os.path.join(config_dir, 'tabs')
        
        # Create tabs directory if it doesn't exist
        if not os.path.exists(tabs_dir):
            os.makedirs(tabs_dir)

        # Default placeholder config for tabs
        placeholder_tab_config = {
            "tab_info": {
                "name": "",
                "logo_url": common_config.get('app_config', {}).get('barclays_logo_url', ''),
                "implementation_status": "pending"
            }
        }

        # Create placeholder configs for all tabs in final_summary_tabs
        final_summary_tabs = common_config['tabs_config']['final_summary_tabs']
        for tab_name in final_summary_tabs:
            tab_id = tab_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            tab_configs[tab_id] = placeholder_tab_config.copy()
            tab_configs[tab_id]['tab_info']['name'] = tab_name

        # Load actual tab configs where available
        for file in os.listdir(tabs_dir):
            if file.endswith('_config.json'):
                tab_id = file.replace('_config.json', '')
                config_path = os.path.join(tabs_dir, file)
                
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            tab_configs[tab_id] = json.loads(content)
                            tab_configs[tab_id]['tab_info']['implementation_status'] = 'active'
                except Exception as e:
                    st.warning(f"Could not load config for tab {tab_id}: {str(e)}")
                    continue
                    
        return common_config, bank_configs, tab_configs
        
    except Exception as e:
        st.error(f"Error loading configurations: {str(e)}")
        st.error(f"Current working directory: {os.getcwd()}")
        raise

def initialize_services(common_config):
    try:
        bedrock_service = BedrockService(common_config['model_config'])
        return bedrock_service
    except Exception as e:
        st.error(f"Error initializing services: {str(e)}")
        raise

def render_main_cards(cards_config):
    """Render main navigation cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    for i, (title, data) in enumerate(cards_config.items()):
        with [col1, col2, col3, col4][i % 4]:
            form_key = f"form_{data['view']}_{i}"
            with st.form(key=form_key):
                st.markdown(
                    f"""
                    <div class="item-card">
                        <div>
                            <h3>{data['icon']} {title}</h3>
                            <p>{data['description']}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.form_submit_button("Open", use_container_width=True):
                    if title == "Documents":
                        st.session_state.current_view = 'documents'
                    else:
                        st.session_state.current_view = data['view']
                    st.rerun()

def render_report_view(common_config: dict, bank_configs: dict, tab_configs: dict):
    """Render the report generation view."""
    try:
        # Initialize services
        s3_service = S3Service(common_config['s3_config'])
        bedrock_service = BedrockService(common_config['model_config'])
        
        report_ui = ReportUI(
            common_config=common_config,
            s3_service=s3_service,
            bedrock_service=bedrock_service
        )
        report_ui.render()
    except Exception as e:
        st.error(f"Error rendering report view: {str(e)}")
        st.error("Please check configurations and try again.")

def render_document_cards(sub_cards_config):
    """Render document management and exploration cards."""
    st.markdown("")
    col1, col2, col3 = st.columns([0.01, 10.5, 1])
    with col2:
        st.markdown("### Document Assistant")
    with col3:
        if st.button("← Back", key="document_back", use_container_width=True):
            st.session_state.current_view = 'main'
            st.rerun()
    
    # Create two columns for the two cards
    col1, col2 = st.columns(2)
    
    for (title, data), col in zip(sub_cards_config.items(), [col1, col2]):
        with col:
            with st.container():
                st.markdown(
                    f"""
                    <div class="item-card">
                        <div>
                            <h3>{data['icon']} {title}</h3>
                            <p>{data['description']}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button("Open", key=f"btn_{data['view']}", use_container_width=True):
                    st.session_state.current_view = data['view']
                    st.rerun()

def render_regular_cards(cards_config):
    """Render the main navigation cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    for i, (title, data) in enumerate(cards_config.items()):
        with [col1, col2, col3, col4][i % 4]:
            form_key = f"form_{data['view']}_{i}"
            with st.form(key=form_key):
                st.markdown(
                    f"""
                    <div class="item-card">
                        <div>
                            <h3>{data['icon']} {title}</h3>
                            <p>{data['description']}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.form_submit_button("Open", use_container_width=True):
                    st.session_state.current_view = data['view']
                    st.rerun()

def render_dashboard(common_config: Dict):
    """Render the competitor analysis dashboard."""
    try:
        dashboard = DashboardUI(common_config)
        dashboard.render()
    except Exception as e:
        st.error(f"Error rendering dashboard: {str(e)}")
        
# Update the CSS styles
def get_card_styles():
    return """
    <style>
    .item-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
        min-height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .item-card h3 {
        margin: 0;
        padding: 0;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        color: #0051A2;
    }
    
    .item-card p {
        margin: 0;
        padding: 0;
        font-size: 0.9rem;
        color: #666;
        flex-grow: 1;
    }

    .item-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    .stButton > button {
        background-color: #0051A2;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        cursor: pointer;
    }

    .stButton > button:hover {
        background-color: #003d82;
    }
    </style>
    """
    
def main():
    st.set_page_config(
        page_title="Competitor Analysis",
        layout="wide",
        page_icon="🏦",
        initial_sidebar_state="expanded",
        menu_items=None
    )

    try:
        # Load configurations
        common_config, bank_configs, tab_configs = load_configs()
        
        # Initialize services
        bedrock_service = initialize_services(common_config)

        # Apply UI styling
        st.markdown(MAIN_UI_CONFIG, unsafe_allow_html=True)
        st.markdown(get_card_styles(), unsafe_allow_html=True)

        # Initialize session state
        if 'current_view' not in st.session_state:
            st.session_state.current_view = 'main'

        # Render Barclays logo
        barclays_logo_url = common_config.get("app_config", {}).get("barclays_logo_url", "")
        if barclays_logo_url:
            st.markdown(f"""
                <div class="logo-container">
                    <img src="{barclays_logo_url}" alt="Barclays Logo" style="height: 25px;" />
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
            
        # Initialize driver year and period if not in session state
        if 'driver_year' not in st.session_state:
            # Get current year
            driver_year_config = common_config.get("app_config", {}).get("driver_year", "")
            driver_period_config = common_config.get("app_config", {}).get("driver_period", "")
            #st.session_state.driver_year = str(datetime.now().year)
            st.session_state.driver_year = driver_year_config
            
            # Get current quarter
            #current_month = datetime.now().month
            #quarter_map = {1: 'Q1', 2: 'Q1', 3: 'Q1',
            #            4: 'Q2', 5: 'Q2', 6: 'Q2',
            #            7: 'Q3', 8: 'Q3', 9: 'Q3',
            #            10: 'Q4', 11: 'Q4', 12: 'Q4'}
            #st.session_state.driver_period = quarter_map[current_month]
            st.session_state.driver_period = driver_period_config        

        # Initialize driver values
        if 'driver_values' not in st.session_state:
            file_path = get_driver_file_path(
                st.session_state.driver_year,
                st.session_state.driver_period
            )
            st.session_state.driver_values = load_driver_values_from_file(file_path)
            if not st.session_state.driver_values:
                st.session_state.driver_values = generate_driver_values(
                    bedrock_service,
                    st.session_state.driver_year,
                    st.session_state.driver_period
                )
        
        # Render header
        st.markdown("""
            <div class="header">
                <h1 style="color: white; margin-bottom: 1rem;">Welcome to Competitor Analysis, Veerabadran</h1>
                <div class="search-container">
                    <input type="text" class="search-input" placeholder="What can we help you with today?">
                    <button class="search-button">Search</button>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display derived application drivers information
        display_app_drivers(common_config, bedrock_service)
        #print(st.session_state.get('driver_values', {}))
        # Main content routing
        if st.session_state.current_view == 'main':
            render_main_cards(common_config.get('app_config', {}).get('cards', {}))
            
             # Add a divider
            st.markdown("---")
            
            # Render dashboard (now using S3 data loading)
            render_dashboard(common_config)
        elif st.session_state.current_view in ['documents', 'manage_documents', 'explore_documents', 'ccar_agent']:
            document_ui = DocumentAssistantUI(common_config)
            document_ui.render()
        elif st.session_state.current_view == 'summary_analysis':
            summary_ui = SummaryAnalysisUI(
                common_config=common_config,
                bank_configs=bank_configs,
                tab_configs=tab_configs,
                bedrock_service=bedrock_service
            )
            summary_ui.render()
        elif st.session_state.current_view == 'report':
            render_report_view(common_config, bank_configs, tab_configs)
        elif st.session_state.current_view == 'admin':
            admin_ui = AdminUI(
                common_config=common_config,
                bank_configs=bank_configs
            )
            admin_ui.render()

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please check the configuration and try again.")

if __name__ == "__main__":
    main()