# src/app.py
# This is a Streamlit dashboard application for analyzing UK Social Housing data
# It integrates data from multiple sources and provides visualization and filtering capabilities

import streamlit as st  # Main dashboard framework
import polars as pl  # Data processing library (faster alternative to pandas)
import plotly.express as px  # High-level plotting library
from pathlib import Path  # OS-independent path handling
import logging  # For application logging
import plotly.graph_objects as go  # Low-level plotting library for more control

# Set up logging to track application execution and errors
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO
logger = logging.getLogger(__name__)  # Get logger for this module

# Configure the Streamlit page appearance
st.set_page_config(page_title="UK Social Housing SFF Dashboard", layout="wide")  # Use wide layout for better visualization

# Define data directory path - navigate from current file to data directory
BASE_DIR = Path(__file__).parent.parent  # Gets to socialhousing folder (two levels up from this file)
DATA_DIR = BASE_DIR / "data"  # Points to the data subdirectory

# Define expected columns for Judgement Data (JD) schema
# This helps ensure consistent data structure when combining current and archived judgements
JD_COLUMNS = [
    "Reg Code", "Provider", "Landlord Type", "Name and Reg Code Change Details",
    "Other providers included in the judgement", "Status", "Con", "Con Date",
    "Con Change", "Gov", "Gov Date", "Gov Change", "Via", "Via Date", "Via Change",
    "Rent Y", "Rent Date", "Rent Change", "Type of Publication", "Publication Date",
    "Engagement Process", "Route", "Explanation"
]

# Define SIC (Standard Industrial Classification) codes relevant for social housing
# These codes are used to filter companies from the Company House dataset
HOUSING_SIC_CODES = [
    "68201 - Renting and operating of Housing Association real estate",  # Primary code for housing associations
    "68209 - Other letting and operating of own or leased real estate",  # Related real estate activities
    "68100 - Buying and selling of own real estate"  # Additional real estate category
]

# Load and process data function with caching to improve performance
@st.cache_data  # Streamlit's caching decorator to avoid reloading data on every interaction
def load_and_process_data():
    try:
        # Load File-1 (Company House Data) with lazy loading and SIC filter
        # Lazy loading defers execution until collect() is called for better memory efficiency
        cd = pl.scan_csv(DATA_DIR / "BasicCompanyDataAsOneFile-2025-03-01.csv").filter(
            pl.col("SICCode.SicText_1").is_in(HOUSING_SIC_CODES)  # Filter by relevant SIC codes
        ).with_columns(
            pl.col("CompanyName").str.to_lowercase().alias("CompanyName_lower")  # Create lowercase column for case-insensitive joins
        )
        logger.info(f"Successfully scanned Company House Data with {cd.collect().height} rows after SIC filter")
        
        # Load File-2 (Registered Providers from the regulator)
        rp = pl.read_excel(
            DATA_DIR / "List_of_registered_providers_17_February_2025.xlsx",
            sheet_name="Organisation Advanced Find View",  # Specific sheet to read
            read_options={"header_row": 0, "skip_rows": 0}  # Read from the first row
        ).with_columns(
            pl.col("Organisation name").str.to_lowercase().alias("Organisation_name_lower")  # Create lowercase column for joins
        )
        logger.info("Successfully loaded Registered Providers")
        
        # Verify essential columns exist in the registered providers data
        if "Organisation name" not in rp.columns:
            raise ValueError("Column 'Organisation name' not found in Registered Providers data")
            
        # Join Company House data with Registered Providers data
        # This identifies which companies are officially registered social housing providers
        cd = cd.join(
            rp.lazy(),  # Convert to lazy dataframe for efficient joining
            left_on="CompanyName_lower",  # Join on lowercase company name
            right_on="Organisation_name_lower",  # Match with lowercase organization name
            how="left"  # Keep all companies, even those not in the registered providers list
        )
        
        # Load File-3 (Current Regulatory Judgements)
        rj_current = pl.read_excel(
            DATA_DIR / "20250226_RegulatoryJudgementsNotices_Published.xlsx",
            sheet_name="RegulatoryJudgements",
            read_options={"header_row": 2, "skip_rows": 2}  # Skip header rows
        ).with_columns([
            pl.lit("").alias("Route"),  # Add empty columns for schema alignment
            pl.lit("").alias("Explanation"),
            pl.col("Provider").str.to_lowercase().alias("Provider_lower")  # Create lowercase for joins
        ])
        logger.info("Successfully loaded Current Judgements")
        
        # Load File-4 (Archived Regulatory Judgements)
        rj_archived = pl.read_excel(
            DATA_DIR / "20240627_FINALRegulatoryJudgementsNotices_Archived.xlsx",
            sheet_name="RegulatoryJudgementsNotices",
            read_options={"header_row": 1, "skip_rows": 1}  # Skip header row
        ).with_columns(
            pl.col("Provider").str.to_lowercase().alias("Provider_lower")  # Create lowercase for joins
        )
        logger.info("Successfully loaded Archived Judgements")
        
        # Process archived judgements to avoid duplicates with current judgements
        rj_archived = rj_archived.filter(
            ~pl.col("Provider_lower").is_in(rj_current["Provider_lower"])  # Exclude providers already in current judgements
        ).sort("Publication Date", descending=True).unique(  # Sort by date and keep only the most recent judgment
            subset=["Provider_lower"], keep="first"
        )
        
        # Align archived judgement columns with the standard schema
        available_cols = [col for col in JD_COLUMNS if col in rj_archived.columns]  # Find columns that exist
        missing_cols = [col for col in JD_COLUMNS if col not in rj_archived.columns]  # Find missing columns
        
        # Select available columns and add missing ones as null values
        rj_archived = rj_archived.select(available_cols).with_columns(
            [pl.lit(None).alias(col) for col in missing_cols]  # Add null columns for missing fields
        ).select(JD_COLUMNS)  # Reorder columns to match schema
        
        # Ensure current judgements match the schema as well
        rj_current = rj_current.select(
            [col for col in JD_COLUMNS if col in rj_current.columns]  # Select existing columns
        ).with_columns(
            [pl.lit(None).alias(col) for col in JD_COLUMNS if col not in rj_current.columns]  # Add missing columns
        ).select(JD_COLUMNS)  # Reorder columns
        
        # Combine current and archived judgements
        jd = pl.concat([rj_current, rj_archived]).with_columns(
            pl.col("Provider").str.to_lowercase().alias("Provider_lower")  # Ensure provider name is lowercase for joining
        )
        
        # Create final dataset by joining company data with judgement data
        final_data = cd.join(
            jd.lazy(),
            left_on="CompanyName_lower",  # Join on lowercase company name
            right_on="Provider_lower",  # Match with lowercase provider name
            how="left"  # Keep all companies, even those without judgements
        ).collect()  # Execute all the lazy operations and collect into memory
        
        # Log success statistics
        logger.info("Data processing completed successfully")
        logger.info(f"Rows with registration: {final_data.filter(pl.col('Registration number').is_not_null()).height}")
        logger.info(f"Rows with judgements: {final_data.filter(pl.col('Status').is_not_null()).height}")
        
        return cd.collect(), jd, final_data  # Return all three datasets
    
    except Exception as e:
        # Log and display any errors during data processing
        logger.error(f"Error in data processing: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None, None, None  # Return None if data loading fails

# Page header
st.subheader("Social Housing Analysis")
st.divider()

# Load all data using the cached function
cd, jd, final_data = load_and_process_data()

# Stop execution if data loading failed
if final_data is None:
    st.stop()

# Create sidebar filters to interact with the data
st.sidebar.header("Filters")

# Text search filter
search_term = st.sidebar.text_input("Search Company/Provider", "")

# Company status filter (multiselect)
status_filter = st.sidebar.multiselect(
    "Company Status", 
    options=final_data["CompanyStatus"].unique().to_list(),
    default="Active"  # Default to show only active companies
)

# Registration status filter (dropdown)
reg_filter = st.sidebar.selectbox(
    "Registration Status",
    ["All", "Registered", "Unregistered"],
    index=0  # Default to "All"
)

# Governance rating filter (multiselect)
gov_filter = st.sidebar.multiselect(
    "Governance Rating",
    options=final_data["Gov"].unique().drop_nulls().to_list(),  # Drop null values from options
    default=[]  # No default selection
)

# Viability rating filter (multiselect)
via_filter = st.sidebar.multiselect(
    "Viability Rating",
    options=final_data["Via"].unique().drop_nulls().to_list(),  # Drop null values from options
    default=[]  # No default selection
)

# Set pagination settings for displaying large tables
ROWS_PER_PAGE = 1000  # Number of rows to show per page

# Apply all filters to the data based on user selections
filtered_data = final_data  # Start with all data

# Apply text search filter
if search_term:
    filtered_data = filtered_data.filter(
        # Case-insensitive search in company name or provider name
        pl.col("CompanyName_lower").str.contains(search_term.lower(), literal=True) |
        pl.col("Provider_lower").str.contains(search_term.lower(), literal=True)
    )

# Apply company status filter
if status_filter:
    filtered_data = filtered_data.filter(pl.col("CompanyStatus").is_in(status_filter))

# Apply registration status filter
if reg_filter != "All":
    if reg_filter == "Registered":
        filtered_data = filtered_data.filter(pl.col("Registration number").is_not_null())
    else:  # Unregistered
        filtered_data = filtered_data.filter(pl.col("Registration number").is_null())

# Apply governance rating filter
if gov_filter:
    filtered_data = filtered_data.filter(pl.col("Gov").is_in(gov_filter))

# Apply viability rating filter
if via_filter:
    filtered_data = filtered_data.filter(pl.col("Via").is_in(via_filter))

# Main dashboard title
st.title("UK Social Housing - Sustainable Finance Framework Dashboard")

# Function to display paginated dataframes with download button
def display_paginated_df(df, title, key):
    total_rows = len(df)
    total_pages = (total_rows + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE  # Calculate total pages needed
    
    # Page selector input
    page = st.number_input(
        f"Page (1-{total_pages})",
        min_value=1,
        max_value=max(1, total_pages),  # Ensure at least 1 page
        value=1,
        key=f"{key}_page"
    )
    
    # Calculate start and end indices for the current page
    start_idx = (page - 1) * ROWS_PER_PAGE
    end_idx = min(start_idx + ROWS_PER_PAGE, total_rows)
    
    # Display the current page of data
    st.dataframe(df.slice(start_idx, end_idx - start_idx), use_container_width=True)
    st.write(f"Showing rows {start_idx + 1} to {end_idx} of {total_rows}")
    
    # Create download button for the full dataset
    csv = df.write_csv()
    st.download_button(
        label=f"Download {title} Data",
        data=csv,
        file_name=f"{title.lower().replace(' ', '_')}_data.csv",
        mime="text/csv",
        key=f"{key}_download"
    )

# Create expandable sections for the raw datasets
with st.expander("Company House Data (CD)", expanded=False):
    display_paginated_df(cd, "Company House", "cd")

with st.expander("Judgement Data (JD)", expanded=False):
    display_paginated_df(jd, "Judgement", "jd")

with st.expander("Final Dataset", expanded=False):
    display_paginated_df(filtered_data, "Final", "final")

# Visualizations section
st.subheader("Visual Analytics")

# Debug information to help troubleshoot
st.write(f"Total rows in filtered_data: {filtered_data.height}")

# Calculate registration counts for charts
reg_count = filtered_data.filter(pl.col("Registration number").is_not_null()).height
unreg_count = filtered_data.filter(pl.col("Registration number").is_null()).height

# Create two-column layout for charts
col1, col2 = st.columns(2)

with col1:
    # Registration Status Pie Chart using Plotly Graph Objects
    try:
        # Create pie chart with registered vs unregistered counts
        fig1 = go.Figure(data=[go.Pie(
            labels=['Registered', 'Unregistered'],
            values=[reg_count, unreg_count],
            hole=0.3,  # Create a donut chart
            marker_colors=['rgb(65, 105, 225)', 'rgb(135, 206, 250)']  # Blue colors
        )])
        
        # Set chart layout options
        fig1.update_layout(
            title="Registered vs Unregistered Providers",
            height=400  # Fixed height
        )
        
        # Display the chart
        st.plotly_chart(fig1, use_container_width=True)
        
        # Show raw numbers for debugging/verification
        st.write(f"Registered: {reg_count}, Unregistered: {unreg_count}")
    except Exception as e:
        # Handle any errors in chart creation
        st.error(f"Error creating registration chart: {str(e)}")
        # Fallback to text display if chart fails
        st.write(f"Registered: {reg_count}, Unregistered: {unreg_count}")

with col2:
    # Judgement Status Bar Chart using Plotly Graph Objects
    judgment_count = filtered_data.filter(pl.col("Status").is_not_null()).height
    no_judgment_count = filtered_data.filter(pl.col("Status").is_null()).height
    
    try:
        # Create bar chart comparing judgment counts
        fig2 = go.Figure(data=[go.Bar(
            x=['No Judgement', 'With Judgement'],
            y=[no_judgment_count, judgment_count],
            marker_color=['rgb(102, 178, 255)', 'rgb(65, 105, 225)']  # Blue colors
        )])
        
        # Set chart layout with proper y-axis range
        fig2.update_layout(
            title="Providers by Judgement Status",
            yaxis=dict(
                title="Count",
                range=[0, max(no_judgment_count, judgment_count) * 1.1]  # Set axis to start at 0
            ),
            height=400  # Fixed height
        )
        
        # Display the chart
        st.plotly_chart(fig2, use_container_width=True)
        
        # Show raw numbers for debugging/verification
        st.write(f"No Judgement: {no_judgment_count}, With Judgement: {judgment_count}")
    except Exception as e:
        # Handle any errors in chart creation
        st.error(f"Error creating judgement chart: {str(e)}")
        # Fallback to text display if chart fails
        st.write(f"No Judgement: {no_judgment_count}, With Judgement: {judgment_count}")

# Create another two-column layout for governance and viability charts
col3, col4 = st.columns(2)

with col3:
    # Governance Ratings Bar Chart
    # Filter to only include rows with governance ratings
    gov_data = filtered_data.filter(pl.col("Gov").is_not_null())
    gov_counts = gov_data.group_by("Gov").agg(pl.len().alias("Count"))
    
    try:
        if gov_counts.height > 0:
            # Extract data for the chart
            labels = gov_counts["Gov"].to_list()
            values = gov_counts["Count"].to_list()
            
            # Create the bar chart
            fig3 = go.Figure(data=[go.Bar(
                x=labels,
                y=values,
                marker_color='rgb(65, 105, 225)'  # Blue color
            )])
            
            # Set chart layout with proper y-axis range
            fig3.update_layout(
                title="Governance Ratings Distribution",
                yaxis=dict(
                    title="Count",
                    range=[0, max(values) * 1.1] if values else [0, 10]  # Set axis to start at 0
                ),
                height=400  # Fixed height
            )
            
            # Display the chart
            st.plotly_chart(fig3, use_container_width=True)
            
            # Debug info (commented out to reduce clutter)
            #st.write("Governance ratings:")
            #for label, value in zip(labels, values):
            #    st.write(f"{label}: {value}")
        else:
            st.write("No governance ratings data available")
    except Exception as e:
        # Handle any errors in chart creation
        st.error(f"Error creating governance chart: {str(e)}")
        # Fallback to text display if chart fails
        if gov_counts.height > 0:
            for i in range(gov_counts.height):
                st.write(f"{gov_counts['Gov'][i]}: {gov_counts['Count'][i]}")
    
    # Display count of providers without governance ratings
    no_rating_count = filtered_data.filter(pl.col("Gov").is_null()).height
    st.metric("Providers with No Governance Rating", no_rating_count)

with col4:
    # Viability Ratings Bar Chart
    # Filter to only include rows with viability ratings
    via_data = filtered_data.filter(pl.col("Via").is_not_null())
    via_counts = via_data.group_by("Via").agg(pl.len().alias("Count"))
    
    try:
        if via_counts.height > 0:
            # Extract data for the chart
            labels = via_counts["Via"].to_list()
            values = via_counts["Count"].to_list()
            
            # Create the bar chart
            fig4 = go.Figure(data=[go.Bar(
                x=labels,
                y=values,
                marker_color='rgb(0, 128, 128)'  # Teal color
            )])
            
            # Set chart layout with proper y-axis range
            fig4.update_layout(
                title="Viability Ratings Distribution",
                yaxis=dict(
                    title="Count",
                    range=[0, max(values) * 1.1] if values else [0, 10]  # Set axis to start at 0
                ),
                height=400  # Fixed height
            )
            
            # Display the chart
            st.plotly_chart(fig4, use_container_width=True)
            
            # Debug info (commented out to reduce clutter)
            #st.write("Viability ratings:")
            #for label, value in zip(labels, values):
            #    st.write(f"{label}: {value}")
        else:
            st.write("No viability ratings data available")
    except Exception as e:
        # Handle any errors in chart creation
        st.error(f"Error creating viability chart: {str(e)}")
        # Fallback to text display if chart fails
        if via_counts.height > 0:
            for i in range(via_counts.height):
                st.write(f"{via_counts['Via'][i]}: {via_counts['Count'][i]}")
    
    # Display count of providers without viability ratings
    no_rating_count = filtered_data.filter(pl.col("Via").is_null()).height
    st.metric("Providers with No Viability Rating", no_rating_count)

# Key Statistics section - display metrics in 6 columns
st.subheader("Key Statistics")
col5, col6, col7, col8, col9, col10 = st.columns(6)

with col5:
    # Total number of providers after filtering
    total_providers = len(filtered_data)
    st.metric("Total Providers", total_providers)

with col6:
    # Count of registered providers
    registered = len(filtered_data.filter(pl.col("Registration number").is_not_null()))
    st.metric("Registered Providers", registered)

with col7:
    # Count of unregistered providers
    unregistered = len(filtered_data.filter(pl.col("Registration number").is_null()))
    st.metric("Unregistered Providers", unregistered)

with col8:
    # Count of providers with judgements
    with_judgement = len(filtered_data.filter(pl.col("Status").is_not_null()))
    st.metric("Providers with Judgements", with_judgement)

with col9:
    # Count of providers without judgements
    without_judgement = len(filtered_data.filter(pl.col("Status").is_null()))
    st.metric("Providers without Judgements", without_judgement)

with col10:
    # Count of SFF eligible providers (meeting all criteria)
    eligible = len(filtered_data.filter(
        (pl.col("Registration number").is_not_null()) &  # Must be registered
        (pl.col("CompanyStatus") == "Active") &          # Must be active
        (pl.col("Gov").is_in(["G1", "G2"])) &           # Must have acceptable governance rating
        (pl.col("Via").is_in(["V1", "V2"]))             # Must have acceptable viability rating
    ))
    st.metric("SFF Eligible Providers", eligible)

# Additional Downloadable Datasets section - filtered views for specific analyses
st.subheader("Downloadable Datasets")

# Registered Providers dataset
with st.expander("Registered Providers Dataset", expanded=False):
    registered_df = filtered_data.filter(pl.col("Registration number").is_not_null())
    display_paginated_df(registered_df, "Registered Providers", "registered")

# Unregistered Providers dataset
with st.expander("Unregistered Providers Dataset", expanded=False):
    unregistered_df = filtered_data.filter(pl.col("Registration number").is_null())
    display_paginated_df(unregistered_df, "Unregistered Providers", "unregistered")

# Providers with Judgements dataset
with st.expander("Providers with Judgements Dataset", expanded=False):
    with_judgement_df = filtered_data.filter(pl.col("Status").is_not_null())
    display_paginated_df(with_judgement_df, "Providers with Judgements", "with_judgement")

# Providers without Judgements dataset
with st.expander("Providers without Judgements Dataset", expanded=False):
    without_judgement_df = filtered_data.filter(pl.col("Status").is_null())
    display_paginated_df(without_judgement_df, "Providers without Judgements", "without_judgement")

# SFF Eligible Providers dataset
with st.expander("SFF Eligible Providers Dataset", expanded=False):
    eligible_df = filtered_data.filter(
        (pl.col("Registration number").is_not_null()) &  # Must be registered
        (pl.col("CompanyStatus") == "Active") &          # Must be active
        (pl.col("Gov").is_in(["G1", "G2"])) &           # Must have acceptable governance rating
        (pl.col("Via").is_in(["V1", "V2"]))             # Must have acceptable viability rating
    )
    display_paginated_df(eligible_df, "SFF Eligible Providers", "eligible")

# Information section explaining SFF eligibility criteria
st.subheader("Sustainable Finance Framework Eligibility Criteria")
st.markdown("""
Providers are typically SFF eligible if they:
- Are registered with the Regulator of Social Housing
- Have active company status
- Meet governance standards (G1 or G2 ratings)
- Meet viability standards (V1 or V2 ratings)
""")
