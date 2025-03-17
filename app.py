# src/app.py
import streamlit as st
import polars as pl
import plotly.express as px
from pathlib import Path
import logging
import plotly.graph_objects as go

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(page_title="UK Social Housing SFF Dashboard", layout="wide")

# Define data directory path
BASE_DIR = Path(__file__).parent.parent  # Gets to socialhousing folder
DATA_DIR = BASE_DIR / "data"

# Define JD schema
JD_COLUMNS = [
    "Reg Code", "Provider", "Landlord Type", "Name and Reg Code Change Details",
    "Other providers included in the judgement", "Status", "Con", "Con Date",
    "Con Change", "Gov", "Gov Date", "Gov Change", "Via", "Via Date", "Via Change",
    "Rent Y", "Rent Date", "Rent Change", "Type of Publication", "Publication Date",
    "Engagement Process", "Route", "Explanation"
]

# Define SIC codes for filtering
HOUSING_SIC_CODES = [
    "68201 - Renting and operating of Housing Association real estate",
    "68209 - Other letting and operating of own or leased real estate",
    "68100 - Buying and selling of own real estate"
]

# Load and process data
@st.cache_data
def load_and_process_data():
    try:
        # Load File-1 (Company House Data) with lazy loading and SIC filter
        cd = pl.scan_csv(DATA_DIR / "BasicCompanyDataAsOneFile-2025-03-01.csv").filter(
            pl.col("SICCode.SicText_1").is_in(HOUSING_SIC_CODES)
        ).with_columns(
            pl.col("CompanyName").str.to_lowercase().alias("CompanyName_lower")
        )
        logger.info(f"Successfully scanned Company House Data with {cd.collect().height} rows after SIC filter")
        
        # Load File-2 (Registered Providers)
        rp = pl.read_excel(
            DATA_DIR / "List_of_registered_providers_17_February_2025.xlsx",
            sheet_name="Organisation Advanced Find View",
            read_options={"header_row": 0, "skip_rows": 0}
        ).with_columns(
            pl.col("Organisation name").str.to_lowercase().alias("Organisation_name_lower")
        )
        logger.info("Successfully loaded Registered Providers")
        
        # Verify columns in rp
        if "Organisation name" not in rp.columns:
            raise ValueError("Column 'Organisation name' not found in Registered Providers data")
            
        # Prepare CD dataset with case-insensitive join
        cd = cd.join(
            rp.lazy(),
            left_on="CompanyName_lower",
            right_on="Organisation_name_lower",
            how="left"
        )
        
        # Load File-3 (Current Judgements)
        rj_current = pl.read_excel(
            DATA_DIR / "20250226_RegulatoryJudgementsNotices_Published.xlsx",
            sheet_name="RegulatoryJudgements",
            read_options={"header_row": 2, "skip_rows": 2}
        ).with_columns([
            pl.lit("").alias("Route"),
            pl.lit("").alias("Explanation"),
            pl.col("Provider").str.to_lowercase().alias("Provider_lower")
        ])
        logger.info("Successfully loaded Current Judgements")
        
        # Load File-4 (Archived Judgements)
        rj_archived = pl.read_excel(
            DATA_DIR / "20240627_FINALRegulatoryJudgementsNotices_Archived.xlsx",
            sheet_name="RegulatoryJudgementsNotices",
            read_options={"header_row": 1, "skip_rows": 1}
        ).with_columns(
            pl.col("Provider").str.to_lowercase().alias("Provider_lower")
        )
        logger.info("Successfully loaded Archived Judgements")
        
        # Filter archived data and align with JD schema
        rj_archived = rj_archived.filter(
            ~pl.col("Provider_lower").is_in(rj_current["Provider_lower"])
        ).sort("Publication Date", descending=True).unique(
            subset=["Provider_lower"], keep="first"
        )
        
        # Select and align columns for rj_archived
        available_cols = [col for col in JD_COLUMNS if col in rj_archived.columns]
        missing_cols = [col for col in JD_COLUMNS if col not in rj_archived.columns]
        
        rj_archived = rj_archived.select(available_cols).with_columns(
            [pl.lit(None).alias(col) for col in missing_cols]
        ).select(JD_COLUMNS)
        
        # Ensure rj_current has all JD columns
        rj_current = rj_current.select(
            [col for col in JD_COLUMNS if col in rj_current.columns]
        ).with_columns(
            [pl.lit(None).alias(col) for col in JD_COLUMNS if col not in rj_current.columns]
        ).select(JD_COLUMNS)
        
        # Prepare JD dataset
        jd = pl.concat([rj_current, rj_archived]).with_columns(
            pl.col("Provider").str.to_lowercase().alias("Provider_lower")
        )
        
        # Prepare Final Dataset with case-insensitive join
        final_data = cd.join(
            jd.lazy(),
            left_on="CompanyName_lower",
            right_on="Provider_lower",
            how="left"
        ).collect()  # Collect only at the end
        
        logger.info("Data processing completed successfully")
        logger.info(f"Rows with registration: {final_data.filter(pl.col('Registration number').is_not_null()).height}")
        logger.info(f"Rows with judgements: {final_data.filter(pl.col('Status').is_not_null()).height}")
        return cd.collect(), jd, final_data
    
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

st.subheader("Social Housing Analysis")
st.divider()
# Load data
cd, jd, final_data = load_and_process_data()

if final_data is None:
    st.stop()

# Sidebar for filters
st.sidebar.header("Filters")
search_term = st.sidebar.text_input("Search Company/Provider", "")
status_filter = st.sidebar.multiselect(
    "Company Status", 
    options=final_data["CompanyStatus"].unique().to_list(),
    default="Active"
)
reg_filter = st.sidebar.selectbox(
    "Registration Status",
    ["All", "Registered", "Unregistered"],
    index=0
)
gov_filter = st.sidebar.multiselect(
    "Governance Rating",
    options=final_data["Gov"].unique().drop_nulls().to_list(),
    default=[]
)
via_filter = st.sidebar.multiselect(
    "Viability Rating",
    options=final_data["Via"].unique().drop_nulls().to_list(),
    default=[]
)

# Pagination settings
ROWS_PER_PAGE = 1000

# Apply filters (case-insensitive search)
filtered_data = final_data
if search_term:
    filtered_data = filtered_data.filter(
        pl.col("CompanyName_lower").str.contains(search_term.lower(), literal=True) |
        pl.col("Provider_lower").str.contains(search_term.lower(), literal=True)
    )
if status_filter:
    filtered_data = filtered_data.filter(pl.col("CompanyStatus").is_in(status_filter))
if reg_filter != "All":
    if reg_filter == "Registered":
        filtered_data = filtered_data.filter(pl.col("Registration number").is_not_null())
    else:
        filtered_data = filtered_data.filter(pl.col("Registration number").is_null())
if gov_filter:
    filtered_data = filtered_data.filter(pl.col("Gov").is_in(gov_filter))
if via_filter:
    filtered_data = filtered_data.filter(pl.col("Via").is_in(via_filter))

# Main dashboard
st.title("UK Social Housing - Sustainable Finance Framework Dashboard")

# Dataset Expanders with Pagination
def display_paginated_df(df, title, key):
    total_rows = len(df)
    total_pages = (total_rows + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE
    
    page = st.number_input(
        f"Page (1-{total_pages})",
        min_value=1,
        max_value=total_pages,
        value=1,
        key=f"{key}_page"
    )
    
    start_idx = (page - 1) * ROWS_PER_PAGE
    end_idx = min(start_idx + ROWS_PER_PAGE, total_rows)
    
    st.dataframe(df.slice(start_idx, end_idx - start_idx), use_container_width=True)
    st.write(f"Showing rows {start_idx + 1} to {end_idx} of {total_rows}")
    csv = df.write_csv()
    st.download_button(
        label=f"Download {title} Data",
        data=csv,
        file_name=f"{title.lower().replace(' ', '_')}_data.csv",
        mime="text/csv",
        key=f"{key}_download"
    )

with st.expander("Company House Data (CD)", expanded=False):
    display_paginated_df(cd, "Company House", "cd")

with st.expander("Judgement Data (JD)", expanded=False):
    display_paginated_df(jd, "Judgement", "jd")

with st.expander("Final Dataset", expanded=False):
    display_paginated_df(filtered_data, "Final", "final")

# Visualizations
st.subheader("Visual Analytics")

# Debug information
st.write(f"Total rows in filtered_data: {filtered_data.height}")

# Get registration counts
reg_count = filtered_data.filter(pl.col("Registration number").is_not_null()).height
unreg_count = filtered_data.filter(pl.col("Registration number").is_null()).height

col1, col2 = st.columns(2)

with col1:
    # Registration Status Pie Chart with Graph Objects
    try:
        fig1 = go.Figure(data=[go.Pie(
            labels=['Registered', 'Unregistered'],
            values=[reg_count, unreg_count],
            hole=0.3,
            marker_colors=['rgb(65, 105, 225)', 'rgb(135, 206, 250)']
        )])
        
        fig1.update_layout(
            title="Registered vs Unregistered Providers",
            height=400
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Debug info
        st.write(f"Registered: {reg_count}, Unregistered: {unreg_count}")
    except Exception as e:
        st.error(f"Error creating registration chart: {str(e)}")
        # Fallback display
        st.write(f"Registered: {reg_count}, Unregistered: {unreg_count}")

with col2:
    # Judgement Status Bar Chart with Graph Objects
    judgment_count = filtered_data.filter(pl.col("Status").is_not_null()).height
    no_judgment_count = filtered_data.filter(pl.col("Status").is_null()).height
    
    try:
        fig2 = go.Figure(data=[go.Bar(
            x=['No Judgement', 'With Judgement'],
            y=[no_judgment_count, judgment_count],
            marker_color=['rgb(102, 178, 255)', 'rgb(65, 105, 225)']
        )])
        
        fig2.update_layout(
            title="Providers by Judgement Status",
            yaxis=dict(
                title="Count",
                range=[0, max(no_judgment_count, judgment_count) * 1.1]
            ),
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Debug info
        st.write(f"No Judgement: {no_judgment_count}, With Judgement: {judgment_count}")
    except Exception as e:
        st.error(f"Error creating judgement chart: {str(e)}")
        # Fallback display
        st.write(f"No Judgement: {no_judgment_count}, With Judgement: {judgment_count}")

# Process governance and viability ratings
col3, col4 = st.columns(2)

with col3:
    # Collect governance ratings excluding "No Rating"
    gov_data = filtered_data.filter(pl.col("Gov").is_not_null())
    gov_counts = gov_data.group_by("Gov").agg(pl.len().alias("Count"))
    
    try:
        if gov_counts.height > 0:
            labels = gov_counts["Gov"].to_list()
            values = gov_counts["Count"].to_list()
            
            fig3 = go.Figure(data=[go.Bar(
                x=labels,
                y=values,
                marker_color='rgb(65, 105, 225)'
            )])
            
            fig3.update_layout(
                title="Governance Ratings Distribution",
                yaxis=dict(
                    title="Count",
                    range=[0, max(values) * 1.1] if values else [0, 10]
                ),
                height=400
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Debug info
            #st.write("Governance ratings:")
            #for label, value in zip(labels, values):
            #    st.write(f"{label}: {value}")
        else:
            st.write("No governance ratings data available")
    except Exception as e:
        st.error(f"Error creating governance chart: {str(e)}")
        # Fallback display
        if gov_counts.height > 0:
            for i in range(gov_counts.height):
                st.write(f"{gov_counts['Gov'][i]}: {gov_counts['Count'][i]}")
    
    # Display No Rating count
    no_rating_count = filtered_data.filter(pl.col("Gov").is_null()).height
    st.metric("Providers with No Governance Rating", no_rating_count)

with col4:
    # Collect viability ratings excluding "No Rating"
    via_data = filtered_data.filter(pl.col("Via").is_not_null())
    via_counts = via_data.group_by("Via").agg(pl.len().alias("Count"))
    
    try:
        if via_counts.height > 0:
            labels = via_counts["Via"].to_list()
            values = via_counts["Count"].to_list()
            
            fig4 = go.Figure(data=[go.Bar(
                x=labels,
                y=values,
                marker_color='rgb(0, 128, 128)'
            )])
            
            fig4.update_layout(
                title="Viability Ratings Distribution",
                yaxis=dict(
                    title="Count",
                    range=[0, max(values) * 1.1] if values else [0, 10]
                ),
                height=400
            )
            
            st.plotly_chart(fig4, use_container_width=True)
            
            # Debug info
            #st.write("Viability ratings:")
            #for label, value in zip(labels, values):
            #    st.write(f"{label}: {value}")
        else:
            st.write("No viability ratings data available")
    except Exception as e:
        st.error(f"Error creating viability chart: {str(e)}")
        # Fallback display
        if via_counts.height > 0:
            for i in range(via_counts.height):
                st.write(f"{via_counts['Via'][i]}: {via_counts['Count'][i]}")
    
    # Display No Rating count
    no_rating_count = filtered_data.filter(pl.col("Via").is_null()).height
    st.metric("Providers with No Viability Rating", no_rating_count)

# Statistics
st.subheader("Key Statistics")
col5, col6, col7, col8, col9, col10 = st.columns(6)
with col5:
    total_providers = len(filtered_data)
    st.metric("Total Providers", total_providers)
with col6:
    registered = len(filtered_data.filter(pl.col("Registration number").is_not_null()))
    st.metric("Registered Providers", registered)
with col7:
    unregistered = len(filtered_data.filter(pl.col("Registration number").is_null()))
    st.metric("Unregistered Providers", unregistered)
with col8:
    with_judgement = len(filtered_data.filter(pl.col("Status").is_not_null()))
    st.metric("Providers with Judgements", with_judgement)
with col9:
    without_judgement = len(filtered_data.filter(pl.col("Status").is_null()))
    st.metric("Providers without Judgements", without_judgement)
with col10:
    eligible = len(filtered_data.filter(
        (pl.col("Registration number").is_not_null()) &
        (pl.col("CompanyStatus") == "Active") &
        (pl.col("Gov").is_in(["G1", "G2"])) &
        (pl.col("Via").is_in(["V1", "V2"]))
    ))
    st.metric("SFF Eligible Providers", eligible)

# Additional Downloadable Datasets
st.subheader("Downloadable Datasets")
with st.expander("Registered Providers Dataset", expanded=False):
    registered_df = filtered_data.filter(pl.col("Registration number").is_not_null())
    display_paginated_df(registered_df, "Registered Providers", "registered")

with st.expander("Unregistered Providers Dataset", expanded=False):
    unregistered_df = filtered_data.filter(pl.col("Registration number").is_null())
    display_paginated_df(unregistered_df, "Unregistered Providers", "unregistered")

with st.expander("Providers with Judgements Dataset", expanded=False):
    with_judgement_df = filtered_data.filter(pl.col("Status").is_not_null())
    display_paginated_df(with_judgement_df, "Providers with Judgements", "with_judgement")

with st.expander("Providers without Judgements Dataset", expanded=False):
    without_judgement_df = filtered_data.filter(pl.col("Status").is_null())
    display_paginated_df(without_judgement_df, "Providers without Judgements", "without_judgement")

with st.expander("SFF Eligible Providers Dataset", expanded=False):
    eligible_df = filtered_data.filter(
        (pl.col("Registration number").is_not_null()) &
        (pl.col("CompanyStatus") == "Active") &
        (pl.col("Gov").is_in(["G1", "G2"])) &
        (pl.col("Via").is_in(["V1", "V2"]))
    )
    display_paginated_df(eligible_df, "SFF Eligible Providers", "eligible")

# SFF Eligibility Section
st.subheader("Sustainable Finance Framework Eligibility Criteria")
st.markdown("""
Providers are typically SFF eligible if they:
- Are registered with the Regulator of Social Housing
- Have active company status
- Meet governance standards (G1 or G2 ratings)
- Meet viability standards (V1 or V2 ratings)
""")