# src/app.py
# Streamlit dashboard application for analyzing UK Social Housing data
# Modularized to separate data loading, cleaning, processing, and visualization logic
# Uses Polars for BasicCompanyDataAsOneFile-2025-03-01.csv, Pandas for all other processing

import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
from pathlib import Path
import logging
import plotly.graph_objects as go
import traceback
from rich import print
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(page_title="UK Social Housing SFF Dashboard", layout="wide")

# Define paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Define expected columns for Judgement Data
JD_COLUMNS = [
    "Reg Code", "Provider", "Landlord Type", "Name and Reg Code Change Details",
    "Other providers included in the judgement", "Status", "Con", "Con Date",
    "Con Change", "Gov", "Gov Date", "Gov Change", "Via", "Via Date", "Via Change",
    "Rent Y", "Rent Date", "Rent Change", "Type of Publication", "Publication Date",
    "Engagement Process", "Route", "Explanation"
]

# Define SIC codes for social housing
HOUSING_SIC_CODES = [
    "68201 - Renting and operating of Housing Association real estate",
    "68209 - Other letting and operating of own or leased real estate",
    "68100 - Buying and selling of own real estate"
]

# Data loading functions
@st.cache_data
def load_company_data():
    """Load and clean Company House Data using Polars."""
    try:
        raw_cd = pl.read_csv(DATA_DIR / "BasicCompanyDataAsOneFile-2025-03-01.csv")
        clean_columns = {col: col.strip().strip("'").strip() for col in raw_cd.columns}
        cd_clean = raw_cd.rename(clean_columns)
        
        cd = cd_clean.filter(
            pl.col("SICCode.SicText_1").is_in(HOUSING_SIC_CODES)
        ).with_columns([
            pl.col("CompanyName").str.to_lowercase().alias("CompanyName_lower"),
            pl.when(pl.col("CompanyNumber").str.contains("^0"))
            .then(pl.col("CompanyNumber").str.replace_all("^0+", ""))
            .otherwise(pl.col("CompanyNumber"))
            .alias("CompanyNumber")
        ])
        
        logger.info(f"Loaded Company House Data with {cd.height} rows after SIC filter")
        return cd
    except Exception as e:
        logger.error(f"Error loading Company House Data: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return fallback_load_company_data()

@st.cache_data
def fallback_load_company_data():
    """Fallback method for loading Company House Data using Polars."""
    try:
        cd = pl.scan_csv(
            DATA_DIR / "BasicCompanyDataAsOneFile-2025-03-01.csv",
            ignore_errors=True,
            infer_schema_length=1000
        )
        cd = cd.filter(
            pl.col("SICCode.SicText_1").is_in(HOUSING_SIC_CODES)
        ).with_columns([
            pl.col("CompanyName").str.to_lowercase().alias("CompanyName_lower")
        ]).collect()
        logger.info(f"Fallback loaded Company House Data with {cd.height} rows")
        return cd
    except Exception as e:
        logger.error(f"Fallback load failed: {str(e)}")
        return None

@st.cache_data
def load_current_judgments():
    """Load and clean Current Regulatory Judgements using Pandas."""
    try:
        rj_current = pd.read_excel(
            DATA_DIR / "20250226_RegulatoryJudgementsNotices_Published.xlsx",
            sheet_name="RegulatoryJudgements",
            header=0,
            skiprows=2
        )
        
        logger.info(f"Current Judgments columns: {rj_current.columns.tolist()}")
        rj_current = clean_judgment_data(rj_current)
        if rj_current is None:
            logger.error("Failed to clean Current Judgments: No Provider column found")
            return None
        logger.info(f"Loaded Current Judgments with {len(rj_current)} rows")
        return rj_current
    except Exception as e:
        logger.error(f"Error loading Current Judgments: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

@st.cache_data
def load_archived_judgments():
    """Load and clean Archived Regulatory Judgements using Pandas."""
    try:
        rj_archived = pd.read_excel(
            DATA_DIR / "20240627_FINALRegulatoryJudgementsNotices_Archived.xlsx",
            sheet_name="RegulatoryJudgementsNotices",
            header=0,
            skiprows=1
        )
        
        logger.info(f"Archived Judgments columns: {rj_archived.columns.tolist()}")
        rj_archived = clean_judgment_data(rj_archived)
        if rj_archived is None:
            logger.error("Failed to clean Archived Judgments: No Provider column found")
            return None
        if "Status" not in rj_archived.columns:
            rj_archived["Status"] = "Previous"
        logger.info(f"Loaded Archived Judgments with {len(rj_archived)} rows")
        return rj_archived
    except Exception as e:
        logger.error(f"Error loading Archived Judgments: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

@st.cache_data
def load_and_process_sds_data():
    """Load and process SDS-cohouseid_list.xlsx, joining Cohouse and BCBCUSID sheets."""
    try:
        # Read Cohouse sheet
        cohouse_df = pd.read_excel(
            DATA_DIR / "SDS-cohouseid_list.xlsx",
            sheet_name="Cohouse"
        )
        logger.info(f"Loaded Cohouse sheet with {len(cohouse_df)} rows")
        
        # Read BCBCUSID sheet
        bcbcusid_df = pd.read_excel(
            DATA_DIR / "SDS-cohouseid_list.xlsx",
            sheet_name="BCBCUSID"
        )
        logger.info(f"Loaded BCBCUSID sheet with {len(bcbcusid_df)} rows")
        
        # Rename MapId columns
        cohouse_df = cohouse_df.rename(columns={"MapId": "cohouse"})
        bcbcusid_df = bcbcusid_df.rename(columns={"MapId": "cusid"})
        
        # Join on CpartyId
        sds_data = pd.merge(
            cohouse_df[["name", "CpartyId", "cohouse"]],
            bcbcusid_df[["CpartyId", "cusid"]],
            on="CpartyId",
            how="inner"
        )
        logger.info(f"Joined SDS data with {len(sds_data)} rows")
        
        # Add ClientFlag column as boolean
        sds_data["ClientFlag"] = True  # Explicitly set as boolean
        sds_data["ClientFlag"] = sds_data["ClientFlag"].astype(bool)
        
        # Ensure output columns
        sds_data = sds_data[["name", "CpartyId", "cohouse", "cusid", "ClientFlag"]]
        return sds_data
    
    except Exception as e:
        logger.error(f"Error processing SDS-cohouseid_list.xlsx: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

@st.cache_data
def load_cre_social_housing_data():
    """Load and process CRE_Social_housing.xlsx."""
    try:
        cre_data = pd.read_excel(
            DATA_DIR / "CRE_Social_housing.xlsx",
            header=0
        )
        logger.info(f"Loaded CRE Social Housing data with {len(cre_data)} rows")
        return cre_data[["counterparty_id", "facility_id", "drawn_balance", "facility_limit", "undrawn_balance"]]
    except Exception as e:
        logger.error(f"Error loading CRE_Social_housing.xlsx: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def compute_venn_counts(cd, filtered_data):
    """Compute counts for Venn diagram of in-scope companies, SFF-eligible providers, and clients."""
    try:
        # In-scope Company House data (all rows in cd)
        in_scope_count = cd.height
        in_scope_companies = set(cd["CompanyNumber"].to_list())
        
        # SFF-eligible providers
        sff_eligible = filtered_data[
            (filtered_data["IsRegistered"]) &
            (filtered_data["CompanyStatus"] == "Active") &
            (
                (filtered_data["Con"].isin(["C1", "C2", "C3", "C4"])) |
                (filtered_data["Gov"].isin(["G1", "G2", "G3", "G4"])) |
                (filtered_data["Via"].isin(["V1", "V2", "V3", "V4"]))
            )
        ]
        sff_eligible_count = len(sff_eligible)
        sff_eligible_companies = set(sff_eligible["CompanyNumber"].dropna().to_list())
        
        # Clients
        clients = filtered_data[filtered_data["ClientFlag"] == True]
        client_count = len(clients)
        client_companies = set(clients["CompanyNumber"].dropna().to_list())
        
        # Compute intersections
        in_scope_sff = len(in_scope_companies & sff_eligible_companies)
        in_scope_client = len(in_scope_companies & client_companies)
        sff_client = len(sff_eligible_companies & client_companies)
        in_scope_sff_client = len(in_scope_companies & sff_eligible_companies & client_companies)
        
        # Compute exclusive counts for Venn diagram regions
        venn_counts = {
            "In-scope only": in_scope_count - in_scope_sff - in_scope_client + in_scope_sff_client,
            "SFF-eligible only": sff_eligible_count - in_scope_sff - sff_client + in_scope_sff_client,
            "Client only": client_count - in_scope_client - sff_client + in_scope_sff_client,
            "In-scope & SFF-eligible": in_scope_sff - in_scope_sff_client,
            "In-scope & Client": in_scope_client - in_scope_sff_client,
            "SFF-eligible & Client": sff_client - in_scope_sff_client,
            "In-scope & SFF-eligible & Client": in_scope_sff_client,
            "Total In-scope": in_scope_count,
            "Total SFF-eligible": sff_eligible_count,
            "Total Clients": client_count
        }
        
        logger.info(f"Venn diagram counts: {venn_counts}")
        # Log additional diagnostics for zero clients
        if client_count == 0:
            sds_data = load_and_process_sds_data()
            if sds_data is None:
                logger.warning("No SDS data loaded, resulting in zero clients")
            else:
                logger.info(f"SDS data rows: {len(sds_data)}, ClientFlag unique values: {sds_data['ClientFlag'].unique().tolist()}")
                logger.info(f"SDS cohouse values sample: {sds_data['cohouse'].head().tolist()}")
                logger.info(f"CompanyNumber sample: {cd['CompanyNumber'].head().to_list()}")
        
        return venn_counts
    except Exception as e:
        logger.error(f"Error computing Venn counts: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
    
# Data cleaning function
def clean_judgment_data(df):
    """Clean judgment data (current or archived) using Pandas."""
    # Find Provider-like column
    provider_col = None
    for col in df.columns:
        if "provider" in col.lower():
            provider_col = col
            break
    
    if provider_col is None:
        logger.error(f"No Provider-like column found in columns: {df.columns.tolist()}")
        return None
    
    logger.info(f"Using '{provider_col}' as Provider column")
    
    # Rename to standard 'Provider' for consistency
    df = df.rename(columns={provider_col: "Provider"})
    
    # Clean Provider column
    df["Provider"] = df["Provider"].astype(str).str.strip().str.replace(r"^0+", "", regex=True).str.strip()
    df["Provider_lower"] = df["Provider"].str.lower()
    
    # Filter out rows with empty or null Provider
    initial_rows = len(df)
    df = df[df["Provider"].notnull() & (df["Provider"] != "")]
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        logger.warning(f"Removed {removed_rows} rows with empty or null Provider values")
    
    if len(df) == 0:
        logger.error("No rows remain after filtering empty Provider values")
        return None
    
    df["Route"] = ""
    df["Explanation"] = ""
    
    filter_cols = [col for col in ["Gov", "Via"] if col in df.columns]
    if filter_cols:
        for col in filter_cols:
            df = df[~df[col].astype(str).str.contains(r"\*", na=False)]
    
    return df

# Data processing functions
def combine_judgments(current, archived):
    """Combine current and archived judgments using Pandas."""
    if current is None or archived is None:
        logger.error("Cannot combine judgments: one or both datasets are None")
        return None
    
    current["source"] = "current"
    archived["source"] = "archived"
    
    try:
        current["Publication_Date_Sorted"] = pd.to_datetime(current["Publication Date"])
        archived["Publication_Date_Sorted"] = pd.to_datetime(archived["Publication Date"])
        sort_date_col = "Publication_Date_Sorted"
    except Exception as e:
        logger.warning(f"Date conversion failed: {str(e)}. Using original column.")
        sort_date_col = "Publication Date"
    
    common_cols = list(set(current.columns).intersection(set(archived.columns)))
    required_cols = ["Provider", "Provider_lower", "source", sort_date_col]
    cols_to_select = list(set(common_cols).union(set(required_cols)))
    
    current_selected = current[[col for col in cols_to_select if col in current.columns]]
    archived_selected = archived[[col for col in cols_to_select if col in archived.columns]]
    
    combined = pd.concat([current_selected, archived_selected], ignore_index=True)
    logger.info(f"Combined judgments: {len(combined)} rows")
    return combined, sort_date_col

def deduplicate_judgments(combined, sort_date_col):
    """Deduplicate combined judgments using Pandas."""
    sort_columns = ["Provider_lower"]
    sort_ascending = [True]
    
    if sort_date_col in combined.columns:
        sort_columns.append(sort_date_col)
        sort_ascending.append(False)
    elif "Publication Date" in combined.columns:
        sort_columns.append("Publication Date")
        sort_ascending.append(False)
    
    sorted_combined = combined.sort_values(by=sort_columns, ascending=sort_ascending)
    deduplicated = sorted_combined.drop_duplicates(subset=["Provider_lower"], keep="first")
    
    if "source" in deduplicated.columns:
        deduplicated = deduplicated.drop(columns=["source"])
    if sort_date_col != "Publication Date" and sort_date_col in deduplicated.columns:
        deduplicated = deduplicated.drop(columns=[sort_date_col])
    
    logger.info(f"After deduplication: {len(deduplicated)} rows")
    return deduplicated

def prepare_judgment_data(deduplicated):
    """Prepare final judgment dataset with required schema using Pandas."""
    available_cols = [col for col in JD_COLUMNS if col in deduplicated.columns]
    missing_cols = [col for col in JD_COLUMNS if col not in deduplicated.columns]
    
    jd = deduplicated[available_cols].copy()
    for col in missing_cols:
        jd[col] = None
    
    jd = jd[JD_COLUMNS]
    
    if "Reg Code" in jd.columns:
        jd["Reg Code"] = jd["Reg Code"].astype(str).replace("nan", None)
    if "Gov" in jd.columns:
        jd["Gov"] = jd["Gov"].astype(str).str.strip().replace("nan", None)
    if "Via" in jd.columns:
        jd["Via"] = jd["Via"].astype(str).str.strip().replace("nan", None)
    
    if "Provider_lower" not in jd.columns:
        jd["Provider_lower"] = jd["Provider"].str.lower()
    
    logger.info(f"Final judgment data: {len(jd)} rows")
    return jd

def create_final_dataset(cd, jd, sds_data=None, cre_data=None):
    """Create final dataset by joining Polars company data with Pandas judgment, SDS, and CRE data."""
    if cd is None or jd is None:
        logger.error("Cannot create final dataset: one or both datasets are None")
        return None
    
    # Convert Polars DataFrame to Pandas
    cd_pandas = cd.to_pandas()
    
    # Join with judgment data
    final_data = pd.merge(
        cd_pandas,
        jd,
        left_on="CompanyName_lower",
        right_on="Provider_lower",
        how="left"
    )
    
    # Ensure Reg Code is string
    if "Reg Code" in final_data.columns:
        final_data["Reg Code"] = final_data["Reg Code"].astype(str).replace("nan", None)
    
    # Add IsRegistered column
    final_data["IsRegistered"] = final_data["Reg Code"].notnull()
    
    # Join with SDS data if available
    if sds_data is not None:
        final_data = pd.merge(
            final_data,
            sds_data[["name", "CpartyId", "cohouse", "cusid", "ClientFlag"]],
            left_on="CompanyNumber",
            right_on="cohouse",
            how="left"
        )
        # Set ClientFlag to False for unmatched records and ensure boolean dtype
        final_data["ClientFlag"] = final_data["ClientFlag"].fillna(False).astype(bool)
        # Drop cohouse column to avoid redundancy
        final_data = final_data.drop(columns=["cohouse"], errors="ignore")
        logger.info(f"Joined final dataset with SDS data, resulting in {len(final_data)} rows")
    
    # Join with CRE data if available
    if cre_data is not None:
        final_data = pd.merge(
            final_data,
            cre_data[["counterparty_id", "facility_id", "drawn_balance", "facility_limit", "undrawn_balance"]],
            left_on="CpartyId",
            right_on="counterparty_id",
            how="left"
        )
        # Drop counterparty_id column to avoid redundancy
        final_data = final_data.drop(columns=["counterparty_id"], errors="ignore")
        logger.info(f"Joined final dataset with CRE data, resulting in {len(final_data)} rows")
    
    logger.info(f"Final dataset: {len(final_data)} rows")
    logger.info(f"Rows with registration: {len(final_data[final_data['IsRegistered']])}")
    logger.info(f"Rows with judgments: {len(final_data[final_data['Status'].notnull()])}")
    logger.info(f"Rows with ClientFlag True: {len(final_data[final_data['ClientFlag'] == True])}")
    return final_data

# Main data loading and processing function
@st.cache_data
def load_and_process_data():
    """Orchestrate data loading and processing."""
    cd = load_company_data()
    rj_current = load_current_judgments()
    rj_archived = load_archived_judgments()
    sds_data = load_and_process_sds_data()
    cre_data = load_cre_social_housing_data()
    
    if any(data is None for data in [cd, rj_current, rj_archived]):
        st.error("Data loading failed. Check logs for details.")
        return None, None, None
    
    combined, sort_date_col = combine_judgments(rj_current, rj_archived)
    if combined is None:
        return None, None, None
    
    deduplicated = deduplicate_judgments(combined, sort_date_col)
    jd = prepare_judgment_data(deduplicated)
    final_data = create_final_dataset(cd, jd, sds_data, cre_data)
    
    return cd, jd, final_data

# UI and Visualization functions
def display_paginated_df(df, title, key):
    total_rows = len(df)
    total_pages = (total_rows + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE
    page = st.number_input(
        f"Page (1-{total_pages})",
        min_value=1,
        max_value=max(1, total_pages),
        value=1,
        key=f"{key}_page"
    )
    start_idx = (page - 1) * ROWS_PER_PAGE
    end_idx = min(start_idx + ROWS_PER_PAGE, total_rows)
    
    # Convert problematic columns to string to avoid Arrow serialization issues
    display_df = df.iloc[start_idx:end_idx].copy()
    for col in display_df.columns:
        if display_df[col].dtype == "object" or pd.api.types.is_string_dtype(display_df[col]):
            display_df[col] = display_df[col].astype(str).replace("nan", None)
    
    st.dataframe(display_df, use_container_width=True)
    st.write(f"Showing rows {start_idx + 1} to {end_idx} of {total_rows}")
    csv = df.to_csv(index=False)
    st.download_button(
        label=f"Download {title} Data",
        data=csv,
        file_name=f"{title.lower().replace(' ', '_')}_data.csv",
        mime="text/csv",
        key=f"{key}_download"
    )

# Main application
st.subheader("Social Housing Analysis")
st.divider()

cd, jd, final_data = load_and_process_data()
if final_data is None:
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
search_term = st.sidebar.text_input("Search Company/Provider", "")
status_filter = st.sidebar.multiselect(
    "Company Status",
    options=final_data["CompanyStatus"].unique().tolist(),
    default=["Active"]
)
reg_filter = st.sidebar.selectbox(
    "Registration Status",
    ["All", "Registered", "Unregistered"],
    index=0
)
gov_filter = st.sidebar.multiselect(
    "Governance Rating",
    options=final_data["Gov"].dropna().unique().tolist(),
    default=[]
)
via_filter = st.sidebar.multiselect(
    "Viability Rating",
    options=final_data["Via"].dropna().unique().tolist(),
    default=[]
)

# Apply filters
filtered_data = final_data
if search_term:
    filtered_data = filtered_data[
        filtered_data["CompanyName"].str.lower().str.contains(search_term.lower(), na=False) |
        filtered_data["Provider"].str.lower().str.contains(search_term.lower(), na=False)
    ]
if status_filter:
    filtered_data = filtered_data[filtered_data["CompanyStatus"].isin(status_filter)]
if reg_filter != "All":
    filtered_data = filtered_data[filtered_data["IsRegistered"] == (reg_filter == "Registered")]
if gov_filter:
    filtered_data = filtered_data[filtered_data["Gov"].isin(gov_filter)]
if via_filter:
    filtered_data = filtered_data[filtered_data["Via"].isin(via_filter)]

# Main dashboard
st.title("UK Social Housing - Sustainable Finance Framework Dashboard")

# Display datasets
ROWS_PER_PAGE = 1000
with st.expander("Company House Data (CD)", expanded=False):
    display_paginated_df(cd.to_pandas(), "Company House", "cd")
with st.expander("Judgement Data (JD)", expanded=False):
    display_paginated_df(jd, "Judgement", "jd")
with st.expander("Final Dataset", expanded=False):
    display_paginated_df(filtered_data, "Final", "final")

# Visualizations
st.subheader("Visual Analytics")
st.write(f"Total rows in filtered_data: {len(filtered_data)}")

reg_count = len(filtered_data[filtered_data["IsRegistered"]])
unreg_count = len(filtered_data[~filtered_data["IsRegistered"]])
client_count = len(filtered_data[filtered_data["ClientFlag"] == True])
non_client_count = len(filtered_data[filtered_data["ClientFlag"] == False])
col1, col2, col3 = st.columns(3)

with col1:
    try:
        fig1 = go.Figure(data=[go.Pie(
            labels=['Registered', 'Unregistered'],
            values=[reg_count, unreg_count],
            hole=0.3,
            marker_colors=['rgb(65, 105, 225)', 'rgb(135, 206, 250)']
        )])
        fig1.update_layout(title="Registered vs Unregistered Providers", height=400)
        st.plotly_chart(fig1, use_container_width=True)
        st.write(f"Registered: {reg_count}, Unregistered: {unreg_count}")
    except Exception as e:
        st.error(f"Error creating registration chart: {str(e)}")
        st.write(f"Registered: {reg_count}, Unregistered: {unreg_count}")

with col2:
    judgment_count = len(filtered_data[filtered_data["Status"].notnull()])
    no_judgment_count = len(filtered_data[filtered_data["Status"].isnull()])
    try:
        fig2 = go.Figure(data=[go.Bar(
            x=['No Judgement', 'With Judgement'],
            y=[no_judgment_count, judgment_count],
            marker_color=['rgb(102, 178, 255)', 'rgb(65, 105, 225)']
        )])
        fig2.update_layout(
            title="Providers by Judgement Status",
            yaxis=dict(title="Count", range=[0, max(no_judgment_count, judgment_count) * 1.1]),
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.write(f"No Judgement: {no_judgment_count}, With Judgement: {judgment_count}")
    except Exception as e:
        st.error(f"Error creating judgement chart: {str(e)}")
        st.write(f"No Judgement: {no_judgment_count}, With Judgement: {judgment_count}")

with col3:
    try:
        fig3 = go.Figure(data=[go.Pie(
            labels=['Clients', 'Non-Clients'],
            values=[client_count, non_client_count],
            hole=0.3,
            marker_colors=['rgb(34, 139, 34)', 'rgb(169, 169, 169)']
        )])
        fig3.update_layout(title="Clients vs Non-Clients", height=400)
        st.plotly_chart(fig3, use_container_width=True)
        st.write(f"Clients: {client_count}, Non-Clients: {non_client_count}")
    except Exception as e:
        st.error(f"Error creating client chart: {str(e)}")
        st.write(f"Clients: {client_count}, Non-Clients: {non_client_count}")

col4, col5 = st.columns(2)

with col4:
    gov_data = filtered_data[filtered_data["Gov"].notnull()]
    gov_counts = gov_data["Gov"].value_counts().reset_index()
    gov_counts.columns = ["Gov", "Count"]
    try:
        if len(gov_counts) > 0:
            fig4 = go.Figure(data=[go.Bar(
                x=gov_counts["Gov"].tolist(),
                y=gov_counts["Count"].tolist(),
                marker_color='rgb(65, 105, 225)'
            )])
            fig4.update_layout(
                title="Governance Ratings Distribution",
                yaxis=dict(title="Count", range=[0, gov_counts["Count"].max() * 1.1]),
                height=400
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.write("No governance ratings data available")
    except Exception as e:
        st.error(f"Error creating governance chart: {str(e)}")
        if len(gov_counts) > 0:
            for _, row in gov_counts.iterrows():
                st.write(f"{row['Gov']}: {row['Count']}")
    no_rating_count = len(filtered_data[filtered_data["Gov"].isnull()])
    st.metric("Providers with No Governance Rating", no_rating_count)

with col5:
    via_data = filtered_data[filtered_data["Via"].notnull()]
    via_counts = via_data["Via"].value_counts().reset_index()
    via_counts.columns = ["Via", "Count"]
    try:
        if len(via_counts) > 0:
            fig5 = go.Figure(data=[go.Bar(
                x=via_counts["Via"].tolist(),
                y=via_counts["Count"].tolist(),
                marker_color='rgb(0, 128, 128)'
            )])
            fig5.update_layout(
                title="Viability Ratings Distribution",
                yaxis=dict(title="Count", range=[0, via_counts["Count"].max() * 1.1]),
                height=400
            )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.write("No viability ratings data available")
    except Exception as e:
        st.error(f"Error creating viability chart: {str(e)}")
        if len(via_counts) > 0:
            for _, row in via_counts.iterrows():
                st.write(f"{row['Via']}: {row['Count']}")
    no_rating_count = len(filtered_data[filtered_data["Via"].isnull()])
    st.metric("Providers with No Viability Rating", no_rating_count)

col6, col7 = st.columns(2)

with col6:
    try:
        if "drawn_balance" in filtered_data.columns:
            total_drawn = filtered_data["drawn_balance"].sum()
            total_undrawn = filtered_data["undrawn_balance"].sum()
            total_limit = filtered_data["facility_limit"].sum()
            fig6 = go.Figure(data=[go.Bar(
                x=['Drawn Balance', 'Undrawn Balance', 'Facility Limit'],
                y=[total_drawn, total_undrawn, total_limit],
                marker_color=['rgb(255, 99, 71)', 'rgb(60, 179, 113)', 'rgb(70, 130, 180)']
            )])
            fig6.update_layout(
                title="Total Loan Balances",
                yaxis=dict(title="Amount (£)", range=[0, max(total_drawn, total_undrawn, total_limit) * 1.1]),
                height=400
            )
            st.plotly_chart(fig6, use_container_width=True)
            st.write(f"Drawn: £{total_drawn:,.2f}, Undrawn: £{total_undrawn:,.2f}, Limit: £{total_limit:,.2f}")
        else:
            st.write("No CRE loan data available")
    except Exception as e:
        st.error(f"Error creating loan balances chart: {str(e)}")
        if "drawn_balance" in filtered_data.columns:
            st.write(f"Drawn: £{filtered_data['drawn_balance'].sum():,.2f}, "
                     f"Undrawn: £{filtered_data['undrawn_balance'].sum():,.2f}, "
                     f"Limit: £{filtered_data['facility_limit'].sum():,.2f}")

with col7:
    try:
        if "drawn_balance" in filtered_data.columns:
            valid_data = filtered_data[filtered_data["drawn_balance"].notnull() & filtered_data["undrawn_balance"].notnull()]
            total_drawn = valid_data["drawn_balance"].sum()
            total_undrawn = valid_data["undrawn_balance"].sum()
            if total_drawn + total_undrawn > 0:
                fig7 = go.Figure(data=[go.Pie(
                    labels=['Drawn Balance', 'Undrawn Balance'],
                    values=[total_drawn, total_undrawn],
                    hole=0.3,
                    marker_colors=['rgb(255, 99, 71)', 'rgb(60, 179, 113)']
                )])
                fig7.update_layout(title="Drawn vs Undrawn Balance", height=400)
                st.plotly_chart(fig7, use_container_width=True)
                st.write(f"Drawn: £{total_drawn:,.2f}, Undrawn: £{total_undrawn:,.2f}")
            else:
                st.write("No valid drawn or undrawn balance data available")
        else:
            st.write("No CRE loan data available")
    except Exception as e:
        st.error(f"Error creating drawn vs undrawn chart: {str(e)}")
        if "drawn_balance" in filtered_data.columns:
            valid_data = filtered_data[filtered_data["drawn_balance"].notnull() & filtered_data["undrawn_balance"].notnull()]
            st.write(f"Drawn: £{valid_data['drawn_balance'].sum():,.2f}, "
                     f"Undrawn: £{valid_data['undrawn_balance'].sum():,.2f}")

# Venn Diagram
st.subheader("Venn Diagram: In-scope, SFF-eligible, and Clients")
try:
    venn_counts = compute_venn_counts(cd, filtered_data)
    if venn_counts is not None and venn_counts["Total In-scope"] > 0:
        # Create a Plotly figure for the Venn diagram
        fig_venn = go.Figure()
        
        # Define circle data, excluding Clients if count is 0
        circle_data = [
            {"x": 0.4, "y": 0.6, "size": venn_counts["Total In-scope"], "color": "rgba(65, 105, 225, 0.4)", "label": f"In-scope\n{venn_counts['Total In-scope']}"},
            {"x": 0.6, "y": 0.6, "size": venn_counts["Total SFF-eligible"], "color": "rgba(255, 99, 71, 0.4)", "label": f"SFF-eligible\n{venn_counts['Total SFF-eligible']}"}
        ]
        if venn_counts["Total Clients"] > 0:
            circle_data.append(
                {"x": 0.5, "y": 0.4, "size": venn_counts["Total Clients"], "color": "rgba(34, 139, 34, 0.4)", "label": f"Clients\n{venn_counts['Total Clients']}"}
            )
        
        # Adjust scaling to ensure visibility of small counts
        max_size = max([d["size"] for d in circle_data])
        min_radius = 0.05  # Minimum radius for visibility
        for circle in circle_data:
            # Use logarithmic scaling for radius to handle large disparities
            size = max(circle["size"], 1)  # Avoid log(0)
            radius = min_radius + 0.1 * (size / max_size) ** 0.5
            fig_venn.add_shape(
                type="circle",
                xref="paper", yref="paper",
                x0=circle["x"] - radius, y0=circle["y"] - radius,
                x1=circle["x"] + radius, y1=circle["y"] + radius,
                fillcolor=circle["color"],
                line_color="black",
                opacity=0.4
            )
            fig_venn.add_annotation(
                x=circle["x"], y=circle["y"],
                text=circle["label"],
                showarrow=False,
                font=dict(size=12)
            )
        
        # Add intersection labels based on available circles
        if venn_counts["Total SFF-eligible"] > 0:
            fig_venn.add_annotation(
                x=0.5, y=0.55,  # Center of In-scope & SFF-eligible overlap
                text=f"In-scope & SFF: {venn_counts['In-scope & SFF-eligible']}",
                showarrow=False,
                font=dict(size=10)
            )
        if venn_counts["Total Clients"] > 0:
            fig_venn.add_annotation(
                x=0.45, y=0.45,  # Center of In-scope & Client overlap
                text=f"In-scope & Client: {venn_counts['In-scope & Client']}",
                showarrow=False,
                font=dict(size=10)
            )
            fig_venn.add_annotation(
                x=0.55, y=0.45,  # Center of SFF-eligible & Client overlap
                text=f"SFF & Client: {venn_counts['SFF-eligible & Client']}",
                showarrow=False,
                font=dict(size=10)
            )
            fig_venn.add_annotation(
                x=0.5, y=0.5,  # Center of all three
                text=f"All: {venn_counts['In-scope & SFF-eligible & Client']}",
                showarrow=False,
                font=dict(size=10)
            )
        
        fig_venn.update_layout(
            title="Venn Diagram: In-scope Companies, SFF-eligible, Clients",
            showlegend=False,
            width=600,
            height=500,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        st.plotly_chart(fig_venn, use_container_width=True)
        
        # Display note if Clients circle is omitted
        if venn_counts["Total Clients"] == 0:
            st.write("Note: Clients circle omitted due to zero client count.")
    else:
        st.write("Unable to render Venn diagram: No in-scope data or counts unavailable")
except Exception as e:
    st.error(f"Error creating Venn diagram: {str(e)}")
    st.write("Venn diagram data unavailable")

# Key Statistics
st.subheader("Key Statistics")
col5, col6, col7, col8, col9, col10 = st.columns(6)

with col5:
    st.metric("Total Providers", len(filtered_data))
with col6:
    st.metric("Registered Providers", len(filtered_data[filtered_data["IsRegistered"]]))
with col7:
    st.metric("Unregistered Providers", len(filtered_data[~filtered_data["IsRegistered"]]))
with col8:
    st.metric("Providers with Judgements", len(filtered_data[filtered_data["Status"].notnull()]))
with col9:
    st.metric("Providers without Judgements", len(filtered_data[filtered_data["Status"].isnull()]))
with col10:
    eligible = len(filtered_data[
        (filtered_data["IsRegistered"]) &
        (filtered_data["CompanyStatus"] == "Active") &
        (
            (filtered_data["Con"].isin(["C1", "C2", "C3", "C4"])) |
            (filtered_data["Gov"].isin(["G1", "G2", "G3", "G4"])) |
            (filtered_data["Via"].isin(["V1", "V2", "V3", "V4"]))
        )
    ])
    st.metric("SFF Eligible Providers", eligible)

# Downloadable Datasets
st.subheader("Downloadable Datasets")

with st.expander("Registered Providers Dataset", expanded=False):
    registered_df = filtered_data[filtered_data["IsRegistered"]]
    display_paginated_df(registered_df, "Registered Providers", "registered")

with st.expander("Unregistered Providers Dataset", expanded=False):
    unregistered_df = filtered_data[~filtered_data["IsRegistered"]]
    display_paginated_df(unregistered_df, "Unregistered Providers", "unregistered")

with st.expander("Providers with Judgements Dataset", expanded=False):
    with_judgement_df = filtered_data[filtered_data["Status"].notnull()]
    display_paginated_df(with_judgement_df, "Providers with Judgements", "with_judgement")

with st.expander("Providers without Judgements Dataset", expanded=False):
    without_judgement_df = filtered_data[filtered_data["Status"].isnull()]
    display_paginated_df(without_judgement_df, "Providers without Judgements", "without_judgement")

with st.expander("SFF Eligible Providers Dataset", expanded=False):
    eligible_df = filtered_data[
        (filtered_data["IsRegistered"]) &
        (filtered_data["CompanyStatus"] == "Active") &
        (
            (filtered_data["Con"].isin(["C1", "C2", "C3", "C4"])) |
            (filtered_data["Gov"].isin(["G1", "G2", "G3", "G4"])) |
            (filtered_data["Via"].isin(["V1", "V2", "V3", "V4"]))
        )
    ]
    display_paginated_df(eligible_df, "SFF Eligible Providers", "eligible")

# SFF Eligibility Criteria
st.subheader("Sustainable Finance Framework Eligibility Criteria")
st.markdown("""
Providers are typically SFF eligible if they:
- Are registered with the Regulator of Social Housing
- Have active company status
- Meet Con standards (C1 or C2 or C3 or C4 ratings)
- Meet governance standards (G1 or G2 or G3 or G4 ratings)
- Meet viability standards (V1 or V2 or V3 or V4 ratings)
""")
