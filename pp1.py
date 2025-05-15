import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder
from pathlib import Path

# --- Constants ---
DATA_PATH = Path("data")
PCAF_FILE = DATA_PATH / "group_client_coverage_dec24.xlsx"
LLM_FILE = DATA_PATH / "llm_generated.xlsx"
SFF_FILE = DATA_PATH / "Mar PP list_cF.xlsx"

# --- Data Processing Module ---
def load_and_process_data():
    """
    Load and process PCAF_DATA, LLM_GENERATED, and SFF_DATA datasets,
    then create the GREEN_REVENUE dataset.
    """
    try:
        # Load PCAF_DATA
        pca_data = pd.read_excel(PCAF_FILE)
        # Select unique records based on specified columns
        pca_data = pca_data[[
            'cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name',
            'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code'
        ]].drop_duplicates()

        # Load LLM_GENERATED
        llm_data = pd.read_excel(LLM_FILE)

        # Join datasets on counterparty_name and companyName (case-insensitive, trimmed)
        pca_data['join_key'] = pca_data['counterparty_name'].str.lower().str.strip()
        llm_data['join_key'] = llm_data['companyName'].str.lower().str.strip()
        green_revenue = pd.merge(
            pca_data,
            llm_data,
            on='join_key',
            how='inner'
        )

        # Select required columns for GREEN_REVENUE dataset
        green_revenue = green_revenue[[
            'cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name',
            'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code', 'year',
            'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources'
        ]]

        # Add pure_play_flag: "Y" if greenRevenuePercent >= 50, else "N"
        green_revenue['pure_play_flag'] = green_revenue['greenRevenuePercent'].apply(
            lambda x: 'Y' if x >= 50 else 'N'
        )

        # Load SFF_DATA
        sff_data = pd.read_excel(SFF_FILE)

        return green_revenue, sff_data

    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return None, None

# --- Visualization Module ---
def create_venn_diagram(green_revenue, sff_data):
    """
    Create a Venn diagram to represent the overlap between datasets as per the attached image.
    """
    # Compute counts for the Venn diagram
    llm_gen_count = len(green_revenue)
    sff_count = len(sff_data)
    
    # Overlap: Companies in both GREEN_REVENUE and SFF_DATA (based on counterparty_name and Client Name)
    green_revenue['join_key'] = green_revenue['counterparty_name'].str.lower().str.strip()
    sff_data['join_key'] = sff_data['Client Name'].str.lower().str.strip()
    overlap = pd.merge(green_revenue, sff_data, on='join_key', how='inner')
    overlap_count = len(overlap)

    # Companies in LLM_GENERATED but not in SFF_DATA
    llm_not_sff = green_revenue[~green_revenue['join_key'].isin(sff_data['join_key'])]
    llm_not_sff_count = len(llm_not_sff)

    # Companies in SFF_DATA but not in LLM_GENERATED
    sff_not_llm = sff_data[~sff_data['join_key'].isin(green_revenue['join_key'])]
    sff_not_llm_count = len(sff_not_llm)

    # Companies in LLM_GENERATED with pure_play_flag="Y"
    llm_pure_play = green_revenue[green_revenue['pure_play_flag'] == 'Y']
    llm_pure_play_count = len(llm_pure_play)

    # Create Venn diagram using Plotly
    fig = go.Figure()

    # Circle for LLM_GENERATED
    fig.add_shape(type="circle", xref="x", yref="y", x0=0, y0=0, x1=4, y1=4, fillcolor="blue", opacity=0.3)
    fig.add_annotation(x=2, y=4.5, text=f"LLM-Generated\n{llm_gen_count}", showarrow=False)

    # Circle for SFF_DATA
    fig.add_shape(type="circle", xref="x", yref="y", x0=2, y0=0, x1=6, y1=4, fillcolor="green", opacity=0.3)
    fig.add_annotation(x=4, y=4.5, text=f"SFF Data\n{sff_count}", showarrow=False)

    # Circle for Pure Play (greenRevenuePercent >= 50)
    fig.add_shape(type="circle", xref="x", yref="y", x0=1, y0=-2, x1=5, y1=2, fillcolor="red", opacity=0.3)
    fig.add_annotation(x=3, y=-2.5, text=f"Pure Play\n{llm_pure_play_count}", showarrow=False)

    # Add overlap annotations
    fig.add_annotation(x=3, y=2, text=f"Overlap\n{overlap_count}", showarrow=False)
    fig.add_annotation(x=1.5, y=2, text=f"LLM but not SFF\n{llm_not_sff_count}", showarrow=False)
    fig.add_annotation(x=4.5, y=2, text=f"SFF but not LLM\n{sff_not_llm_count}", showarrow=False)

    fig.update_layout(title="Venn Diagram: LLM-Generated, SFF Data, and Pure Play Companies",
                      xaxis=dict(show=False), yaxis=dict(show=False))
    return fig

def create_additional_charts(green_revenue):
    """
    Create additional charts for metrics and statistics.
    """
    # Chart 1: Green Revenue Percentage Distribution
    fig1 = px.histogram(green_revenue, x='greenRevenuePercent', nbins=20,
                        title="Distribution of Green Revenue Percentage",
                        labels={'greenRevenuePercent': 'Green Revenue %'})
    
    # Chart 2: Total Revenue by Country
    fig2 = px.bar(green_revenue.groupby('country_code')['totalRevenue'].sum().reset_index(),
                  x='country_code', y='totalRevenue',
                  title="Total Revenue by Country",
                  labels={'totalRevenue': 'Total Revenue ($)', 'country_code': 'Country'})
    
    # Chart 3: Scatter plot of Total Revenue vs Green Revenue Percentage
    fig3 = px.scatter(green_revenue, x='totalRevenue', y='greenRevenuePercent', color='country_code',
                      title="Total Revenue vs Green Revenue Percentage by Country",
                      labels={'totalRevenue': 'Total Revenue ($)', 'greenRevenuePercent': 'Green Revenue %'})
    
    return fig1, fig2, fig3

# --- Table Display Module ---
def display_table(data, attributes):
    """
    Display a table using streamlit-aggrid.
    """
    if data.empty:
        st.write("No data available.")
        return
    gb = GridOptionsBuilder.from_dataframe(data[attributes])
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    grid_options = gb.build()
    AgGrid(data[attributes], gridOptions=grid_options, height=400, fit_columns_on_grid_load=True)

# --- Main Dashboard ---
def main():
    st.set_page_config(page_title="Sustainable Data Dashboard", layout="wide")
    st.title("Sustainable Data Dashboard")

    # Load and process data
    green_revenue, sff_data = load_and_process_data()
    
    if green_revenue is None or sff_data is None:
        st.stop()

    # Section 1: Metrics and Statistics
    st.header("Metrics and Statistics")
    
    # Venn Diagram
    venn_fig = create_venn_diagram(green_revenue, sff_data)
    st.plotly_chart(venn_fig, use_container_width=True)

    # Additional Charts
    fig1, fig2, fig3 = create_additional_charts(green_revenue)
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)

    # Section 2: Pure Play Companies (>=50%)
    st.header("Companies that are classified as pure play in GREEN REVENUE (>=50%)")
    pure_play = green_revenue[green_revenue['pure_play_flag'] == 'Y']
    
    # Compute overlap and differences
    overlap = pd.merge(pure_play, sff_data, on='join_key', how='inner')
    sff_not_pure = sff_data[~sff_data['join_key'].isin(pure_play['join_key'])]
    pure_not_sff = pure_play[~pure_play['join_key'].isin(sff_data['join_key'])]

    # Display counts
    st.write(f"Total Pure Play Companies: {len(pure_play)}")
    st.write(f"Overlap Count: {len(overlap)}")
    st.write(f"Identified in SFF but not in Pure Play: {len(sff_not_pure)}")
    st.write(f"Un-Identified (Pure Play but not in SFF): {len(pure_not_sff)}")

    # Tabs for Pure Play
    tab1, tab2, tab3 = st.tabs(["Overlap", "Identified Clients", "Un-Identified Clients"])
    
    with tab1:
        st.subheader("The companies that exists in both GREEN_REVENUE and SFF_DATA")
        display_table(overlap, ['cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name',
                                'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code', 'year',
                                'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag'])
    
    with tab2:
        st.subheader("The companies that exists in SFF_DATA but not exists in GREEN_REVENUE")
        display_table(sff_not_pure, ['Pureplay Status', 'SDS', 'Alt SDS', 'Client Name', 'Themes',
                                     'Sub Theme', 'TLN', 'SLN', 'CSID', 'additional CSID', 'BIC'])
    
    with tab3:
        st.subheader("The companies that exists in GREEN_REVENUE but not exists in SFF_DATA")
        display_table(pure_not_sff, ['cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name',
                                     'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code', 'year',
                                     'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag'])

    # Section 3: Non-Pure Play Companies (<50%)
    st.header("Companies that are not classified as pure play in GREEN REVENUE (<50%)")
    non_pure_play = green_revenue[green_revenue['pure_play_flag'] == 'N']
    
    # Compute overlap and differences
    overlap_non_pure = pd.merge(non_pure_play, sff_data, on='join_key', how='inner')
    sff_not_non_pure = sff_data[~sff_data['join_key'].isin(non_pure_play['join_key'])]
    non_pure_not_sff = non_pure_play[~non_pure_play['join_key'].isin(sff_data['join_key'])]

    # Display counts
    st.write(f"Total Non-Pure Play Companies: {len(non_pure_play)}")
    st.write(f"Overlap Count: {len(overlap_non_pure)}")
    st.write(f"Identified in SFF but not in Non-Pure Play: {len(sff_not_non_pure)}")
    st.write(f"Un-Identified (Non-Pure Play but not in SFF): {len(non_pure_not_sff)}")

    # Tabs for Non-Pure Play
    tab4, tab5, tab6 = st.tabs(["Overlap", "Identified Clients", "Un-Identified Clients"])
    
    with tab4:
        st.subheader("The companies that exists in both GREEN_REVENUE and SFF_DATA")
        display_table(overlap_non_pure, ['cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name',
                                        'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code', 'year',
                                        'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag'])
    
    with tab5:
        st.subheader("The companies that exists in SFF_DATA but not exists in GREEN_REVENUE")
        display_table(sff_not_non_pure, ['Pureplay Status', 'SDS', 'Alt SDS', 'Client Name', 'Themes',
                                         'Sub Theme', 'TLN', 'SLN', 'CSID', 'additional CSID', 'BIC'])
    
    with tab6:
        st.subheader("The companies that exists in GREEN_REVENUE but not exists in SFF_DATA")
        display_table(non_pure_not_sff, ['cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name',
                                        'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code', 'year',
                                        'totalRevenue', 'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag'])

if __name__ == "__main__":
    main()
