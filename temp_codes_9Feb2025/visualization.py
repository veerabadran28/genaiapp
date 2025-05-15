import streamlit as st
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict

def plot_venn_diagram(stats: Dict[str, int]) -> None:
    """
    Render a Venn diagram based on the calculated statistics.
    
    Args:
        stats: Dictionary containing Venn diagram statistics
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create Venn diagram
    v = venn2(subsets=(
        stats['pure_play_count'] - stats['pure_play_in_sff'],
        stats['sff_data_count'] - stats['pure_play_in_sff'],
        stats['pure_play_in_sff']
    ), set_labels=('LLM Generated (>=50%)', 'SFF Data'))
    
    # Customize the diagram
    if v.get_label_by_id('10'):
        v.get_label_by_id('10').set_text(f"Pure Play\nNot in SFF\n{stats['pure_play_not_in_sff']}")
    if v.get_label_by_id('01'):
        v.get_label_by_id('01').set_text(f"SFF\nNot Pure Play\n{stats['sff_not_in_pure_play']}")
    if v.get_label_by_id('11'):
        v.get_label_by_id('11').set_text(f"Both\n{stats['pure_play_in_sff']}")
    
    plt.title("Comparison of Pure Play Companies (>=50%) vs SFF Data")
    st.pyplot(fig)

def plot_green_revenue_distribution(green_revenue: pd.DataFrame) -> None:
    """
    Plot distribution of green revenue percentages.
    
    Args:
        green_revenue: GREEN_REVENUE dataset
    """
    fig = px.histogram(
        green_revenue,
        x='greenRevenuePercent',
        nbins=20,
        title='Distribution of Green Revenue Percentages',
        labels={'greenRevenuePercent': 'Green Revenue Percentage'},
        color='pure_play_flag',
        color_discrete_map={'Y': 'green', 'N': 'gray'}
    )
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)

def plot_country_analysis(green_revenue: pd.DataFrame) -> None:
    """
    Plot analysis of green revenue by country with proper country code handling.
    
    Args:
        green_revenue: GREEN_REVENUE dataset with country_code column
    """
    # Ensure we have country codes and they're clean
    if 'country_code' not in green_revenue.columns:
        st.warning("Country code data not available")
        return
    
    # Prepare country stats
    country_stats = (
        green_revenue
        .dropna(subset=['country_code'])
        .assign(country_code=lambda x: x['country_code'].astype(str).str.strip().str.upper())
        .groupby('country_code')
        .agg(
            company_count=('counterparty_id', 'nunique'),
            avg_green_revenue=('greenRevenuePercent', 'mean'),
            pure_play_count=('pure_play_flag', lambda x: (x == 'Y').sum())
        )
        .reset_index()
        .query("country_code != 'NAN'")  # Remove any invalid country codes
    )
    
    if country_stats.empty:
        st.warning("No valid country data available for mapping")
        return
    
    # Create the choropleth map
    fig = px.choropleth(
        country_stats,
        locations='country_code',
        locationmode='ISO-3',  # Use 3-letter country codes
        color='avg_green_revenue',
        hover_name='country_code',
        hover_data={
            'company_count': True,
            'pure_play_count': True,
            'avg_green_revenue': ':.1f',
            'country_code': False  # Hide from hover as it's in the name
        },
        title='Average Green Revenue Percentage by Country',
        color_continuous_scale='Greens',
        projection='natural earth'
    )
    
    # Customize the layout
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        coloraxis_colorbar={
            'title': 'Avg Green %',
            'ticksuffix': '%'
        }
    )
    
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)
