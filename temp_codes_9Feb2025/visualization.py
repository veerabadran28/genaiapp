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
    Plot analysis of green revenue by country.
    
    Args:
        green_revenue: GREEN_REVENUE dataset
    """
    country_stats = green_revenue.groupby('country_code').agg(
        company_count=('counterparty_id', 'nunique'),
        avg_green_revenue=('greenRevenuePercent', 'mean')
    ).reset_index()
    
    fig = px.choropleth(
        country_stats,
        locations='country_code',
        color='avg_green_revenue',
        hover_name='country_code',
        hover_data=['company_count'],
        title='Average Green Revenue Percentage by Country',
        color_continuous_scale='Greens'
    )
    st.plotly_chart(fig, use_container_width=True)
