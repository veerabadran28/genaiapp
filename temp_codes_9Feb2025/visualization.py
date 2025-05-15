import streamlit as st
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict
import pandas as pd

def plot_three_circle_venn(stats: Dict[str, int]) -> None:
    """Render a 3-circle Venn diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.style.use('seaborn')
    
    subsets = (
        stats['pure_play_count'] - stats['pure_play_in_sff'],
        stats['sff_data_count'] - stats['pure_play_in_sff'] - stats['not_pure_but_30_in_sff'],
        stats['pure_play_in_sff'],
        stats['not_pure_but_30_count'] - stats['not_pure_but_30_in_sff'],
        0,
        stats['not_pure_but_30_in_sff']
    )
    
    v = venn3(
        subsets=subsets,
        set_labels=('LLM Generated\n(>=50%)', 'SFF Data', 'LLM Generated\n(>=30%)'),
        set_colors=('#4CAF50', '#2196F3', '#FFC107'),
        alpha=0.7
    )
    
    # Customize labels
    label_map = {
        '100': f"Pure Play\nNot in SFF\n{stats['pure_play_not_in_sff']:,}",
        '010': f"SFF Only\n{stats['sff_not_in_pure_play']:,}",
        '001': f"30-49%\nNot in SFF\n{stats['not_pure_but_30_count'] - stats['not_pure_but_30_in_sff']:,}",
        '110': f"Pure Play\nin SFF\n{stats['pure_play_in_sff']:,}",
        '011': f"30-49%\nin SFF\n{stats['not_pure_but_30_in_sff']:,}"
    }
    
    for label_id, text in label_map.items():
        lbl = v.get_label_by_id(label_id)
        if lbl: lbl.set_text(text)
    
    plt.title("Sustainable Finance Company Classification")
    st.pyplot(fig)

def plot_green_revenue_distribution(green_revenue: pd.DataFrame) -> None:
    """Plot distribution of green revenue percentages."""
    fig = px.histogram(
        green_revenue,
        x='greenRevenuePercent',
        nbins=20,
        title='Green Revenue Distribution',
        color='pure_play_flag',
        color_discrete_map={'Y': 'green', 'N': 'gray'}
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_country_analysis(green_revenue: pd.DataFrame) -> None:
    """Plot green revenue by country."""
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
        color_continuous_scale='Greens',
        projection='natural earth'
    )
    st.plotly_chart(fig, use_container_width=True)
