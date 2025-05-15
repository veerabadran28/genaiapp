import streamlit as st
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict
import pandas as pd
import numpy as np
import pydeck as pdk

# Set a modern style that works without seaborn
plt.style.use('default')

def plot_three_circle_venn(stats: Dict[str, int]) -> None:
    """
    Render a professional 3-circle Venn diagram with improved styling.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    
    # Calculate subsets
    subsets = (
        stats['pure_play_count'] - stats['pure_play_in_sff'],
        stats['sff_data_count'] - stats['pure_play_in_sff'] - stats['not_pure_but_30_in_sff'],
        stats['pure_play_in_sff'],
        stats['not_pure_but_30_count'] - stats['not_pure_but_30_in_sff'],
        0,
        stats['not_pure_but_30_in_sff'],
        0
    )
    
    # Create Venn diagram with modern styling
    v = venn3(
        subsets=subsets,
        set_labels=('LLM Generated\n(≥50%)', 'SFF Data', 'LLM Generated\n(30-49%)'),
        set_colors=('#4CAF50', '#2196F3', '#FFC107'),
        alpha=0.7,
        subset_label_formatter=lambda x: f"{x:,}" if x > 0 else ""
    )
    
    # Customize labels
    label_map = {
        '100': f"Pure Play\nNot in SFF\n{stats['pure_play_not_in_sff']:,}",
        '010': f"SFF Only\n{stats['sff_not_in_pure_play']:,}",
        '001': f"30-49%\nNot in SFF\n{stats['not_pure_but_30_count'] - stats['not_pure_but_30_in_sff']:,}",
        '110': f"Pure Play &\nSFF\n{stats['pure_play_in_sff']:,}",
        '011': f"SFF &\n30-49%\n{stats['not_pure_but_30_in_sff']:,}"
    }
    
    for label_id, text in label_map.items():
        lbl = v.get_label_by_id(label_id)
        if lbl: 
            lbl.set_text(text)
            lbl.set_fontsize(9)
    
    plt.title("Sustainable Finance Company Classification", pad=20)
    st.pyplot(fig)

def plot_green_revenue_distribution(green_revenue: pd.DataFrame) -> None:
    """Enhanced distribution plot with custom bins."""
    fig = px.histogram(
        green_revenue,
        x='greenRevenuePercent',
        nbins=30,
        title='Distribution of Green Revenue Percentages',
        labels={'greenRevenuePercent': 'Green Revenue %'},
        color='pure_play_flag',
        color_discrete_map={'Y': '#2ecc71', 'N': '#95a5a6'},
        template='plotly_white'
    )
    fig.update_layout(
        bargap=0.1,
        xaxis_title="Green Revenue Percentage",
        yaxis_title="Number of Companies",
        legend_title="Pure Play Status"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_country_analysis(green_revenue: pd.DataFrame) -> None:
    """Enhanced country analysis using PyDeck for better mapping"""
    if 'country_code' not in green_revenue.columns:
        st.warning("Country data not available")
        return
    
    # Prepare data
    country_stats = (
        green_revenue
        .dropna(subset=['country_code'])
        .assign(country_code=lambda x: x['country_code'].astype(str).str.strip().str.upper())
        .groupby('country_code')
        .agg(
            company_count=('counterparty_id', 'nunique'),
            avg_green=('greenRevenuePercent', 'mean')
        )
        .reset_index()
    )
    
    if country_stats.empty:
        st.warning("No valid country data available")
        return
    
    # Use PyDeck for better mapping
    view_state = pdk.ViewState(
        latitude=30,
        longitude=0,
        zoom=1
    )
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=country_stats,
        get_position=["longitude", "latitude"],
        get_color="[200, 30, 0, 160]",
        get_radius="company_count * 5000",
        pickable=True
    )
    
    tooltip = {
        "html": "<b>Country:</b> {country_code}<br>"
                "<b>Companies:</b> {company_count}<br>"
                "<b>Avg Green %:</b> {avg_green:.1f}%",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
    
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=view_state,
        layers=[layer],
        tooltip=tooltip
    ))

def plot_industry_analysis(industry_stats: pd.DataFrame) -> None:
    """Interactive industry analysis visualization."""
    fig = px.bar(
        industry_stats,
        x='industry_sector',
        y='avg_green_revenue',
        color='pure_play_pct',
        title='Green Revenue by Industry Sector',
        labels={
            'avg_green_revenue': 'Average Green Revenue %',
            'industry_sector': 'Industry Sector',
            'pure_play_pct': 'Pure Play %'
        },
        color_continuous_scale='Tealgrn',
        hover_data=['company_count']
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_title="Average Green Revenue %",
        coloraxis_colorbar={
            'title': 'Pure Play %',
            'ticksuffix': '%'
        }
    )
    st.plotly_chart(fig, use_container_width=True)
