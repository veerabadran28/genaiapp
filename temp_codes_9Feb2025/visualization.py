import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Dict
import pydeck as pdk  # Using PyDeck for better maps

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
