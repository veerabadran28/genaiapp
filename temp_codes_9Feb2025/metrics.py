import streamlit as st
from typing import Dict

def display_key_metrics(stats: Dict[str, int]) -> None:
    """
    Display key metrics in a visually appealing card layout.
    
    Args:
        stats: Dictionary containing various metrics to display
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Pure Play Companies (≥50%)",
            value=f"{stats.get('pure_play_count', 0):,}",
            help="Companies with green revenue ≥50% from LLM data"
        )
    
    with col2:
        st.metric(
            label="Pure Play in SFF Data",
            value=f"{stats.get('pure_play_in_sff', 0):,}",
            help="Pure play companies also in SFF dataset"
        )
    
    with col3:
        st.metric(
            label="Pure Play Not in SFF",
            value=f"{stats.get('pure_play_not_in_sff', 0):,}",
            help="Pure play companies not in SFF dataset"
        )
    
    with col4:
        st.metric(
            label="SFF Not Pure Play",
            value=f"{stats.get('sff_not_in_pure_play', 0):,}",
            help="Companies in SFF but not pure play in LLM data"
        )

def display_comparison_metrics(
    overlap_count: int,
    sff_only_count: int,
    green_only_count: int,
    section_title: str
) -> None:
    """
    Display comparison metrics for a specific section.
    
    Args:
        overlap_count: Number of overlapping companies
        sff_only_count: Number of companies only in SFF
        green_only_count: Number of companies only in Green Revenue
        section_title: Title for the metrics section
    """
    st.subheader(f"📊 {section_title} Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Overlap Count",
            value=f"{overlap_count:,}",
            help="Companies present in both datasets"
        )
    
    with col2:
        st.metric(
            label="SFF Only Count",
            value=f"{sff_only_count:,}",
            help="Companies only in SFF dataset"
        )
    
    with col3:
        st.metric(
            label="Green Only Count",
            value=f"{green_only_count:,}",
            help="Companies only in Green Revenue dataset"
        )
