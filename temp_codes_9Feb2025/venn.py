import streamlit as st
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from typing import Dict

def plot_venn_diagram(stats: Dict[str, int]) -> None:
    """
    Render a customized Venn diagram based on the calculated statistics.
    
    Args:
        stats: Dictionary containing Venn diagram statistics
    """
    # Create figure with custom styling
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#f5f5f5')
    ax.set_facecolor('#f5f5f5')
    
    # Create Venn diagram
    v = venn2(
        subsets=(
            stats['pure_play_count'] - stats['pure_play_in_sff'],
            stats['sff_data_count'] - stats['pure_play_in_sff'],
            stats['pure_play_in_sff']
        ),
        set_labels=('LLM Generated (≥50%)', 'SFF Data'),
        set_colors=('#4CAF50', '#2196F3'),  # Green and blue
        alpha=0.7
    )
    
    # Customize labels with counts and descriptions
    if v.get_label_by_id('10'):
        v.get_label_by_id('10').set_text(
            f"Pure Play\nNot in SFF\n{stats['pure_play_not_in_sff']:,}"
        )
    if v.get_label_by_id('01'):
        v.get_label_by_id('01').set_text(
            f"SFF\nNot Pure Play\n{stats['sff_not_in_pure_play']:,}"
        )
    if v.get_label_by_id('11'):
        v.get_label_by_id('11').set_text(
            f"Both\n{stats['pure_play_in_sff']:,}"
        )
    
    # Style the diagram
    for text in v.set_labels:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    
    for text in v.subset_labels:
        text.set_fontsize(11)
    
    plt.title(
        "Comparison of Pure Play Companies (≥50%) vs SFF Data",
        fontsize=14,
        pad=20
    )
    
    # Display in Streamlit
    st.pyplot(fig)
    st.caption(
        "This Venn diagram shows the overlap between companies classified as "
        "pure play (≥50% green revenue) in the LLM-generated data and those "
        "in the Sustainable Finance Framework (SFF) dataset."
    )

def plot_venn_with_threshold(
    green_revenue: pd.DataFrame,
    sff_data: pd.DataFrame,
    threshold: int = 30
) -> None:
    """
    Render a Venn diagram comparing companies above a specific green revenue threshold
    with SFF data.
    
    Args:
        green_revenue: GREEN_REVENUE dataset
        sff_data: SFF dataset
        threshold: Green revenue percentage threshold
    """
    # Filter companies above threshold
    above_threshold = green_revenue[green_revenue['greenRevenuePercent'] >= threshold]
    
    # Calculate stats
    stats = {
        'above_threshold_count': len(above_threshold),
        'sff_data_count': len(sff_data),
        'overlap_count': len(set(above_threshold['counterparty_id']) & set(sff_data['sds']))
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#f5f5f5')
    ax.set_facecolor('#f5f5f5')
    
    # Create Venn diagram
    v = venn2(
        subsets=(
            stats['above_threshold_count'] - stats['overlap_count'],
            stats['sff_data_count'] - stats['overlap_count'],
            stats['overlap_count']
        ),
        set_labels=(f'Green Revenue ≥{threshold}%', 'SFF Data'),
        set_colors=('#8BC34A', '#2196F3'),  # Light green and blue
        alpha=0.7
    )
    
    # Customize labels
    if v.get_label_by_id('10'):
        v.get_label_by_id('10').set_text(
            f"≥{threshold}%\nNot in SFF\n{stats['above_threshold_count'] - stats['overlap_count']:,}"
        )
    if v.get_label_by_id('01'):
        v.get_label_by_id('01').set_text(
            f"SFF\nNot ≥{threshold}%\n{stats['sff_data_count'] - stats['overlap_count']:,}"
        )
    if v.get_label_by_id('11'):
        v.get_label_by_id('11').set_text(
            f"Both\n{stats['overlap_count']:,}"
        )
    
    plt.title(
        f"Companies with Green Revenue ≥{threshold}% vs SFF Data",
        fontsize=14,
        pad=20
    )
    
    st.pyplot(fig)
