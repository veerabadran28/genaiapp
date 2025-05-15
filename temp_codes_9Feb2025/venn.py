import streamlit as st
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from typing import Dict

def plot_three_circle_venn(stats: Dict[str, int]) -> None:
    """
    Render a 3-circle Venn diagram showing:
    - LLM Generated (>=50%)
    - LLM Generated (>=30%)
    - SFF Data
    
    Args:
        stats: Dictionary containing Venn diagram statistics
    """
    # Create figure with custom styling
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#f5f5f5')
    ax.set_facecolor('#f5f5f5')
    
    # Define the subsets for the 3-circle Venn
    subsets = (
        stats['pure_play_count'] - stats['pure_play_in_sff'] - stats['not_pure_but_30_in_sff'],
        stats['sff_data_count'] - stats['pure_play_in_sff'] - stats['not_pure_but_30_in_sff'],
        stats['pure_play_in_sff'],
        stats['not_pure_but_30_count'] - stats['not_pure_but_30_in_sff'],
        0,  # We're not tracking SFF with <30% separately
        stats['not_pure_but_30_in_sff']
    )
    
    # Create Venn diagram
    v = venn3(
        subsets=subsets,
        set_labels=('LLM Generated\n(>=50%)', 'SFF Data', 'LLM Generated\n(>=30%)'),
        set_colors=('#4CAF50', '#2196F3', '#FFC107'),  # Green, Blue, Yellow
        alpha=0.7
    )
    
    # Customize labels
    if v.get_label_by_id('100'):
        v.get_label_by_id('100').set_text(f"Pure Play\nNot in SFF\n{stats['pure_play_not_in_sff']:,}")
    if v.get_label_by_id('010'):
        v.get_label_by_id('010').set_text(f"SFF\nNot in LLM\n{stats['sff_not_in_pure_play']:,}")
    if v.get_label_by_id('001'):
        v.get_label_by_id('001').set_text(f"30-49%\nNot in SFF\n{stats['not_pure_but_30_count'] - stats['not_pure_but_30_in_sff']:,}")
    if v.get_label_by_id('110'):
        v.get_label_by_id('110').set_text(f"Pure Play\nin SFF\n{stats['pure_play_in_sff']:,}")
    if v.get_label_by_id('101'):
        v.get_label_by_id('101').set_text("")  # Empty - no direct relationship
    if v.get_label_by_id('011'):
        v.get_label_by_id('011').set_text(f"30-49%\nin SFF\n{stats['not_pure_but_30_in_sff']:,}")
    if v.get_label_by_id('111'):
        v.get_label_by_id('111').set_text("")  # Empty
    
    # Style the diagram
    for text in v.set_labels:
        text.set_fontsize(12)
        text.set_fontweight('bold')
    
    for text in v.subset_labels:
        if text:  # Only if label exists
            text.set_fontsize(11)
    
    plt.title(
        "Company Classification by Green Revenue and SFF Inclusion",
        fontsize=14,
        pad=20
    )
    
    # Add legend
    plt.legend(
        ["Pure Play (≥50%)", "SFF Companies", "Green Revenue (30-49%)"],
        loc='upper left',
        bbox_to_anchor=(1, 1)
    
    # Display in Streamlit
    st.pyplot(fig)
    st.caption(
        "Three-way Venn diagram showing the relationship between:\n"
        "- Companies with ≥50% green revenue (Pure Play)\n"
        "- Companies in the Sustainable Finance Framework (SFF)\n"
        "- Companies with 30-49% green revenue"
    )
    
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
