import streamlit as st
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from typing import Dict
import numpy as np

def plot_three_circle_venn(stats: Dict[str, int]) -> None:
    """
    Render a 3-circle Venn diagram matching the reference image layout.
    Circles arranged in triangular formation with:
    - Top: LLM Generated (>=50%)
    - Bottom Left: SFF Data
    - Bottom Right: LLM Generated (>=30%)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Custom circle positions to match reference image
    v = venn3(
        subsets=(
            stats['pure_play_count'] - stats['pure_play_in_sff'] - (stats['not_pure_but_30_count'] - stats['not_pure_but_30_in_sff']),
            stats['sff_data_count'] - stats['pure_play_in_sff'] - stats['not_pure_but_30_in_sff'],
            stats['pure_play_in_sff'],
            stats['not_pure_but_30_count'] - stats['not_pure_but_30_in_sff'],
            0,
            stats['not_pure_but_30_in_sff'],
            0
        ),
        set_labels=('LLM Generated\n(>=50%)', 'SFF Data', 'LLM Generated\n(>=30%)'),
        set_colors=('#4CAF50', '#2196F3', '#FFC107'),
        alpha=0.7,
        subset_label_formatter=lambda x: f"{x:,}" if x > 0 else ""
    )
    
    # Adjust positions to match reference
    if hasattr(v, 'patches'):
        v.patches[0].center = (0.3, 0.6)  # Top circle (LLM >=50%)
        v.patches[1].center = (0.2, 0.4)  # Bottom left (SFF)
        v.patches[2].center = (0.4, 0.4)  # Bottom right (LLM >=30%)
    
    # Custom labels matching reference image
    label_map = {
        '100': f"COMPANIES\nNOT IN\nSFR_DATA\n{stats['pure_play_not_in_sff']:,}",
        '010': f"COMPANIES\nNOT PART OF\nSPF_DATA\n{stats['sff_not_in_pure_play']:,}",
        '001': f"COMPANIES\n{stats['not_pure_but_30_count'] - stats['not_pure_but_30_in_sff']:,}",
        '110': f"COMPANIES\nIN BOTH\nSFR_DATA &\nLLM_GENERATED\n{stats['pure_play_in_sff']:,}",
        '011': f"COMPANIES\n{stats['not_pure_but_30_in_sff']:,}"
    }
    
    for label_id, text in label_map.items():
        lbl = v.get_label_by_id(label_id)
        if lbl:
            lbl.set_text(text)
            lbl.set_fontsize(8)
            lbl.set_linespacing(1.5)
    
    plt.title("Company Classification by Green Revenue", pad=20)
    st.pyplot(fig)
