# sustainable_finance_dashboard/visualizations.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles

def plot_green_revenue_distribution(df, column='greenRevenuePercent', title="Distribution of Green Revenue (%)"):
    if df.empty or column not in df.columns or df[column].isnull().all():
        st.info(f"Not enough data or column '{column}' is empty/missing for distribution plot.")
        return
    df_plot = df.copy()
    df_plot[column] = pd.to_numeric(df_plot[column], errors='coerce')
    df_plot.dropna(subset=[column], inplace=True)
    if df_plot.empty:
        st.info(f"No valid numeric data in column '{column}' for distribution plot after cleaning.")
        return
    fig = px.histogram(df_plot, x=column, nbins=30, title=title, labels={column: "Green Revenue Percentage"})
    fig.update_layout(bargap=0.1, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

def plot_pure_play_flag_distribution(df, column='pure_play_flag', title="Pure Play Flag Distribution (GREEN_REVENUE)"):
    if df.empty or column not in df.columns:
        st.info("Not enough data for pure play flag distribution.")
        return
    counts = df[column].value_counts().reset_index()
    counts.columns = ['flag', 'count']
    fig = px.pie(counts, names='flag', values='count', title=title,
                 color_discrete_map={'Y': 'mediumseagreen', 'N': 'lightcoral', 'N/A': 'lightgrey'}, hole=0.3)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

def plot_revenue_by_attribute(df, attribute_col, revenue_col='totalRevenue', top_n=10, title_prefix="Total Revenue"):
    if df.empty or attribute_col not in df.columns or revenue_col not in df.columns:
        st.info(f"Cannot plot '{title_prefix} by {attribute_col}': Data or columns missing.")
        return
    df_copy = df.copy()
    df_copy[revenue_col] = pd.to_numeric(df_copy[revenue_col], errors='coerce')
    df_copy.dropna(subset=[revenue_col, attribute_col], inplace=True)
    if df_copy.empty:
        st.info(f"No valid data for '{title_prefix} by {attribute_col}' after cleaning.")
        return
    if df_copy[attribute_col].nunique() == 0:
        st.info(f"Attribute column '{attribute_col}' has no unique values for grouping.")
        return
    grouped_data = df_copy.groupby(attribute_col, observed=True)[revenue_col].sum().nlargest(top_n).reset_index()
    if grouped_data.empty:
        st.info(f"No data to display for '{title_prefix} by {attribute_col}' after grouping.")
        return
    fig = px.bar(grouped_data, x=attribute_col, y=revenue_col,
                 title=f"{title_prefix} by {attribute_col} (Top {top_n})",
                 labels={revenue_col: 'Total Revenue', attribute_col: attribute_col.replace('_', ' ').title()},
                 text_auto=True)
    fig.update_layout(xaxis_title=attribute_col.replace('_', ' ').title(), yaxis_title="Total Revenue",
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

def plot_custom_venn_diagram(set1_size, set2_size, set3_size,
                             set12_overlap, set13_overlap, set23_overlap,
                             set123_overlap,
                             set_labels=('Set 1', 'Set 2', 'Set 3'),
                             title="Company Data Overlap"):
    s1_only = set1_size - set12_overlap - set13_overlap + set123_overlap
    s2_only = set2_size - set12_overlap - set23_overlap + set123_overlap
    s3_only = set3_size - set13_overlap - set23_overlap + set123_overlap
    s12_only = set12_overlap - set123_overlap
    s13_only = set13_overlap - set123_overlap
    s23_only = set23_overlap - set123_overlap
    s123 = set123_overlap
    subsets = tuple(max(0, s) for s in [s1_only, s2_only, s12_only, s3_only, s13_only, s23_only, s123])

    if sum(subsets) == 0 and any(s > 0 for s in [set1_size, set2_size, set3_size]):
        st.warning(f"Venn diagram subset calculation issue. Check overlap consistency. Inputs: Sizes=({set1_size}, {set2_size}, {set3_size}), Overlaps=({set12_overlap}, {set13_overlap}, {set23_overlap}, {set123_overlap})")
        return

    fig, ax = plt.subplots(figsize=(12, 9))
    v = venn3(subsets=subsets, set_labels=set_labels,
              set_colors=('#3E7CB1', '#F4A261', '#8CB369'), alpha=0.75, ax=ax)

    if v:
        for text_obj in v.set_labels:
            if text_obj: text_obj.set_fontsize(14); text_obj.set_fontweight('bold')
        for text_obj in v.subset_labels:
            if text_obj: text_obj.set_fontsize(11); text_obj.set_color('white'); text_obj.set_fontweight('bold')
        
        # Example of adding more descriptive text (adjust positions and content as needed)
        # This is highly dependent on the specific interpretation of venn1.jpeg
        if v.get_label_by_id('011') and (s23_only + s123 > 0): # LLM_PP and SFF
            patch = v.get_patch_by_id('011')
            if patch:
                x = patch.get_path().vertices[:,0].mean()
                y = patch.get_path().vertices[:,1].mean() - 0.1 # Adjust y for better placement
                ax.text(x, y, "LLM (>=50%)\n&\nSFF", ha='center', va='center', fontsize=8, color='black',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))


    venn3_circles(subsets=subsets, linestyle="--", linewidth=1.5, color="dimgray", ax=ax)
    plt.title(title, fontsize=18, pad=20, fontweight='bold')
    ax.set_facecolor('#f0f2f6')
    fig.patch.set_facecolor('rgba(0,0,0,0)')
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

def plot_comparison_metrics(overlap_count, identified_count, unidentified_count, section_title):
    if any(count is None or not isinstance(count, (int, float)) for count in [overlap_count, identified_count, unidentified_count]):
        st.info(f"Insufficient or invalid data for comparison metrics chart in '{section_title}'.")
        return
    categories = ['Overlap', 'Identified (SFF only)', 'Un-Identified (GR only)']
    counts = [overlap_count, identified_count, unidentified_count]
    df_summary = pd.DataFrame({'Category': categories, 'Count': counts})
    fig = px.bar(df_summary, x='Category', y='Count', title=f"Summary: {section_title}",
                 color='Category', text_auto=True,
                 color_discrete_map={'Overlap': '#1f77b4', 'Identified (SFF only)': '#ff7f0e', 'Un-Identified (GR only)': '#2ca02c'})
    fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
