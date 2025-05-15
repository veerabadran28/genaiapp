import streamlit as st
import pandas as pd
from typing import List

def display_data_table(
    df: pd.DataFrame,
    columns: List[str],
    title: str,
    key: str
) -> None:
    """
    Display a styled data table with the specified columns.
    
    Args:
        df: DataFrame to display
        columns: List of columns to show
        title: Section title
        key: Unique key for Streamlit component
    """
    st.subheader(title)
    
    if df.empty:
        st.warning("No data available")
        return
    
    # Select only the requested columns that exist in the DataFrame
    display_cols = [col for col in columns if col in df.columns]
    
    # Convert to string for display
    display_df = df[display_cols].astype(str)
    
    # Display with Streamlit
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={col: {"width": "medium"} for col in display_cols}
    )
