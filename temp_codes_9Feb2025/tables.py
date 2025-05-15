import streamlit as st
import pandas as pd
from typing import List, Optional

def display_data_table(
    df: pd.DataFrame,
    columns: List[str],
    title: str,
    key: Optional[str] = None  # Made optional with default None
) -> None:
    """
    Display a styled data table with the specified columns.
    
    Args:
        df: DataFrame to display
        columns: List of columns to show
        title: Section title
        key: Optional unique key for Streamlit component
    """
    st.subheader(title)
    
    if df.empty:
        st.warning("No data available")
        return
    
    display_cols = [col for col in columns if col in df.columns]
    
    st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={col: {"width": "medium"} for col in display_cols},
        key=key  # Pass the key parameter
    )
