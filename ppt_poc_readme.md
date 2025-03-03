# Sales Dashboard - Code Explanation

## Imports Section

```python
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from io import BytesIO
```

This section imports the necessary libraries:
- `streamlit` - For creating the interactive web interface
- `pandas` and `numpy` - For data manipulation and generation
- `altair` - For creating visualizations in Streamlit
- `pptx` and related modules - For PowerPoint creation and customization
- `BytesIO` - For handling in-memory file operations

## Data Generation

```python
def create_enhanced_data():
    np.random.seed(42)  # For reproducibility
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    regions = ['North', 'South', 'East', 'West']
    products = ['Product A', 'Product B', 'Product C']
    
    data = []
    
    for month in months:
        for region in regions:
            for product in products:
                # Base sales
                sales = np.random.randint(15000, 50000)
                # Cost (60-80% of sales)
                cost = int(sales * np.random.uniform(0.6, 0.8))
                # Profit
                profit = sales - cost
                # Margin percentage
                margin = round((profit / sales) * 100, 1)
                # Units sold
                unit_price = np.random.uniform(50, 200)
                units = int(sales / unit_price)
                # Market share
                market_share = np.random.uniform(5, 25)
                # Customer satisfaction
                csat = np.random.randint(70, 98)
                
                data.append({
                    'Month': month,
                    'Region': region,
                    'Product': product,
                    'Sales': sales,
                    'Cost': cost,
                    'Profit': profit,
                    'Margin': margin,
                    'Units': units,
                    'UnitPrice': round(unit_price, 2),
                    'MarketShare': round(market_share, 1),
                    'CSAT': csat
                })
    
    return pd.DataFrame(data)
```

This function creates a rich sample dataset with:
- Three dimensions: Month, Region, and Product
- Multiple metrics: Sales, Cost, Profit, Margin, Units, Unit Price, Market Share, and Customer Satisfaction
- Realistic relationships between metrics (e.g., Profit = Sales - Cost)
- Random but realistic values for demonstration

The function generates data for every combination of month, region, and product, creating a comprehensive dataset with 72 rows (6 months × 4 regions × 3 products).

## Data Aggregation

```python
def prepare_chart_datasets(df, dimension, measure):
    # Group by the selected dimension
    aggregated = df.groupby(dimension)[measure].sum().reset_index()
    
    # If dimension is Month, sort it chronologically
    if dimension == 'Month':
        month_order = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6}
        aggregated['MonthNum'] = aggregated['Month'].map(month_order)
        aggregated = aggregated.sort_values('MonthNum')
        if 'MonthNum' in aggregated.columns:
            aggregated = aggregated.drop('MonthNum', axis=1)
    
    return aggregated
```

This function:
- Takes the full dataset and aggregates it based on a selected dimension and measure
- Groups the data by the chosen dimension (Month, Region, or Product)
- Sums the values of the selected measure for each unique value of the dimension
- Sorts months chronologically if Month is the selected dimension
- Returns a clean, aggregated DataFrame for charting

## PowerPoint Generation

```python
def create_ppt(df, chart_type, dimension, measure):
    # Create a presentation object
    prs = Presentation()
    
    # Make sure we're working with clean data
    df = df.reset_index(drop=True).copy()
    
    # Create aggregated dataset
    chart_data_df = prepare_chart_datasets(df, dimension, measure)
    
    # Title Slide
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = f"{measure} by {dimension}"
    subtitle.text = f"Generated from Streamlit App - March 2025"
```

The PowerPoint generation begins by:
- Creating a Presentation object
- Making a clean copy of the dataset
- Aggregating the data based on user selections
- Creating a title slide

### Creating the Data Table Slide

```python
    # Dataset Slide
    dataset_slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(dataset_slide_layout)
    title = slide.shapes.title
    title.text = f"{measure} by {dimension} Data"
    
    # Add table
    rows, cols = len(chart_data_df) + 1, len(chart_data_df.columns)
    table = slide.shapes.add_table(rows, cols, left, top, width, height).table
    
    # Headers
    for col_idx, col_name in enumerate(chart_data_df.columns):
        cell = table.cell(0, col_idx)
        cell.text = str(col_name)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(230, 230, 230)  # Light gray background
        paragraph = cell.text_frame.paragraphs[0]
        paragraph.font.bold = True
        paragraph.font.size = Pt(12)
    
    # Data
    for row_idx, row in chart_data_df.iterrows():
        for col_idx, value in enumerate(row):
            cell = table.cell(row_idx + 1, col_idx)
            cell.text = str(value)
            # Make measure values blue to indicate they're editable
            if col_idx == 1:  # Measure column (second column)
                paragraph = cell.text_frame.paragraphs[0]
                paragraph.font.color.rgb = RGBColor(0, 0, 220)  # Blue color
                paragraph.font.bold = True
```

This section:
- Creates a slide for the data table
- Adds a table with the aggregated data
- Formats the table headers with a gray background and bold text
- Highlights the measure values in blue to indicate they're editable

### Creating the Chart Slide

```python
    # Chart Slide
    chart_slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(chart_slide_layout)
    title = slide.shapes.title
    title.text = f"{measure} by {dimension}"
    
    # Create chart with the selected data
    chart_data = CategoryChartData()
    chart_data.categories = chart_data_df[dimension].tolist()
    chart_data.add_series(measure, chart_data_df[measure].tolist())
    
    # Adjust chart dimensions based on chart type
    if chart_type == XL_CHART_TYPE.PIE:
        x, y, cx, cy = Inches(1.0), Inches(1.5), Inches(8.0), Inches(4.5)
    elif chart_type in [XL_CHART_TYPE.LINE, XL_CHART_TYPE.LINE_MARKERS]:
        x, y, cx, cy = Inches(1.0), Inches(1.5), Inches(6.0), Inches(4.0)
    else:
        x, y, cx, cy = Inches(0.75), Inches(1.5), Inches(6.5), Inches(4.5)
    
    chart = slide.shapes.add_chart(
        chart_type, x, y, cx, cy, chart_data
    ).chart
```

This section:
- Creates a slide for the chart
- Prepares chart data using the aggregated dataset
- Adjusts chart dimensions based on the chart type
- Adds the chart to the slide with the selected chart type

### Chart Customization

```python
    # Configure chart with minimal formatting to avoid repair issues
    chart.has_title = True
    chart.chart_title.text_frame.text = f"{measure} by {dimension}"
    
    # Always place legend outside the chart
    chart.has_legend = True
    if chart_type == XL_CHART_TYPE.PIE:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    else:
        chart.legend.position = XL_LEGEND_POSITION.RIGHT
    
    # Do not use include_in_layout for line charts
    if chart_type not in [XL_CHART_TYPE.LINE, XL_CHART_TYPE.LINE_MARKERS]:
        chart.legend.include_in_layout = False
```

This section handles chart customization:
- Adds a title to the chart
- Positions the legend based on chart type
- Uses special handling for line charts to avoid PowerPoint repair prompts

### Data Labels and Axis Formatting

```python
    # Axis formatting for non-pie charts only
    if chart_type != XL_CHART_TYPE.PIE:
        try:
            category_axis = chart.category_axis
            if hasattr(category_axis, 'tick_labels'):
                category_axis.tick_labels.font.size = Pt(12)
                category_axis.tick_labels.font.bold = True
        except (AttributeError, ValueError):
            pass
    
    # Data labels
    plot = chart.plots[0]
    plot.has_data_labels = True
    data_labels = plot.data_labels
    data_labels.font.size = Pt(12)
    data_labels.font.bold = True
    data_labels.number_format = '#,##0'
    
    # Different data label position for different chart types
    if chart_type == XL_CHART_TYPE.PIE:
        pass
    elif chart_type in [XL_CHART_TYPE.LINE, XL_CHART_TYPE.LINE_MARKERS]:
        data_labels.position = XL_LABEL_POSITION.ABOVE
    else:
        data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
```

This section:
- Formats the axis labels for non-pie charts
- Adds data labels to the chart
- Positions the data labels differently based on chart type
- Uses a try-except block to handle chart-specific properties safely

## Streamlit Interface

```python
def main():
    st.title("Enhanced Sales Dashboard to PowerPoint Converter")
    
    # Generate and cache enhanced data
    @st.cache_data
    def get_data():
        return create_enhanced_data()
    
    df = get_data()
    
    # Dashboard configuration
    st.sidebar.header("Dashboard Configuration")
    
    # Chart type selection
    chart_type_name = st.sidebar.selectbox(
        "Select Chart Type", 
        list(CHART_TYPES.keys())
    )
    chart_type = CHART_TYPES[chart_type_name]
    
    # Dimension selection
    dimension_options = ['Month', 'Region', 'Product']
    dimension = st.sidebar.selectbox(
        "Group By (Dimension)", 
        dimension_options
    )
    
    # Measure selection
    measure_options = ['Sales', 'Cost', 'Profit', 'Units', 'Margin', 'MarketShare', 'CSAT']
    measure = st.sidebar.selectbox(
        "Measure", 
        measure_options
    )
```

The main function sets up the Streamlit interface:
- Creates a title
- Generates and caches the sample data
- Sets up sidebar controls for selecting:
  - Chart type (from the `CHART_TYPES` dictionary)
  - Dimension to group by
  - Measure to aggregate

### Data Preview

```python
    # Show data preview with filters
    with st.expander("View Data", expanded=False):
        st.dataframe(df, use_container_width=True)
    
    # Generate aggregated data for selected dimension and measure
    chart_data_df = prepare_chart_datasets(df, dimension, measure)
    
    # Show aggregated data
    st.subheader(f"{measure} by {dimension}")
    st.dataframe(chart_data_df, use_container_width=True)
```

This section:
- Shows the full dataset in an expandable section
- Aggregates the data based on user selections
- Displays the aggregated data that will be used for the chart

### Chart Preview

```python
    # Display chart in Streamlit with data labels using Altair
    st.subheader("Chart Preview")
    
    # Create appropriate Altair chart based on selection
    if chart_type_name in ["Column Chart"]:
        chart = alt.Chart(chart_data_df).mark_bar().encode(
            x=alt.X(f'{dimension}:N', title=dimension),
            y=alt.Y(f'{measure}:Q', title=measure)
        )
    elif chart_type_name in ["Bar Chart"]:
        chart = alt.Chart(chart_data_df).mark_bar().encode(
            y=alt.Y(f'{dimension}:N', title=dimension),
            x=alt.X(f'{measure}:Q', title=measure)
        )
    # ... other chart types
```

This section:
- Creates the appropriate Altair chart based on the selected chart type
- Configures the chart with the selected dimension and measure
- Sets up different chart specifications for different chart types

### Chart Formatting and Download

```python
    # Add text labels for all non-pie charts
    if chart_type_name != "Pie Chart":
        text = chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5,
            fontSize=14,
            color='black'
        ).encode(
            text=alt.Text(f'{measure}:Q', format=',.0f')
        )
        chart = chart + text
    
    # Adjust chart properties
    chart_properties = { ... }
    
    # Configure legend position
    if chart_type_name == "Pie Chart":
        final_chart = chart.properties(**chart_properties).configure_legend(
            orient='bottom'
        )
    else:
        final_chart = chart.properties(**chart_properties).configure_legend(
            orient='right',
            offset=20
        )
    
    st.altair_chart(final_chart, use_container_width=True)
    
    # Download button for PPT
    if st.button("Generate PowerPoint"):
        with st.spinner("Creating PowerPoint..."):
            ppt_buffer = create_ppt(df, chart_type, dimension, measure)
        
        st.success("PowerPoint generated successfully!")
        
        st.download_button(
            label="Download PowerPoint",
            data=ppt_buffer.getvalue(),
            file_name=f"{measure}_by_{dimension}_{chart_type_name}.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )
```

The final section:
- Adds text labels to non-pie charts
- Configures chart properties and legend position
- Displays the chart in the Streamlit interface
- Provides a button to generate the PowerPoint
- Creates a download button for the generated PowerPoint file
- Shows usage instructions for working with the PowerPoint

## Chart Type Mapping

```python
CHART_TYPES = {
    "Column Chart": XL_CHART_TYPE.COLUMN_CLUSTERED,
    "Bar Chart": XL_CHART_TYPE.BAR_CLUSTERED,
    "Line Chart": XL_CHART_TYPE.LINE,
    "Line Chart with Markers": XL_CHART_TYPE.LINE_MARKERS,
    "Pie Chart": XL_CHART_TYPE.PIE,
    "Area Chart": XL_CHART_TYPE.AREA
}
```

This dictionary maps user-friendly chart names to PowerPoint chart type constants, making it easy to add or remove chart types as needed.
