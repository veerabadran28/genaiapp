# components/non_pure_play_section.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class NonPurePlaySection:
    """
    Class responsible for displaying the Non-Pure Play section
    """
    
    def __init__(self, data_processor):
        """
        Initialize the NonPurePlaySection with data processor
        
        Args:
            data_processor (DataProcessor): The data processor object
        """
        self.data_processor = data_processor
    
    def create_summary_stats(self):
        """
        Create summary statistics for Non-Pure Play section
        """
        # Get counts
        overlap_count = len(self.data_processor.non_pure_play_overlap)
        identified_count = len(self.data_processor.non_pure_play_identified)
        unidentified_count = len(self.data_processor.non_pure_play_unidentified)
        
        # Create pie chart
        labels = ['Overlap', 'In SFF Only', 'In GREEN_REVENUE Only']
        values = [overlap_count, identified_count, unidentified_count]
        colors = ['#9b59b6', '#3498db', '#e74c3c']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.4,
            marker_colors=colors
        )])
        
        fig.update_layout(
            title_text="Non-Pure Play Companies Comparison",
            annotations=[dict(text='Non-Pure Play', x=0.5, y=0.5, font_size=20, showarrow=False)],
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        
        return fig, overlap_count, identified_count, unidentified_count
    
    def create_country_distribution(self):
        """
        Create chart showing country distribution for Non-Pure Play companies
        """
        # Filter data
        df = self.data_processor.green_revenue[self.data_processor.green_revenue['pure_play_flag'] == 'N']
        
        # Get top 10 countries
        country_counts = df['country_code'].value_counts().head(10)
        
        # Create bar chart
        fig = px.bar(
            x=country_counts.index,
            y=country_counts.values,
            color=country_counts.values,
            color_continuous_scale='Reds',
            title='Top 10 Countries for Non-Pure Play Companies',
            labels={'x': 'Country Code', 'y': 'Number of Companies'},
            text=country_counts.values
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Country',
            yaxis_title='Number of Companies',
            coloraxis_showscale=False,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            font=dict(size=12)
        )
        
        # Update trace to show values on top of bars
        fig.update_traces(textposition='outside')
        
        return fig
    
    def create_revenue_ranges(self):
        """
        Create chart showing revenue ranges for Non-Pure Play companies
        """
        # Filter data
        df = self.data_processor.green_revenue[self.data_processor.green_revenue['pure_play_flag'] == 'N'].copy()
        
        # Convert total revenue to numeric
        df['totalRevenue'] = pd.to_numeric(df['totalRevenue'], errors='coerce')
        
        # Create revenue ranges
        bins = [0, 1e6, 1e7, 1e8, 1e9, 1e10, float('inf')]
        labels = ['<1M', '1M-10M', '10M-100M', '100M-1B', '1B-10B', '>10B']
        
        df['revenue_range'] = pd.cut(df['totalRevenue'], bins=bins, labels=labels)
        
        # Get counts by revenue range
        revenue_counts = df['revenue_range'].value_counts().sort_index()
        
        # Create bar chart
        fig = px.bar(
            x=revenue_counts.index,
            y=revenue_counts.values,
            color=revenue_counts.values,
            color_continuous_scale='Reds',
            title='Revenue Ranges for Non-Pure Play Companies',
            labels={'x': 'Revenue Range', 'y': 'Number of Companies'},
            text=revenue_counts.values
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Revenue Range',
            yaxis_title='Number of Companies',
            coloraxis_showscale=False,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            font=dict(size=12)
        )
        
        # Update trace to show values on top of bars
        fig.update_traces(textposition='outside')
        
        return fig
    
    def create_green_revenue_histogram(self):
        """
        Create histogram for green revenue percentages
        """
        # Filter data for non-pure play companies
        df = self.data_processor.green_revenue[self.data_processor.green_revenue['pure_play_flag'] == 'N']
        
        # Create histogram
        fig = px.histogram(
            df,
            x='greenRevenuePercent',
            nbins=10,
            color_discrete_sequence=['#e74c3c'],
            title='Green Revenue Percentage Distribution for Non-Pure Play Companies',
            labels={'greenRevenuePercent': 'Green Revenue Percentage'},
            opacity=0.8
        )
        
        # Add line at 50% for pure play threshold
        fig.add_vline(
            x=50,
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text="Pure Play Threshold (50%)",
            annotation_position="top right"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Green Revenue Percentage',
            yaxis_title='Number of Companies',
            bargap=0.1,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            font=dict(size=12)
        )
        
        return fig
    
    def display(self):
        """
        Display the Non-Pure Play section
        """
        st.markdown('<div class="section-header"><h2>Companies that are not classified as pure play in GREEN REVENUE (<50%)</h2></div>', unsafe_allow_html=True)
        
        # Add note about joining condition
        st.info("Note: Companies are matched based on the SDS in SFF_DATA and counterparty_id in GREEN_REVENUE")
        
        # Get summary stats
        summary_fig, overlap_count, identified_count, unidentified_count = self.create_summary_stats()
        
        # Create two columns for summary and charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(summary_fig, use_container_width=True)
            
            # Add metrics
            st.metric("Total Non-Pure Play Companies", f"{self.data_processor.non_pure_play_count:,}")
            st.metric("Companies in Both Sets (Overlap)", f"{overlap_count:,}")
            st.metric("Companies Only in SFF Data", f"{identified_count:,}")
            st.metric("Companies Only in GREEN REVENUE", f"{unidentified_count:,}")
        
        with col2:
            country_fig = self.create_country_distribution()
            st.plotly_chart(country_fig, use_container_width=True)
        
        # Create two columns for revenue ranges and green revenue histogram
        col1, col2 = st.columns(2)
        
        with col1:
            revenue_fig = self.create_revenue_ranges()
            st.plotly_chart(revenue_fig, use_container_width=True)
        
        with col2:
            green_rev_fig = self.create_green_revenue_histogram()
            st.plotly_chart(green_rev_fig, use_container_width=True)
        
        # Create tabs for data tables
        tabs = st.tabs(["Overlap", "Identified Clients", "Un-Identified Clients"])
        
        with tabs[0]:
            st.subheader("The companies that exists in both GREEN_REVENUE and SFF_DATA")
            st.dataframe(
                self.data_processor.non_pure_play_overlap[[
                    'cob_date', 'productype', 'legal_entity', 'counterparty_id',
                    'counterparty_name', 'parent_id', 'group_id', 'group_name',
                    'bic_code', 'country_code', 'year', 'totalRevenue',
                    'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag'
                ]] if not self.data_processor.non_pure_play_overlap.empty else pd.DataFrame(),
                hide_index=True,
                height=400,
                use_container_width=True
            )
            st.text(f"Total Records: {len(self.data_processor.non_pure_play_overlap)}")
        
        with tabs[1]:
            st.subheader("The companies that exists in SFF_DATA but not exists in GREEN_REVENUE")
            st.dataframe(
                self.data_processor.non_pure_play_identified[[
                    'Pureplay Status', 'SDS', 'Alt SDS', 'Client Name',
                    'Themes', 'Sub Theme', 'TLN', 'SLN', 'CSID',
                    'additional CSID', 'BIC'
                ]] if not self.data_processor.non_pure_play_identified.empty else pd.DataFrame(),
                hide_index=True,
                height=400,
                use_container_width=True
            )
            st.text(f"Total Records: {len(self.data_processor.non_pure_play_identified)}")
        
        with tabs[2]:
            st.subheader("The companies that exists in GREEN_REVENUE but not exists in SFF_DATA")
            st.dataframe(
                self.data_processor.non_pure_play_unidentified[[
                    'cob_date', 'productype', 'legal_entity', 'counterparty_id',
                    'counterparty_name', 'parent_id', 'group_id', 'group_name',
                    'bic_code', 'country_code', 'year', 'totalRevenue',
                    'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag'
                ]] if not self.data_processor.non_pure_play_unidentified.empty else pd.DataFrame(),
                hide_index=True,
                height=400,
                use_container_width=True
            )
            st.text(f"Total Records: {len(self.data_processor.non_pure_play_unidentified)}")
