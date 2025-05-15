# components/pure_play_section.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class PurePlaySection:
    """
    Class responsible for displaying the Pure Play section
    """
    
    def __init__(self, data_processor):
        """
        Initialize the PurePlaySection with data processor
        
        Args:
            data_processor (DataProcessor): The data processor object
        """
        self.data_processor = data_processor
    
    def create_summary_stats(self):
        """
        Create summary statistics for Pure Play section
        """
        # Get counts
        overlap_count = len(self.data_processor.pure_play_overlap)
        identified_count = len(self.data_processor.pure_play_identified)
        unidentified_count = len(self.data_processor.pure_play_unidentified)
        
        # Create pie chart
        labels = ['Overlap', 'In SFF Only', 'In GREEN_REVENUE Only']
        values = [overlap_count, identified_count, unidentified_count]
        colors = ['#9b59b6', '#3498db', '#2ecc71']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.4,
            marker_colors=colors
        )])
        
        fig.update_layout(
            title_text="Pure Play Companies Comparison",
            annotations=[dict(text='Pure Play', x=0.5, y=0.5, font_size=20, showarrow=False)],
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        
        return fig, overlap_count, identified_count, unidentified_count
    
    def create_theme_chart(self):
        """
        Create chart showing themes for Pure Play companies
        """
        # Check if Theme column exists in overlap data
        if 'Themes' in self.data_processor.pure_play_overlap.columns:
            # Get theme counts
            themes = self.data_processor.pure_play_overlap['Themes'].value_counts().head(10)
            
            # Create horizontal bar chart
            fig = px.bar(
                x=themes.values,
                y=themes.index,
                orientation='h',
                color=themes.values,
                color_continuous_scale='Viridis',
                title='Top 10 Themes of Pure Play Companies (Joined by SDS-counterparty_id)',
                labels={'x': 'Count', 'y': 'Theme'},
                text=themes.values
            )
            
            # Update layout
            fig.update_layout(
                xaxis_title='Number of Companies',
                yaxis_title='Theme',
                coloraxis_showscale=False,
                plot_bgcolor='rgba(240, 240, 240, 0.5)',
                font=dict(size=12)
            )
            
            # Update trace to show values
            fig.update_traces(textposition='outside')
            
            return fig
        else:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No Theme data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
    
    def create_green_revenue_box_plot(self):
        """
        Create box plot for green revenue percentages
        """
        # Filter data for pure play companies
        df = self.data_processor.green_revenue[self.data_processor.green_revenue['pure_play_flag'] == 'Y']
        
        # Create box plot
        fig = px.box(
            df,
            y='greenRevenuePercent',
            title='Green Revenue Percentage Distribution for Pure Play Companies',
            labels={'greenRevenuePercent': 'Green Revenue Percentage'},
            color_discrete_sequence=['#2ecc71']
        )
        
        # Add line at 50% for pure play threshold
        fig.add_hline(
            y=50,
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text="Pure Play Threshold (50%)",
            annotation_position="top right"
        )
        
        # Update layout
        fig.update_layout(
            yaxis_title='Green Revenue Percentage',
            showlegend=False,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            font=dict(size=12)
        )
        
        return fig
    
    def display(self):
        """
        Display the Pure Play section
        """
        st.markdown('<div class="section-header"><h2>Companies that are classified as pure play in GREEN REVENUE (>=50%)</h2></div>', unsafe_allow_html=True)
        
        # Add note about joining condition
        st.info("Note: Companies are matched based on the SDS in SFF_DATA and counterparty_id in GREEN_REVENUE")
        
        # Get summary stats
        summary_fig, overlap_count, identified_count, unidentified_count = self.create_summary_stats()
        
        # Create two columns for summary and theme chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(summary_fig, use_container_width=True)
            
            # Add metrics
            st.metric("Total Pure Play Companies", f"{self.data_processor.pure_play_count:,}")
            st.metric("Companies in Both Sets (Overlap)", f"{overlap_count:,}")
            st.metric("Companies Only in SFF Data", f"{identified_count:,}")
            st.metric("Companies Only in GREEN REVENUE", f"{unidentified_count:,}")
        
        with col2:
            theme_fig = self.create_theme_chart()
            st.plotly_chart(theme_fig, use_container_width=True)
            
            # Add box plot
            green_rev_fig = self.create_green_revenue_box_plot()
            st.plotly_chart(green_rev_fig, use_container_width=True)
        
        # Create tabs for data tables
        tabs = st.tabs(["Overlap", "Identified Clients", "Un-Identified Clients"])
        
        with tabs[0]:
            st.subheader("The companies that exists in both GREEN_REVENUE and SFF_DATA")
            st.dataframe(
                self.data_processor.pure_play_overlap[[
                    'cob_date', 'productype', 'legal_entity', 'counterparty_id',
                    'counterparty_name', 'parent_id', 'group_id', 'group_name',
                    'bic_code', 'country_code', 'year', 'totalRevenue',
                    'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag'
                ]] if not self.data_processor.pure_play_overlap.empty else pd.DataFrame(),
                hide_index=True,
                height=400,
                use_container_width=True
            )
            st.text(f"Total Records: {len(self.data_processor.pure_play_overlap)}")
        
        with tabs[1]:
            st.subheader("The companies that exists in SFF_DATA but not exists in GREEN_REVENUE")
            st.dataframe(
                self.data_processor.pure_play_identified[[
                    'Pureplay Status', 'SDS', 'Alt SDS', 'Client Name',
                    'Themes', 'Sub Theme', 'TLN', 'SLN', 'CSID',
                    'additional CSID', 'BIC'
                ]] if not self.data_processor.pure_play_identified.empty else pd.DataFrame(),
                hide_index=True,
                height=400,
                use_container_width=True
            )
            st.text(f"Total Records: {len(self.data_processor.pure_play_identified)}")
        
        with tabs[2]:
            st.subheader("The companies that exists in GREEN_REVENUE but not exists in SFF_DATA")
            st.dataframe(
                self.data_processor.pure_play_unidentified[[
                    'cob_date', 'productype', 'legal_entity', 'counterparty_id',
                    'counterparty_name', 'parent_id', 'group_id', 'group_name',
                    'bic_code', 'country_code', 'year', 'totalRevenue',
                    'greenRevenuePercent', 'justification', 'dataSources', 'pure_play_flag'
                ]] if not self.data_processor.pure_play_unidentified.empty else pd.DataFrame(),
                hide_index=True,
                height=400,
                use_container_width=True
            )
            st.text(f"Total Records: {len(self.data_processor.pure_play_unidentified)}")
