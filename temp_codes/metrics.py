# components/metrics.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import io
import base64

class MetricsSection:
    """
    Class responsible for displaying metrics and statistics
    """
    
    def __init__(self, data_processor):
        """
        Initialize the MetricsSection with data processor
        
        Args:
            data_processor (DataProcessor): The data processor object
        """
        self.data_processor = data_processor
    
    def create_venn_diagram(self):
        """
        Create a Venn diagram showing dataset overlaps
        """
        try:
            dataset_info = self.data_processor.get_dataset_info()
            
            # Create a figure
            plt.figure(figsize=(10, 10))
            
            # Create the Venn diagram
            venn = venn3(
                subsets=(
                    dataset_info['pcaf_only'],
                    dataset_info['llm_only'],
                    dataset_info['pcaf_llm'],
                    dataset_info['sff_only'],
                    dataset_info['pcaf_sff'],
                    dataset_info['llm_sff'],
                    dataset_info['all_three']
                ),
                set_labels=('PCAF_DATA', 'LLM_GENERATED', 'SFF_DATA')
            )
            
            # Set colors safely
            if venn:
                # Set colors for main patches if they exist
                for patch_id in ['100', '010', '001']:
                    patch = venn.get_patch_by_id(patch_id)
                    if patch:
                        if patch_id == '100':
                            patch.set_color('#3498db')  # Blue for PCAF
                        elif patch_id == '010':
                            patch.set_color('#2ecc71')  # Green for LLM
                        elif patch_id == '001':
                            patch.set_color('#e74c3c')  # Red for SFF
                
                # Set alpha for better visualization
                for patch in venn.patches:
                    if patch:
                        patch.set_alpha(0.7)
                
                # Add subset labels for clearer understanding
                for label_id, label_text in [
                    ('100', f'PCAF only\n{dataset_info["pcaf_only"]}'),
                    ('010', f'LLM only\n{dataset_info["llm_only"]}'),
                    ('001', f'SFF only\n{dataset_info["sff_only"]}'),
                    ('110', f'PCAF & LLM\n{dataset_info["pcaf_llm"]}'),
                    ('101', f'PCAF & SFF\n{dataset_info["pcaf_sff"]}'),
                    ('011', f'LLM & SFF\n{dataset_info["llm_sff"]}'),
                    ('111', f'All three\n{dataset_info["all_three"]}')
                ]:
                    label = venn.get_label_by_id(label_id)
                    if label:
                        label.set_text(label_text)
            
            plt.title('Dataset Overlaps', fontsize=16)
            
            # Save the figure to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            # Return the figure as base64 encoded data
            return base64.b64encode(buf.read()).decode()
        
        except Exception as e:
            print(f"Error creating Venn diagram: {e}")
            
            # Create a simple fallback image with text
            plt.figure(figsize=(10, 10))
            plt.text(0.5, 0.5, "Venn diagram could not be created.\nPossibly insufficient data for overlaps.",
                    horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.axis('off')
            
            # Save the figure to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            # Return the fallback figure as base64 encoded data
            return base64.b64encode(buf.read()).decode()
    
    def create_green_revenue_distribution(self):
        """
        Create a chart showing the distribution of green revenue percentages
        """
        # Create histogram with plotly
        fig = px.histogram(
            self.data_processor.green_revenue,
            x='greenRevenuePercent',
            nbins=20,
            color_discrete_sequence=['#2ecc71'],
            title='Distribution of Green Revenue Percentages',
            labels={'greenRevenuePercent': 'Green Revenue Percent'},
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
    
    def create_country_distribution(self):
        """
        Create a chart showing the distribution of companies by country
        """
        # Get top 10 countries
        country_counts = self.data_processor.green_revenue['country_code'].value_counts().head(10)
        
        # Create bar chart
        fig = px.bar(
            x=country_counts.index,
            y=country_counts.values,
            color=country_counts.values,
            color_continuous_scale='Viridis',
            title='Top 10 Countries by Number of Companies',
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
    
    def create_pure_play_by_year(self):
        """
        Create a chart showing pure play companies by year
        """
        # Group by year and pure_play_flag
        year_group = self.data_processor.green_revenue.groupby(['year', 'pure_play_flag']).size().reset_index(name='count')
        
        # Create grouped bar chart
        fig = px.bar(
            year_group,
            x='year',
            y='count',
            color='pure_play_flag',
            barmode='group',
            title='Number of Companies by Year and Pure Play Status',
            labels={'year': 'Year', 'count': 'Number of Companies', 'pure_play_flag': 'Pure Play Status'},
            color_discrete_map={'Y': '#2ecc71', 'N': '#e74c3c'}
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Number of Companies',
            legend_title='Pure Play Status',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            font=dict(size=12)
        )
        
        return fig
    
    def create_revenue_by_green_pct(self):
        """
        Create a scatter plot of total revenue vs green revenue percentage
        """
        # Clean and prepare data
        df = self.data_processor.green_revenue.copy()
        df['totalRevenue'] = pd.to_numeric(df['totalRevenue'], errors='coerce')
        df = df.dropna(subset=['totalRevenue', 'greenRevenuePercent'])
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x='greenRevenuePercent',
            y='totalRevenue',
            color='pure_play_flag',
            size='totalRevenue',
            hover_name='counterparty_name',
            title='Total Revenue vs Green Revenue Percentage',
            labels={
                'greenRevenuePercent': 'Green Revenue Percentage',
                'totalRevenue': 'Total Revenue',
                'pure_play_flag': 'Pure Play Status'
            },
            color_discrete_map={'Y': '#2ecc71', 'N': '#e74c3c'},
            opacity=0.7,
            log_y=True  # Use log scale for revenue
        )
        
        # Add line at 50% for pure play threshold
        fig.add_vline(
            x=50,
            line_width=2,
            line_dash="dash",
            line_color="black",
            annotation_text="Pure Play Threshold (50%)",
            annotation_position="top"
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Green Revenue Percentage',
            yaxis_title='Total Revenue (log scale)',
            legend_title='Pure Play Status',
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            font=dict(size=12)
        )
        
        return fig
    
    def display(self):
        """
        Display all metrics and visualizations
        """
        try:
            st.markdown('<div class="section-header"><h2>Metrics and Statistics</h2></div>', unsafe_allow_html=True)
            
            # Add information about data processing
            if self.data_processor.green_revenue_count == 0:
                st.warning("No GREEN_REVENUE data was generated. Please check your data files.")
                return
            
            # Create two columns for the venn diagram and green revenue distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dataset Overlaps")
                try:
                    venn_diagram = self.create_venn_diagram()
                    st.image(f"data:image/png;base64,{venn_diagram}", use_column_width=True)
                except Exception as e:
                    st.error(f"Error displaying Venn diagram: {e}")
            
            with col2:
                st.subheader("Green Revenue Distribution")
                try:
                    if len(self.data_processor.green_revenue) > 0 and 'greenRevenuePercent' in self.data_processor.green_revenue.columns:
                        fig_green_dist = self.create_green_revenue_distribution()
                        st.plotly_chart(fig_green_dist, use_container_width=True)
                    else:
                        st.warning("Insufficient data for Green Revenue Distribution")
                except Exception as e:
                    st.error(f"Error displaying Green Revenue Distribution: {e}")
            
            # Create two columns for country distribution and pure play by year
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top Countries")
                try:
                    if len(self.data_processor.green_revenue) > 0 and 'country_code' in self.data_processor.green_revenue.columns:
                        fig_country = self.create_country_distribution()
                        st.plotly_chart(fig_country, use_container_width=True)
                    else:
                        st.warning("Insufficient data for Country Distribution")
                except Exception as e:
                    st.error(f"Error displaying Country Distribution: {e}")
            
            with col2:
                st.subheader("Companies by Year and Pure Play Status")
                try:
                    if (len(self.data_processor.green_revenue) > 0 and 
                        'year' in self.data_processor.green_revenue.columns and 
                        'pure_play_flag' in self.data_processor.green_revenue.columns):
                        fig_year = self.create_pure_play_by_year()
                        st.plotly_chart(fig_year, use_container_width=True)
                    else:
                        st.warning("Insufficient data for Year and Pure Play Status chart")
                except Exception as e:
                    st.error(f"Error displaying Year and Pure Play Status chart: {e}")
            
            # Display revenue vs green percentage as full width
            st.subheader("Total Revenue vs Green Revenue Percentage")
            try:
                if (len(self.data_processor.green_revenue) > 0 and 
                    'totalRevenue' in self.data_processor.green_revenue.columns and 
                    'greenRevenuePercent' in self.data_processor.green_revenue.columns and
                    'pure_play_flag' in self.data_processor.green_revenue.columns):
                    fig_revenue = self.create_revenue_by_green_pct()
                    st.plotly_chart(fig_revenue, use_container_width=True)
                else:
                    st.warning("Insufficient data for Revenue vs Green Revenue Percentage chart")
            except Exception as e:
                st.error(f"Error displaying Revenue vs Green Revenue Percentage chart: {e}")
        
        except Exception as e:
            st.error(f"Error displaying Metrics Section: {e}")
            st.warning("Please check your data and ensure it's properly formatted according to the requirements.")
