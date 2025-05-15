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
        Create a Venn diagram showing dataset overlaps based on the reference image
        """
        try:
            # Create a figure
            plt.figure(figsize=(10, 10))
            
            # Create sets for the three datasets
            # PCAF_DATA = circle on the left
            # LLM_GENERATED = circle on the top
            # SFF_DATA = circle on the bottom right
            
            # Create a custom Venn3 with just the circles
            from matplotlib_venn import Venn3
            from matplotlib.patches import Circle
            
            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Set up coordinates for three circles
            radius = 1.2
            circle1_center = (1.0, 1.5)  # PCAF_DATA (left)
            circle2_center = (2.5, 2.5)  # LLM_GENERATED (top)
            circle3_center = (3.0, 1.0)  # SFF_DATA (bottom right)
            
            # Create the circles
            circle1 = Circle(circle1_center, radius, alpha=0.4, edgecolor='blue', facecolor='#3498db', label='PCAF_DATA')
            circle2 = Circle(circle2_center, radius, alpha=0.4, edgecolor='green', facecolor='#2ecc71', label='LLM_GENERATED')
            circle3 = Circle(circle3_center, radius, alpha=0.4, edgecolor='red', facecolor='#e74c3c', label='SFF_DATA')
            
            # Add the circles to the plot
            ax.add_patch(circle1)
            ax.add_patch(circle2)
            ax.add_patch(circle3)
            
            # Add labels to the circles
            ax.annotate('PCAF_DATA', xy=(circle1_center[0]-0.5, circle1_center[1]), fontsize=12, weight='bold')
            ax.annotate('LLM_GENERATED', xy=(circle2_center[0]-0.7, circle2_center[1]+0.3), fontsize=12, weight='bold')
            ax.annotate('SFF_DATA', xy=(circle3_center[0]-0.2, circle3_center[1]-0.5), fontsize=12, weight='bold')
            
            # Dataset counts - get from data_processor
            dataset_info = self.data_processor.get_dataset_info()
            
            # Add labels for overlapping regions according to the hand-drawn image
            
            # Companies in PCAF but not in other datasets
            ax.annotate(f"Companies in\nPCAF_DATA\nBUT NOT PART\nOF SFF_DATA", 
                       xy=(circle1_center[0]-0.7, circle1_center[1]+0.5), fontsize=10)
            
            # Companies in SFF but not in others
            ax.annotate(f"Companies in\nSFF_DATA\nBUT NOT IN\nLLM_GENERATED", 
                       xy=(circle3_center[0]+0.2, circle3_center[1]-0.2), fontsize=10)
            
            # Companies common between LLM and SFF
            ax.annotate(f"Companies\nCOMMON\nTO BOTH SFF_DATA &\nLLM_GENERATED", 
                       xy=(2.8, 1.8), fontsize=10)
            
            # Set the limits of the plot
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 4)
            
            # Remove axis ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            
            # Remove axis border
            for spine in ax.spines.values():
                spine.set_visible(False)
            
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
