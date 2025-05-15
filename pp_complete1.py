
requirement.txt
streamlit
pandas
numpy
matplotlib
seaborn
python-slickgrid
openpyxl

------------------------

# app.py

import streamlit as st
from utils.data_loader import load_pcaf_data, load_llm_generated_data, load_sff_data
from utils.data_processor import process_pcaf_data, join_datasets, compare_datasets
from utils.visualization import (
    plot_venn_diagram, plot_metrics, plot_green_revenue_distribution,
    plot_total_revenue_by_country, plot_year_wise_green_revenue,
    plot_pure_play_flag_distribution_by_country, plot_top_companies_by_green_revenue,
    plot_themes_distribution, plot_revenue_vs_green_revenue, plot_bic_code_distribution
)

# Load data
pcaf_data = load_pcaf_data('data/group_client_coverage_dec24.xlsx')
llm_generated_data = load_llm_generated_data('data/llm_generated.csv')
sff_data = load_sff_data('data/Mar_PP_list_vF.xlsx')

# Process data
unique_pcaf_data = process_pcaf_data(pcaf_data)
green_revenue_data = join_datasets(unique_pcaf_data, llm_generated_data)

# Compare datasets
overlap_data, identified_clients_data, un_identified_clients_data = compare_datasets(green_revenue_data, sff_data)

# Streamlit app
st.title('Sustainable Data Dashboard')

# Section 1: Metrics and Statistics
st.header('Metrics and Statistics')

# Create columns for visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader('Venn Diagram')
    plot_venn_diagram(green_revenue_data, sff_data)

    st.subheader('Green Revenue Distribution')
    plot_green_revenue_distribution(green_revenue_data)

    st.subheader('Year-wise Green Revenue')
    plot_year_wise_green_revenue(green_revenue_data)

    st.subheader('Top Companies by Green Revenue')
    plot_top_companies_by_green_revenue(green_revenue_data)

with col2:
    st.subheader('Pure Play Flag Distribution')
    plot_metrics(green_revenue_data, 'Pure Play Flag Distribution')

    st.subheader('Total Revenue by Country')
    plot_total_revenue_by_country(green_revenue_data)

    st.subheader('Pure Play Flag Distribution by Country')
    plot_pure_play_flag_distribution_by_country(green_revenue_data)

    st.subheader('Themes Distribution')
    plot_themes_distribution(sff_data)

# Create expanders for additional visualizations
with st.expander('Additional Visualizations'):
    st.subheader('Comparison of Total Revenue and Green Revenue Percentage')
    plot_revenue_vs_green_revenue(green_revenue_data)

    st.subheader('Distribution of Companies by BIC Code')
    plot_bic_code_distribution(green_revenue_data)

# Section 2: Companies classified as pure play in GREEN_REVENUE (>=50%)
st.header('Companies that are classified as pure play in GREEN_REVENUE (>=50%)')
pure_play_data = green_revenue_data[green_revenue_data['pure_play_flag'] == 'Y']

tab1, tab2, tab3 = st.tabs(['Overlap', 'Identified Clients', 'Un-Identified Clients'])

with tab1:
    st.subheader('The companies that exist in both GREEN_REVENUE and SFF_DATA')
    st.dataframe(overlap_data)

with tab2:
    st.subheader('The companies that exist in SFF_DATA but not in GREEN_REVENUE')
    st.dataframe(identified_clients_data)

with tab3:
    st.subheader('The companies that exist in GREEN_REVENUE but not in SFF_DATA')
    st.dataframe(un_identified_clients_data)

# Section 3: Companies not classified as pure play in GREEN_REVENUE (<50%)
st.header('Companies that are not classified as pure play in GREEN_REVENUE (<50%)')
non_pure_play_data = green_revenue_data[green_revenue_data['pure_play_flag'] == 'N']

tab4, tab5, tab6 = st.tabs(['Overlap', 'Identified Clients', 'Un-Identified Clients'])

with tab4:
    st.subheader('The companies that exist in both GREEN_REVENUE and SFF_DATA')
    st.dataframe(pd.merge(non_pure_play_data, sff_data, left_on='counterparty_name', right_on='Client Name', how='inner'))

with tab5:
    st.subheader('The companies that exist in SFF_DATA but not in GREEN_REVENUE')
    st.dataframe(pd.merge(sff_data, non_pure_play_data, left_on='Client Name', right_on='counterparty_name', how='left', indicator=True).query('_merge == "left_only"').drop(columns=['_merge']))

with tab6:
    st.subheader('The companies that exist in GREEN_REVENUE but not in SFF_DATA')
    st.dataframe(pd.merge(non_pure_play_data, sff_data, left_on='counterparty_name', right_on='Client Name', how='left', indicator=True).query('_merge == "left_only"').drop(columns=['_merge']))

-----------------

# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2
import plotly.express as px

def plot_green_revenue_distribution(green_revenue_data):
    plt.figure(figsize=(10, 6))
    sns.histplot(green_revenue_data['greenRevenuePercent'].dropna(), bins=30, kde=True)
    plt.title('Distribution of Green Revenue Percentage')
    plt.xlabel('Green Revenue Percentage')
    plt.ylabel('Frequency')
    plt.show()

def plot_total_revenue_by_country(green_revenue_data):
    plt.figure(figsize=(12, 8))
    green_revenue_data.groupby('country_code')['totalRevenue'].sum().sort_values(ascending=False).plot(kind='bar')
    plt.title('Total Revenue by Country')
    plt.xlabel('Country Code')
    plt.ylabel('Total Revenue')
    plt.show()

def plot_year_wise_green_revenue(green_revenue_data):
    plt.figure(figsize=(12, 8))
    green_revenue_data.groupby('year')['greenRevenuePercent'].mean().plot(kind='line', marker='o')
    plt.title('Year-wise Green Revenue Percentage')
    plt.xlabel('Year')
    plt.ylabel('Average Green Revenue Percentage')
    plt.show()

def plot_pure_play_flag_distribution_by_country(green_revenue_data):
    plt.figure(figsize=(12, 8))
    sns.countplot(data=green_revenue_data, x='country_code', hue='pure_play_flag')
    plt.title('Pure Play Flag Distribution by Country')
    plt.xlabel('Country Code')
    plt.ylabel('Count')
    plt.show()

def plot_top_companies_by_green_revenue(green_revenue_data):
    top_companies = green_revenue_data.sort_values(by='greenRevenuePercent', ascending=False).head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_companies, x='greenRevenuePercent', y='counterparty_name')
    plt.title('Top Companies by Green Revenue Percentage')
    plt.xlabel('Green Revenue Percentage')
    plt.ylabel('Company Name')
    plt.show()

def plot_themes_distribution(sff_data):
    plt.figure(figsize=(12, 8))
    sff_data['Themes'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Distribution of Themes in SFF_DATA')
    plt.show()

def plot_revenue_vs_green_revenue(green_revenue_data):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=green_revenue_data, x='totalRevenue', y='greenRevenuePercent')
    plt.title('Comparison of Total Revenue and Green Revenue Percentage')
    plt.xlabel('Total Revenue')
    plt.ylabel('Green Revenue Percentage')
    plt.show()

def plot_bic_code_distribution(green_revenue_data):
    plt.figure(figsize=(12, 8))
    green_revenue_data['bic_code'].value_counts().plot(kind='bar')
    plt.title('Distribution of Companies by BIC Code')
    plt.xlabel('BIC Code')
    plt.ylabel('Count')
    plt.show()

------------------------------------

# data_processor.py

import pandas as pd

def process_pcaf_data(pcaf_data):
    # Select unique records
    unique_pcaf_data = pcaf_data.drop_duplicates(subset=['cob_date', 'productype', 'legal_entity', 'counterparty_id', 'counterparty_name', 'parent_id', 'group_id', 'group_name', 'bic_code', 'country_code'])
    return unique_pcaf_data

def join_datasets(pcaf_data, llm_generated_data):
    # Join datasets
    pcaf_data['counterparty_name'] = pcaf_data['counterparty_name'].str.strip().str.lower()
    llm_generated_data['companyName'] = llm_generated_data['companyName'].str.strip().str.lower()

    green_revenue_data = pd.merge(pcaf_data, llm_generated_data, left_on='counterparty_name', right_on='companyName', how='left')

    # Create pure_play_flag
    green_revenue_data['pure_play_flag'] = green_revenue_data['greenRevenuePercent'].apply(lambda x: 'Y' if x >= 50 else 'N')

    return green_revenue_data

def compare_datasets(green_revenue_data, sff_data):
    # Compare datasets
    green_revenue_data['counterparty_name'] = green_revenue_data['counterparty_name'].str.strip().str.lower()
    sff_data['Client Name'] = sff_data['Client Name'].str.strip().str.lower()

    overlap_data = pd.merge(green_revenue_data, sff_data, left_on='counterparty_name', right_on='Client Name', how='inner')
    identified_clients_data = pd.merge(sff_data, green_revenue_data, left_on='Client Name', right_on='counterparty_name', how='left', indicator=True).query('_merge == "left_only"').drop(columns=['_merge'])
    un_identified_clients_data = pd.merge(green_revenue_data, sff_data, left_on='counterparty_name', right_on='Client Name', how='left', indicator=True).query('_merge == "left_only"').drop(columns=['_merge'])

    return overlap_data, identified_clients_data, un_identified_clients_data

-----------------

# data_loader.py

import pandas as pd

def load_pcaf_data(file_path):
    return pd.read_excel(file_path)

def load_llm_generated_data(file_path):
    return pd.read_csv(file_path)

def load_sff_data(file_path):
    return pd.read_excel(file_path)
