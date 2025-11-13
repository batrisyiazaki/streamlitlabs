import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
import numpy as np

# --- 1. CONFIGURATION AND DATA LOADING ---
st.set_page_config(
    page_title="Iris Data Visualization Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use st.cache_data to load the dataset only once
@st.cache_data
def load_data():
    """Loads the Iris dataset from scikit-learn."""
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['species_name'] = df['target'].apply(lambda x: iris.target_names[x])
    return df

df = load_data()

# --- 2. THEME AND LAYOUT ---
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6; 
    }
    .stApp {
        background-color: #f0f2f6;
    }
    h1 {
        color: #5A29E8;
        font-family: 'Georgia', serif;
        text-align: center;
        padding-top: 10px;
    }
    .stMetric {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌸 Interactive Iris Flower Dataset Dashboard")

# --- 3. SIDEBAR FILTER (Requirement 1) ---
st.sidebar.header("Filter Options")

# Get unique species names for the selectbox
species_options = df['species_name'].unique().tolist()
species_options.insert(0, 'All Species') # Add 'All' option

# Sidebar Selectbox Filter
selected_species = st.sidebar.selectbox(
    "Select Species for Analysis",
    options=species_options
)

# Apply filtering
if selected_species != 'All Species':
    df_filtered = df[df['species_name'] == selected_species]
    st.header(f"Analysis for: {selected_species.upper()}")
else:
    df_filtered = df
    st.header("Analysis for: ALL Species")

# --- 4. DATA SUMMARY (Requirement 3) ---
st.subheader("Data Overview and Key Metrics")

# Calculate summary statistics for the filtered data
mean_sepal_length = df_filtered['sepal length (cm)'].mean()
median_petal_width = df_filtered['petal width (cm)'].median()
data_rows = len(df_filtered)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Total Data Points",
        value=f"{data_rows}"
    )

with col2:
    st.metric(
        label="Avg Sepal Length (cm)",
        value=f"{mean_sepal_length:.2f}",
        delta=f"Median: {df_filtered['sepal length (cm)'].median():.2f}"
    )

with col3:
    st.metric(
        label="Median Petal Width (cm)",
        value=f"{median_petal_width:.2f}",
        delta=f"Max: {df_filtered['petal width (cm)'].max():.2f}"
    )

st.divider()

# --- 5. VISUALIZATIONS (Requirement 2: Two Plot Types) ---

st.subheader("Visual Analysis")

# Visualization 1: Scatter Plot (Relationship)
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### Feature Relationship: Sepal vs. Petal Dimensions")
    # Interactive scatter plot
    scatter_fig = px.scatter(
        df_filtered,
        x="sepal length (cm)",
        y="petal length (cm)",
        color="species_name",
        hover_data=['petal width (cm)'],
        title="Sepal Length vs. Petal Length by Species",
        template="plotly_white",
        color_discrete_map={
            'setosa': '#4A90E2',
            'versicolor': '#50E3C2',
            'virginica': '#FF4B4B'
        }
    )
    scatter_fig.update_layout(height=400)
    st.plotly_chart(scatter_fig, use_container_width=True)

# Visualization 2: Histogram (Distribution)
with col_b:
    st.markdown("#### Distribution of Sepal Width")
    
    # Histogram filter to select which column to display
    hist_column = st.selectbox(
        "Select Feature for Histogram",
        options=['sepal width (cm)', 'petal width (cm)', 'petal length (cm)'],
        index=0,
        label_visibility="collapsed"
    )

    hist_fig = px.histogram(
        df_filtered,
        x=hist_column,
        color="species_name",
        marginal="box", # Adds a box plot for distribution detail
        title=f"Distribution of {hist_column.title()}",
        template="plotly_white",
        opacity=0.8,
        color_discrete_map={
            'setosa': '#4A90E2',
            'versicolor': '#50E3C2',
            'virginica': '#FF4B4B'
        }
    )
    hist_fig.update_layout(height=400)
    st.plotly_chart(hist_fig, use_container_width=True)

st.divider()

# Data table display (Alternative Summary)
st.subheader("Raw Data (Filtered)")
st.dataframe(df_filtered, use_container_width=True)