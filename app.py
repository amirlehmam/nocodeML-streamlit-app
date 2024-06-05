import streamlit as st
from pathlib import Path
import os

# Set page config
st.set_page_config(
    page_title="nocodeML",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    /* Main Layout */
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #262730;
    }

    /* Title */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
        color: #FAFAFA;
    }

    /* Headers */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #FAFAFA;
    }

    /* Buttons */
    .stButton button {
        background-color: #1E88E5;
        color: #FAFAFA;
        border-radius: 5px;
    }
    
    /* Dropdown */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #262730;
        color: #FAFAFA;
    }

    /* Text Inputs */
    .stTextInput div[data-baseweb="input"] > div {
        background-color: #262730;
        color: #FAFAFA;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the logo
logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
if os.path.exists(logo_path):
    st.image(logo_path, width=1000)
else:
    st.warning("Logo file not found!")

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = [
    "Overview",
    "Data Ingestion and Preparation",
    "Advanced EDA on Indicators",
    "Optimal Win Ranges",
    "Model on % Away Indicators",
    "Specific Model Focus",
    "Advanced EDA on Specific Model",
    "Win Ranges for Specific Model"
]
page = st.sidebar.selectbox("Choose a page", pages)

if page == "Overview":
    st.header("Project Overview")
    st.write("""
    ## Welcome to the nocodeML Algorithmic Trading Optimization App

    This app is designed to provide an intuitive and powerful interface for optimizing algorithmic trading strategies. Using advanced machine learning and data analysis techniques, you can explore, visualize, and fine-tune your trading algorithms with ease. Here's what you can do with this app:

    ### Features:
    
    - **Data Ingestion and Preparation**: Seamlessly import and clean your trading data, preparing it for in-depth analysis.
    - **Advanced EDA on Indicators**: Perform detailed Exploratory Data Analysis on various trading indicators to uncover valuable insights.
    - **Optimal Win Ranges**: Identify the most profitable trading ranges using sophisticated statistical techniques.
    - **Model on % Away Indicators**: Develop and optimize models based on percentage away indicators to enhance your trading strategies.
    - **Specific Model Focus**: Dive deep into specific models to understand their performance and behavior.
    - **Advanced EDA on Specific Model**: Conduct comprehensive EDA on specific models to gain a deeper understanding of their characteristics.
    - **Win Ranges for Specific Model**: Determine the optimal win ranges for your specific trading models to maximize profitability.

    ### Getting Started:

    Use the sidebar to navigate through the different sections of the app. Each section provides tools and visualizations to help you optimize your trading strategies effectively. Start by ingesting and preparing your data, then move on to exploratory data analysis and model optimization.

    We hope this app helps you achieve your trading goals with the power of machine learning and data science!

    Happy Trading!
    """)

elif page == "Data Ingestion and Preparation":
    from scripts.data_ingestion_preparation import run_data_ingestion_preparation
    run_data_ingestion_preparation()

elif page == "Advanced EDA on Indicators":
    from scripts.advanced_eda_indicators import run_advanced_eda_indicators
    run_advanced_eda_indicators()

elif page == "Optimal Win Ranges":
    from scripts.optimal_win_ranges import run_optimal_win_ranges
    run_optimal_win_ranges()

elif page == "Model on % Away Indicators":
    from scripts.model_percentage_away import run_model_percentage_away
    run_model_percentage_away()

elif page == "Specific Model Focus":
    from scripts.specific_model_focus import run_specific_model_focus
    run_specific_model_focus()

elif page == "Advanced EDA on Specific Model":
    from scripts.advanced_eda_specific_model import run_advanced_eda_specific_model
    run_advanced_eda_specific_model()

elif page == "Win Ranges for Specific Model":
    from scripts.win_ranges_specific_model import run_win_ranges_specific_model
    run_win_ranges_specific_model()