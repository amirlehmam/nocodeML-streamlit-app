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

# Custom CSS for enhanced design
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

    /* Center logo */
    .center-logo {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the logo in the center of the screen
logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
if os.path.exists(logo_path):
    st.markdown(f"<div class='center-logo'><img src='data:image/png;base64,{st.image(logo_path, width=200, use_column_width='auto')._repr_html_()}'></div>", unsafe_allow_html=True)
else:
    st.warning("Logo file not found!")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    [
        "Overview",
        "Data Ingestion and Preparation",
        "Advanced EDA on Indicators",
        "Optimal Win Ranges",
        "Model on % Away Indicators",
        "Specific Model Focus",
        "Advanced EDA on Specific Model",
        "Win Ranges for Specific Model"
    ]
)

if page == "Overview":
    st.write("""
    ## Welcome to the nocodeML Algorithmic Trading Optimization App

    This application offers a robust and user-friendly platform for enhancing your algorithmic trading strategies through advanced machine learning and data analysis techniques. Explore, visualize, and refine your trading algorithms with our comprehensive suite of tools.

    ### Key Features:
    
    - **Data Ingestion and Preparation**: 
      - Effortlessly import your raw trading data.
      - Utilize our cleaning and parsing utilities to ensure your data is ready for analysis.
      - Save the prepared data for further exploration and modeling.

    - **Advanced Exploratory Data Analysis (EDA) on Indicators**:
      - Perform in-depth analysis of various trading indicators.
      - Generate insightful visualizations to understand trends, correlations, and anomalies.
      - Use interactive plots to drill down into specific data points and uncover hidden patterns.

    - **Optimal Win Ranges Identification**:
      - Apply sophisticated statistical techniques to determine the most profitable trading ranges.
      - Visualize the optimal win ranges to enhance your trading decisions.
      - Summarize the findings to quickly identify key insights.

    - **Model Development on % Away Indicators**:
      - Build and optimize predictive models based on percentage away indicators.
      - Utilize machine learning algorithms to forecast market movements.
      - Evaluate model performance with a suite of metrics to ensure accuracy and reliability.

    - **Focused Analysis on Specific Models**:
      - Conduct a deep dive into the performance of specific models.
      - Analyze model behavior under various market conditions.
      - Fine-tune model parameters to achieve optimal results.

    - **Advanced EDA on Specific Models**:
      - Gain a comprehensive understanding of your models through detailed EDA.
      - Visualize the interplay between different model features and outcomes.
      - Use advanced statistical tests to validate model assumptions and hypotheses.

    - **Win Ranges Analysis for Specific Models**:
      - Determine the optimal win ranges tailored to your specific models.
      - Enhance model performance by focusing on the most profitable market conditions.
      - Visualize and interpret the results to make informed trading decisions.

    ### Getting Started:

    1. **Navigate through the Sidebar**: Use the sidebar to access different sections of the app.
    2. **Ingest and Prepare Data**: Begin by uploading your raw data files and preparing them for analysis.
    3. **Perform EDA**: Explore your data and trading indicators with our advanced EDA tools.
    4. **Identify Optimal Win Ranges**: Use statistical methods to find the best trading ranges.
    5. **Develop and Optimize Models**: Build, train, and evaluate machine learning models to improve your trading strategies.
    6. **Deep Dive into Specific Models**: Analyze the performance and characteristics of your models in detail.

    Our goal is to empower you with the tools and insights needed to achieve excellence in algorithmic trading. Harness the power of machine learning and data science to unlock new opportunities and enhance your trading performance.

    ### Happy Trading!

    *nocodeML 2024*
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
