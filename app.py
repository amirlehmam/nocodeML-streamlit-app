import streamlit as st
import streamlit_authenticator as stauth
import sqlite3
import pandas as pd
from pathlib import Path
import os
import base64

# Set page config
st.set_page_config(
    page_title="nocodeML",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to load and encode image
def load_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    return encoded_image

# Custom CSS for enhanced design
st.markdown(
    """
    <style>
    /* Main Layout */
    .main {
        background: url('https://www.transparenttextures.com/patterns/black-linen.png');
        color: #FAFAFA;
        font-family: 'Arial', sans-serif;
    }

    /* Sidebar */
    .css-1d391kg {
        background: url('https://www.transparenttextures.com/patterns/black-linen.png');
        color: #FAFAFA;
    }

    /* Sidebar button style */
    .sidebar-button {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        padding: 10px 20px;
        margin: 5px 0;
        font-size: 18px;
        font-weight: bold;
        color: #FAFAFA;
        background-color: #1E88E5;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        width: 100%;
        text-align: left;
    }
    
    .sidebar-button:hover {
        background-color: #1565C0;
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

    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: #FAFAFA;
        text-align: center;
        padding: 10px 0;
    }

    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: #1E88E5;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px 0;
        position: absolute;
        z-index: 1;
        bottom: 125%; 
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Collapsible */
    .collapsible {
        background-color: #1E88E5;
        color: white;
        cursor: pointer;
        padding: 10px;
        width: 100%;
        border: none;
        text-align: left;
        outline: none;
        font-size: 15px;
        border-radius: 5px;
        margin-bottom: 5px;
    }

    .active, .collapsible:hover {
        background-color: #1565C0;
    }

    .content {
        padding: 0 18px;
        display: none;
        overflow: hidden;
        background-color: #262730;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to fetch user credentials from the database
def fetch_user_credentials():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT username, name, password FROM users')
    users = c.fetchall()
    conn.close()

    credentials = {
        'usernames': {},
    }

    for username, name, password in users:
        credentials['usernames'][username] = {"name": name, "password": password}

    return credentials

# Fetch user credentials
credentials = fetch_user_credentials()

# Debugging: Print the credentials to verify their structure
st.write("Credentials loaded:")
st.write(credentials)

# Define the cookie name and signature key for the authenticator
cookie_name = 'nocodeML_cookie'
signature_key = 'some_random_key'  # You should use a more secure key

# Create an authenticator object
authenticator = stauth.Authenticate(
    credentials,
    cookie_name,
    signature_key,
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')

# If login is successful
if authentication_status:
    # Display the logo
    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=200)
    else:
        st.warning("Logo file not found!")

    # Sidebar for navigation with icons
    st.sidebar.title("Navigation")

    def nav_button(label, page_name, icon):
        if st.sidebar.button(f"{icon} {label}"):
            st.session_state.page = page_name

    nav_button("Overview", "Overview", "🏠")
    nav_button("Data Ingestion and Preparation", "Data Ingestion and Preparation", "📂")
    nav_button("Advanced EDA on Indicators", "Advanced EDA on Indicators", "📊")
    nav_button("Optimal Win Ranges", "Optimal Win Ranges", "🎯")
    nav_button("Model on % Away Indicators", "Model on % Away Indicators", "📈")
    nav_button("Specific Model Focus", "Specific Model Focus", "🔍")
    nav_button("Advanced EDA on Specific Model", "Advanced EDA on Specific Model", "📉")
    nav_button("Win Ranges for Specific Model", "Win Ranges for Specific Model", "🏆")

    # Initialize session state if not already done
    if 'page' not in st.session_state:
        st.session_state.page = "Overview"

    page = st.session_state.page

    if page == "Overview":
        st.write("""
        ###  nocodeML Algorithmic Trading Optimization

        #### Streamlined for Precision and Performance

        ---

        **Key Features:**

        **Data Ingestion and Preparation:**
        - Import raw trading data effortlessly.
        - Utilize cleaning and parsing utilities for ready-to-analyze data.
        - Save prepared data for exploration and modeling.

        **Advanced EDA on Indicators:**
        - Perform in-depth analysis of trading indicators.
        - Generate visualizations to understand trends, correlations, and anomalies.
        - Use interactive plots to uncover hidden patterns.

        **Optimal Win Ranges Identification:**
        - Apply statistical techniques to determine profitable trading ranges.
        - Visualize win ranges to enhance trading decisions.
        - Summarize findings for quick insights.

        **Model Development on % Away Indicators:**
        - Build and optimize predictive models based on % away indicators.
        - Utilize machine learning algorithms to forecast market movements.
        - Evaluate model performance with comprehensive metrics.

        **Focused Analysis on Specific Models:**
        - Conduct deep dives into specific model performances.
        - Analyze model behavior under various market conditions.
        - Fine-tune parameters for optimal results.

        **Advanced EDA on Specific Models:**
        - Gain detailed understanding of models through comprehensive EDA.
        - Visualize feature interactions and outcomes.
        - Validate model assumptions with advanced statistical tests.

        **Win Ranges Analysis for Specific Models:**
        - Determine optimal win ranges tailored to specific models.
        - Enhance performance by focusing on profitable market conditions.
        - Visualize results for informed trading decisions.

        ---

        **Getting Started:**
        1. **Navigate through the Sidebar:** Access different sections of the app.
        2. **Ingest and Prepare Data:** Upload and prepare raw data files.
        3. **Perform EDA:** Explore data and trading indicators with advanced tools.
        4. **Identify Optimal Win Ranges:** Use statistical methods to find the best trading ranges.
        5. **Develop and Optimize Models:** Build, train, and evaluate machine learning models.
        6. **Deep Dive into Specific Models:** Analyze model performance and characteristics.

        ---

        Harness machine learning and data science to unlock new opportunities and enhance trading performance. 

        **Two Plums for One**
                    
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

    # Add logout button
    if st.sidebar.button('Logout'):
        authenticator.logout('Logout', 'sidebar')

    # Footer
    st.markdown(
        """
        <div class='footer'>
            <p>&copy; 2024 nocodeML. All rights reserved.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# If login fails
elif authentication_status is False:
    st.error('Username or password is incorrect')

# If login not attempted yet
elif authentication_status is None:
    st.warning('Please enter your username and password')
