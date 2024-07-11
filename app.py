import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import base64
from streamlit_navigation_bar import st_navbar

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

with open('./config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

authenticator.login()

def overview():
    st.write(f'Welcome **{st.session_state["name"]}**')
    st.write("""
    ### nocodeML Algorithmic Trading Optimization

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

def data_ingestion():
    from scripts.data_ingestion_preparation import run_data_ingestion_preparation
    run_data_ingestion_preparation()

def advanced_trading_dashboard():
    from scripts.model_dashboard import run_model_dashboard
    run_model_dashboard()

def advanced_model_exploration():
    from scripts.advanced_model_exploration import run_advanced_model_exploration
    run_advanced_model_exploration()

def strategy_performance():
    from scripts.strategy_performance import run_strategy_performance
    run_strategy_performance()

def trading_hours():
    from scripts.trading_hours_analyzer import main
    main() 

# Mapping page names to functions
page_dict = {
    'Overview': overview,
    'Data Ingestion': data_ingestion,
    'Trading Dashboard': advanced_trading_dashboard,
    'Model Exploration': advanced_model_exploration,
    'Strategy Performance': strategy_performance,
    'Trading Hours': trading_hours
}

def main():
    if st.session_state["authentication_status"]:
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
                margin-top: 5px;
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

            /* Navigation Bar */
            .navbar {
                display: flex;
                background-color: royalblue;
                justify-content: flex-start;
            }

            .navbar-item {
                color: white;
                padding: 14px;
                cursor: pointer;
            }

            .navbar-item:hover {
                background-color: #1565C0;
            }

            .navbar-item.active {
                background-color: white;
                color: black;
                font-weight: normal;
                padding: 14px;
            }

            </style>
            <script>
                function toggleContent(id) {
                    var content = document.getElementById(id);
                    if (content.style.display === "block") {
                        content.style.display = "none";
                    } else {
                        content.style.display = "block";
                    }
                }
            </script>
            """,
            unsafe_allow_html=True
        )

        # Navigation bar
        pages = ['Overview', 'Data Ingestion', 'Trading Dashboard', 'Model Exploration', 'Strategy Performance', 'Trading Hours']
        selected = st_navbar(
            pages,
            styles={
                "nav": {
                    "background-color": "royalblue",
                    "justify-content": "left",
                },
                "img": {
                    "padding-right": "14px",
                },
                "span": {
                    "color": "white",
                    "padding": "14px",
                },
                "active": {
                    "background-color": "white",
                    "color": "black",
                    "font-weight": "normal",
                    "padding": "14px",
                }
            }
        )

        # Display the logo
        logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
        if (os.path.exists(logo_path)):
            st.markdown(
                f"""
                <div class="center-logo">
                    <img src="data:image/png;base64,{load_image(logo_path)}" width="280">
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("Logo file not found!")

        # Call the function based on the selected page
        page_dict[selected]()

        authenticator.logout()

        # Footer
        st.markdown(
            """
            <div class='footer'>
                <p>&copy; 2024 nocodeML. All rights reserved.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')

if __name__ == "__main__":
    main()
