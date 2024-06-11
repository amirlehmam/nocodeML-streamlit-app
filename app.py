import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import base64
from streamlit_navigation_bar import navigation_bar

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

def advanced_eda_indicators():
    from scripts.advanced_eda_indicators import run_advanced_eda_indicators
    run_advanced_eda_indicators()

def optimal_win_ranges():
    from scripts.optimal_win_ranges import run_optimal_win_ranges
    run_optimal_win_ranges()

def model_percentage_away():
    from scripts.model_percentage_away import run_model_percentage_away
    run_model_percentage_away()

def specific_model_focus():
    from scripts.specific_model_focus import run_specific_model_focus
    run_specific_model_focus()

def advanced_eda_specific_model():
    from scripts.advanced_eda_specific_model import run_advanced_eda_specific_model
    run_advanced_eda_specific_model()

def win_ranges_specific_model():
    from scripts.win_ranges_specific_model import run_win_ranges_specific_model
    run_win_ranges_specific_model()

def advanced_trading_dashboard():
    from scripts.model_dashboard import run_model_dashboard
    run_model_dashboard()

def advanced_model_exploration():
    from scripts.advanced_model_exploration import run_advanced_model_exploration
    run_advanced_model_exploration()

# Mapping page names to functions
page_dict = {
    'Overview': overview,
    'Data Ingestion and Preparation': data_ingestion,
    'Advanced Trading Dashboard': advanced_trading_dashboard,
    'Advanced Model Exploration': advanced_model_exploration,
    'Advanced EDA on Indicators': advanced_eda_indicators,
    'Optimal Win Ranges': optimal_win_ranges,
    'Model on % Away Indicators': model_percentage_away,
    'Specific Model Focus': specific_model_focus,
    'Advanced EDA on Specific Model': advanced_eda_specific_model,
    'Win Ranges for Specific Model': win_ranges_specific_model
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

        # Navigation bar
        selected = navigation_bar(
            "Select a page",
            ["Overview", "Data Ingestion and Preparation", "Advanced Trading Dashboard", "Advanced Model Exploration",
             "Advanced EDA on Indicators", "Optimal Win Ranges", "Model on % Away Indicators", "Specific Model Focus",
             "Advanced EDA on Specific Model", "Win Ranges for Specific Model"]
        )

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
