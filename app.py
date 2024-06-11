import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
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

def top_navbar():
    st.markdown(
        """
        <style>
        .navbar {
            overflow: hidden;
            background-color: #333;
            position: -webkit-sticky; /* Safari */
            position: sticky;
            top: 0;
            width: 100%;
            z-index: 100;
        }

        .navbar a {
            float: left;
            display: block;
            color: #f2f2f2;
            text-align: center;
            padding: 14px 20px;
            text-decoration: none;
            font-size: 17px;
        }

        .navbar a:hover {
            background-color: #ddd;
            color: black;
        }

        .navbar a.active {
            background-color: #1E88E5;
            color: white;
        }
        </style>

        <div class="navbar">
            <a href="/?page=overview" class="{active_overview}">Overview</a>
            <a href="/?page=data_ingestion" class="{active_data_ingestion}">Data Ingestion and Preparation</a>
            <a href="/?page=dashboard" class="{active_dashboard}">Advanced Trading Dashboard</a>
            <a href="/?page=model_exploration" class="{active_model_exploration}">Advanced Model Exploration</a>
            <a href="/?page=eda_indicators" class="{active_eda_indicators}">Advanced EDA on Indicators</a>
            <a href="/?page=win_ranges" class="{active_win_ranges}">Optimal Win Ranges</a>
            <a href="/?page=percentage_away" class="{active_percentage_away}">Model on % Away Indicators</a>
            <a href="/?page=model_focus" class="{active_model_focus}">Specific Model Focus</a>
            <a href="/?page=eda_specific_model" class="{active_eda_specific_model}">Advanced EDA on Specific Model</a>
            <a href="/?page=win_ranges_specific_model" class="{active_win_ranges_specific_model}">Win Ranges for Specific Model</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

def main():
    query_params = st.experimental_get_query_params()
    page = query_params.get("page", ["overview"])[0]

    top_navbar()

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
        if os.path.exists(logo_path):
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

        # Initialize session state if not already done
        if 'page' not in st.session_state:
            st.session_state.page = "overview"

        if page == "overview":
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

        elif page == "data_ingestion":
            from scripts.data_ingestion_preparation import run_data_ingestion_preparation
            run_data_ingestion_preparation()

        elif page == "dashboard":
            from scripts.model_dashboard import run_model_dashboard
            run_model_dashboard()

        elif page == "model_exploration":
            from scripts.advanced_model_exploration import run_advanced_model_exploration
            run_advanced_model_exploration()

        elif page == "eda_indicators":
            from scripts.advanced_eda_indicators import run_advanced_eda_indicators
            run_advanced_eda_indicators()

        elif page == "win_ranges":
            from scripts.optimal_win_ranges import run_optimal_win_ranges
            run_optimal_win_ranges()

        elif page == "percentage_away":
            from scripts.model_percentage_away import run_model_percentage_away
            run_model_percentage_away()

        elif page == "model_focus":
            from scripts.specific_model_focus import run_specific_model_focus
            run_specific_model_focus()

        elif page == "eda_specific_model":
            from scripts.advanced_eda_specific_model import run_advanced_eda_specific_model
            run_advanced_eda_specific_model()

        elif page == "win_ranges_specific_model":
            from scripts.win_ranges_specific_model import run_win_ranges_specific_model
            run_win_ranges_specific_model()

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
