import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import base64
from hydralit import HydraApp
from hydralit_components import HydraHeadApp

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

class OverviewApp(HydraHeadApp):
    def run(self):
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

class DataIngestionApp(HydraHeadApp):
    def run(self):
        from scripts.data_ingestion_preparation import run_data_ingestion_preparation
        run_data_ingestion_preparation()

class AdvancedEDAIndicatorsApp(HydraHeadApp):
    def run(self):
        from scripts.advanced_eda_indicators import run_advanced_eda_indicators
        run_advanced_eda_indicators()

class OptimalWinRangesApp(HydraHeadApp):
    def run(self):
        from scripts.optimal_win_ranges import run_optimal_win_ranges
        run_optimal_win_ranges()

class ModelPercentageAwayApp(HydraHeadApp):
    def run(self):
        from scripts.model_percentage_away import run_model_percentage_away
        run_model_percentage_away()

class SpecificModelFocusApp(HydraHeadApp):
    def run(self):
        from scripts.specific_model_focus import run_specific_model_focus
        run_specific_model_focus()

class AdvancedEDASpecificModelApp(HydraHeadApp):
    def run(self):
        from scripts.advanced_eda_specific_model import run_advanced_eda_specific_model
        run_advanced_eda_specific_model()

class WinRangesSpecificModelApp(HydraHeadApp):
    def run(self):
        from scripts.win_ranges_specific_model import run_win_ranges_specific_model
        run_win_ranges_specific_model()

class AdvancedTradingDashboardApp(HydraHeadApp):
    def run(self):
        from scripts.model_dashboard import run_model_dashboard
        run_model_dashboard()

class AdvancedModelExplorationApp(HydraHeadApp):
    def run(self):
        from scripts.advanced_model_exploration import run_advanced_model_exploration
        run_advanced_model_exploration()

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

        app = HydraApp(title="nocodeML", favicon="📈", hide_streamlit_markers=True)

        app.add_app("Overview", icon="🏠", app=OverviewApp())
        app.add_app("Data Ingestion and Preparation", icon="📂", app=DataIngestionApp())
        app.add_app("Advanced Trading Dashboard", icon="📈", app=AdvancedTradingDashboardApp())
        app.add_app("Advanced Model Exploration", icon="⚙️", app=AdvancedModelExplorationApp())
        app.add_app("Advanced EDA on Indicators", icon="📊", app=AdvancedEDAIndicatorsApp())
        app.add_app("Optimal Win Ranges", icon="🎯", app=OptimalWinRangesApp())
        app.add_app("Model on % Away Indicators", icon="📈", app=ModelPercentageAwayApp())
        app.add_app("Specific Model Focus", icon="🔍", app=SpecificModelFocusApp())
        app.add_app("Advanced EDA on Specific Model", icon="📉", app=AdvancedEDASpecificModelApp())
        app.add_app("Win Ranges for Specific Model", icon="🏆", app=WinRangesSpecificModelApp())

        # Run HydraApp
        app.run()

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
