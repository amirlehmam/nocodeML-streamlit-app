import streamlit as st
from hydralit import HydraApp
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import base64

# Function to load and encode image
def load_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    return encoded_image

# Function to configure authentication
def configure_authentication():
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['pre-authorized']
    )
    return authenticator

# Functions for each app
def overview_app():
    st.write(f'Welcome **{st.session_state["name"]}**')
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

def data_ingestion_app():
    from scripts.data_ingestion_preparation import run_data_ingestion_preparation
    run_data_ingestion_preparation()

def advanced_eda_app():
    from scripts.advanced_eda_indicators import run_advanced_eda_indicators
    run_advanced_eda_indicators()

def optimal_win_ranges_app():
    from scripts.optimal_win_ranges import run_optimal_win_ranges
    run_optimal_win_ranges()

def model_percentage_away_app():
    from scripts.model_percentage_away import run_model_percentage_away
    run_model_percentage_away()

def specific_model_focus_app():
    from scripts.specific_model_focus import run_specific_model_focus
    run_specific_model_focus()

def advanced_eda_specific_model_app():
    from scripts.advanced_eda_specific_model import run_advanced_eda_specific_model
    run_advanced_eda_specific_model()

def win_ranges_specific_model_app():
    from scripts.win_ranges_specific_model import run_win_ranges_specific_model
    run_win_ranges_specific_model()

def advanced_trading_dashboard_app():
    from scripts.model_dashboard import run_model_dashboard
    run_model_dashboard()

def advanced_model_exploration_app():
    from scripts.advanced_model_exploration import run_advanced_model_exploration
    run_advanced_model_exploration()

def main():
    authenticator = configure_authentication()
    authenticator.login()

    if st.session_state["authentication_status"]:
        app = HydraApp(title='nocodeML', hide_streamlit_markers=True, use_navbar=True)

        @app.addapp(is_home=True)
        def overview():
            overview_app()

        @app.addapp(icon="📂")
        def data_ingestion():
            data_ingestion_app()

        @app.addapp(icon="📈")
        def advanced_eda():
            advanced_eda_app()

        @app.addapp(icon="🎯")
        def optimal_win_ranges():
            optimal_win_ranges_app()

        @app.addapp(icon="📊")
        def model_percentage_away():
            model_percentage_away_app()

        @app.addapp(icon="🔍")
        def specific_model_focus():
            specific_model_focus_app()

        @app.addapp(icon="📉")
        def advanced_eda_specific_model():
            advanced_eda_specific_model_app()

        @app.addapp(icon="🏆")
        def win_ranges_specific_model():
            win_ranges_specific_model_app()

        @app.addapp(icon="📊")
        def advanced_trading_dashboard():
            advanced_trading_dashboard_app()

        @app.addapp(icon="⚙️")
        def advanced_model_exploration():
            advanced_model_exploration_app()

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
