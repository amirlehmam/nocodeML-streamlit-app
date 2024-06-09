![nocodeML Banner](https://i.imgur.com/rMGU5VS.png)

# nocodeML

nocodeML is an advanced AI-powered helper agent designed to assist algorithmic traders in building, fine-tuning, and optimizing trading algorithms using a wide range of technical indicators and machine learning techniques. This project enables seamless integration and replication of NinjaTrader 8 (NT8) strategies made from Quagensia® no-code software in Python, providing a robust platform for real-time market analysis and strategy execution.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Indicator Library](#indicator-library)
- [Strategy Replication](#strategy-replication)
- [Optimization Engine](#optimization-engine)
- [Real-Time Execution](#real-time-execution)
- [User Interface](#user-interface)
- [Project Structure](#projet-structure)
- [Diagrams and Explanations](#diagrams-and-explanations)
- [Contributing](#contributing)
- [License](#license)
- [Legal Notice](#legal-notice)

## Overview

nocodeML is designed to:
- Implement and optimize over 30 different trading indicators.
- Replicate NT8 trading strategies in Python.
- Utilize machine learning and deep learning for parameter optimization.
- Provide real-time market analysis and strategy execution capabilities.
- Offer an interactive and user-friendly interface for managing and visualizing trading strategies.

## Features

- **Comprehensive Indicator Library**: A wide range of technical indicators such as ADX, ATR, Chaikin Oscillator, and more.
- **Advanced ML/DL Models**: Machine learning and deep learning models for optimizing trading strategy parameters.
- **Seamless Integration**: Smooth data flow and integration between NT8 and Python.
- **Real-Time Analysis**: Real-time data processing and trade execution.
- **User-Friendly Interface**: Interactive dashboard for strategy management and visualization.

## Installation

### Prerequisites

- Python 3.7 or higher
- Git
- Virtualenv

### Steps

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/nocodeML.git
    cd nocodeML
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Collection and Management

- **Integrate Market Replay Data**: Import historical market data for backtesting.
    ```python
    import pandas as pd
    historical_data = pd.read_csv('path_to_market_replay_data.csv')
    ```

- **Store Market Data**: Use pandas DataFrames to store and manage historical and real-time data.
    ```python
    real_time_data = pd.DataFrame(columns=['timestamp', 'price', 'volume'])
    ```

### 2. Indicator Implementation

- **Compute Technical Indicators**: Use built-in functions to calculate various indicators.
    ```python
    from indicators import compute_adx, compute_atr
    adx = compute_adx(historical_data)
    atr = compute_atr(historical_data)
    ```

### 3. Trading Strategy Replication

- **Translate NT8 Strategies to Python**: Convert C# strategies into Python functions.
    ```python
    def moving_average_crossover_strategy(data, short_window, long_window):
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0
        signals['short_mavg'] = data['close'].rolling(window=short_window, min_periods=1, center=False).mean()
        signals['long_mavg'] = data['close'].rolling(window=long_window, min_periods=1, center=False).mean()
        signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)
        signals['positions'] = signals['signal'].diff()
        return signals
    ```

### 4. Machine Learning & Deep Learning Optimization

- **Optimize Strategy Parameters**: Implement ML/DL models to optimize parameters.
    ```python
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier

    def optimize_parameters(data, parameters):
        model = RandomForestClassifier()
        grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=5)
        grid_search.fit(data.drop('target', axis=1), data['target'])
        return grid_search.best_params_
    ```

### 5. Real-Time Analysis and Execution

- **Real-Time Data Processing**: Process incoming market data in real-time.
    ```python
    import asyncio

    async def process_real_time_data(data_stream):
        async for data in data_stream:
            processed_data = process_data(data)
            execute_trade(processed_data)
    ```

- **Execute Trades**: Automatically execute trades based on real-time analysis.
    ```python
    def execute_trade(signal):
        if signal == 'buy':
            # Code to execute buy trade
            pass
        elif signal == 'sell':
            # Code to execute sell trade
            pass
    ```

### 6. User Interface

- **Dashboard Interaction**: Use the built-in dashboard to visualize data and adjust parameters.
    ```python
    from dashboard import Dashboard

    dashboard = Dashboard(data)
    dashboard.show()
    ```

## Project Structure

The project is organized as follows:

```plaintext
nocodeML/
├── data/
│   ├── raw/                # Raw data files
│   ├── processed/          # Processed data ready for use
│   ├── external/           # External data sources
├── notebooks/              # Jupyter notebooks for exploratory analysis and experiments
├── src/
│   ├── indicators/         # Implementation of technical indicators
│   ├── strategies/         # Trading strategies implementations
│   ├── optimization/       # Machine learning models and optimization code
│   ├── real_time/          # Real-time data processing and execution
│   ├── ui/                 # User interface components (e.g., Dash/Streamlit)
│   ├── utils/              # Utility functions and helper scripts
│   ├── data_ingestion/     # Data ingestion and preparation scripts
│   ├── __init__.py         # Make src a package
├── tests/                  # Unit and integration tests
│   ├── data/               # Test data
│   ├── indicators/         # Tests for indicators
│   ├── strategies/         # Tests for strategies
│   ├── optimization/       # Tests for optimization
│   ├── real_time/          # Tests for real-time processing
│   ├── ui/                 # Tests for UI components
├── bin/                    # Executable scripts
│   ├── start_realtime.py   # Script to start real-time processing
│   ├── backtest.py         # Script to run backtesting
│   ├── optimize.py         # Script to run optimization
├── docs/                   # Documentation and diagrams
│   ├── system_architecture.png
│   ├── indicator_calculation_flow.png
│   ├── strategy_optimization_process.png
│   ├── real_time_execution_pipeline.png
├── .gitignore              # Git ignore file
├── LICENSE                 # License file
├── README.md               # README file
├── requirements.txt        # Project dependencies
├── setup.py                # Setup script for the package
```

## Indicator Library

The library includes implementation of over 30 technical indicators, such as:
- **Average Directional Index (ADX)**: Measures trend strength.
- **Average True Range (ATR)**: Measures market volatility.
- **Chaikin Oscillator**: Combines price and volume data for trend identification.
- And many more...

## Strategy Replication

This module translates NT8 C# trading strategies into Python functions, ensuring accuracy and consistency with NT8 results. The strategies are tested and validated against historical market data to ensure their reliability.

## Optimization Engine

The optimization engine leverages machine learning and deep learning models to fine-tune trading strategy parameters. It includes functionalities for:
- **Parameter Optimization**: Using various ML algorithms to find optimal parameters.
- **Backtesting**: Testing strategies with optimized parameters on historical data.
- **Predictive Modeling**: Training predictive models to enhance trading performance.

## Real-Time Execution

The real-time execution framework enables:
- **Data Ingestion**: Real-time data fetching from market sources.
- **Data Processing**: Processing real-time data to generate trading signals.
- **Trade Execution**: Automated execution of trades based on generated signals.
- **Monitoring and Adjustment**: Continuous real-time monitoring and adjustment of strategies.

## User Interface

The interactive dashboard allows users to:
- **Visualize Market Data**: Display historical and real-time market data.
- **Adjust Strategy Parameters**: Modify and fine-tune strategy parameters in real-time.
- **Monitor Performance**: Track real-time performance and execution of trading strategies.

## Diagrams and Explanations

### System Architecture

![System Architecture](docs/graphs/System_Architecture.png)

**Explanation**: The System Architecture diagram illustrates the high-level structure of the NoCodeML system. It shows the interaction between various components, including data sources, the main processing unit, and the user interface. The flow of data begins with market data providers feeding into the data storage and processing units. The processing engine calculates indicators and optimizes strategies, which are then executed in real-time. Results are displayed on the user interface for user interaction and monitoring.

### Indicator Calculation Flow

![Indicator Calculation Flow](docs/graphs/indicator_calculation_flow.png)

**Explanation**: The Indicator Calculation Flow diagram details the comprehensive process of calculating technical indicators. Starting with raw market data inputs, the data undergoes pre-processing steps to ensure quality and consistency. This pre-processed data is then fed into various indicator functions, each computing a specific technical indicator. The calculated indicators are aggregated for use in trading strategies or further analysis, ensuring accurate and reliable trading decisions.

### Strategy Optimization Process

![Strategy Optimization Process](docs/graphs/strategy_optimization_process.png)

**Explanation**: This diagram outlines the detailed Strategy Optimization Process. It starts with defining initial strategy parameters, which are then input into an optimization engine using machine learning algorithms to find the optimal parameters. The optimized parameters are validated through rigorous backtesting on historical data. The results are meticulously analyzed to ensure strategy robustness. This iterative process continues until the best-performing parameters are identified, ensuring strategies are fine-tuned to maximize profitability and minimize risk.

### Real-Time Execution Pipeline

![Real-Time Execution Pipeline](docs/graphs/real-time_execution_pipeline.png)

**Explanation**: The Real-Time Execution Pipeline diagram illustrates the intricate components involved in the real-time execution of trading strategies. It includes:
- **Data Fetching**: Real-time market data is fetched from various sources.
- **Data Processing**: The ingested data is processed in real-time to compute necessary indicators.
- **Indicator Calculation**: Technical indicators are computed on-the-fly.
- **Trading Signals**: Trading signals are generated based on real-time indicator values.
- **Trade Execution**: Trades are executed based on the generated signals.
- **Real-Time Monitoring**: The system is continuously monitored in real-time to ensure optimal performance.
- **Alert Systems**: Alerts are generated for significant market events or anomalies.
- **Logging**: All activities are logged for audit and analysis purposes.
This pipeline ensures that trading strategies are executed efficiently and effectively in a live trading environment, providing traders with real-time insights and automated trading capabilities.

By integrating these diagrams and detailed explanations, the README provides a comprehensive and thorough guide to the nocodeML project, helping users understand the system architecture and the processes involved in building and optimizing trading strategies.

## Contributing

We welcome contributions from the community! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

Please ensure your code adheres to the existing style guidelines and includes appropriate tests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Legal Notice

* nocodeML is a product of nocodeML, a company which has no affiliation with NinjaTrader, LLC. Neither NinjaTrader, LLC nor any of its affiliates endorse, recommend, or approve nocodeML.
* nocodeML is a product of nocodeML, a company which has no affiliation with Quagensia, Inc. Neither Quagensia, Inc nor any of its affiliates endorse, recommend, or approve nocodeML.
* Quagensia® is a registered trademark of Quagensia, Inc.

