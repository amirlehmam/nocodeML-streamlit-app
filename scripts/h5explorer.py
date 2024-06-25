from diagrams import Diagram, Cluster, Edge
from diagrams.aws.storage import S3
from diagrams.onprem.compute import Server
from diagrams.onprem.client import Users
from diagrams.onprem.database import Postgresql
from diagrams.onprem.mlops import Mlflow
from diagrams.programming.framework import Flask
from diagrams.programming.language import Python

with Diagram("NoCodeML Workflow V2", show=False):

    market_replay_data = S3("Market Replay Data (NQ)")

    with Cluster("Data Processing"):
        csv_conversion = Server("Convert to CSV")
        hdf5_conversion = Server("Transform to HDF5")
        renko_calc = Server("Renko Bars Calculation")
        dynamic_plot = Python("Matplotlib Dynamic Chart")

        market_replay_data >> csv_conversion >> hdf5_conversion >> renko_calc >> dynamic_plot

    with Cluster("Backtesting and Analysis"):
        manual_backtest = Server("Manual Choices")
        backtest_algo = Server("Backtest Algorithm")
        ml_feedback = Mlflow("ML Feedback Loop")
        strategy_output = S3("Strategy Output (CSV)")

        dynamic_plot >> manual_backtest >> backtest_algo >> ml_feedback >> strategy_output

    with Cluster("Web Application"):
        web_app = Flask("NoCode-ML.com")
        ml_analysis = Mlflow("ML/Stats Analysis")
        insights = Server("Trading Insights")

        strategy_output >> web_app >> ml_analysis >> insights

    data_storage = S3("Data Storage")

    strategy_output >> data_storage
    insights >> data_storage

    Postgresql("Database") - Edge(style="dotted") - web_app
