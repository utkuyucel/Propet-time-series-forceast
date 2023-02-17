import pandas as pd
import random
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from typing import Dict

class TimeSeriesForecastingModel:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.model = Prophet(growth=config['growth'],
                             changepoint_prior_scale=config['changepoint_prior_scale'],
                             seasonality_mode=config['seasonality_mode'],
                             holidays_prior_scale=config['holidays_prior_scale'],
                             daily_seasonality=config['daily_seasonality'],
                             weekly_seasonality=config['weekly_seasonality'],
                             yearly_seasonality=config['yearly_seasonality'])
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path, encoding = "utf-8")
        df = df.rename(columns={'Date': 'ds', 'Sales': 'y'})
        df = df.set_index('ds').interpolate().reset_index()
        return df
        
    def train(self, df: pd.DataFrame) -> None:
        self.model.fit(df)
        
    def predict(self, periods: int) -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast
        
    def evaluate(self, cv_initial: str, cv_period: str, cv_horizon: str) -> pd.DataFrame:
        cv_results = cross_validation(self.model, initial=cv_initial, period=cv_period, horizon=cv_horizon)
        perf_metrics = performance_metrics(cv_results)
        return perf_metrics
    
    def save_forecast(self, forecast: pd.DataFrame) -> None: 
        forecast.to_csv("forecasted.csv")
        
    def plot_forecast(self, forecast: pd.DataFrame) -> None:
        fig = self.model.plot(forecast, uncertainty=True)
        
    def plot_components(self, forecast: pd.DataFrame) -> None:
        fig = self.model.plot_components(forecast)


if __name__ == "__main__":

    config = {
        'growth': 'linear',
        'changepoint_prior_scale': 0.05,
        'seasonality_mode': 'multiplicative',
        'holidays_prior_scale': 10.0,
        'daily_seasonality': True,
        'weekly_seasonality': True,
        'yearly_seasonality': True
    }
  
    
    model = TimeSeriesForecastingModel(config)

    df = model.load_data("dset.csv")

    model.train(df)

    forecast = model.predict(periods=730)
    model.save_forecast(forecast)

    model.plot_forecast(forecast)
    print()
    model.plot_components(forecast)
