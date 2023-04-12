import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


symbols = ['AAPL', 'TSLA', 'MSFT', 'IAU', 'SLV', 'GOOG', 'META', 'AMZN']
final_df = None

for symbol in symbols:
    df = yf.download(symbol, period='2y')
    df = df.rename(columns={'Adj Close': symbol})
    if final_df is None:
        final_df = df[[symbol]]
    else:
        final_df = final_df.join(df[[symbol]])

returns_df = final_df.pct_change()
returns_df = returns_df.dropna()

start_date = final_df.index.min() + timedelta(days=10)
end_date = final_df.index.max()

while start_date + timedelta(days=10) < end_date:
    slice_df = final_df[(final_df.index > start_date - timedelta(days=10)) & (final_df.index <= start_date)]
    future_df = returns_df[(returns_df.index > start_date) & (returns_df.index < start_date + timedelta(days=10))].copy()
    future_df = future_df.sum(axis=0)
    mu = expected_returns.mean_historical_return(slice_df)
    S = risk_models.sample_cov(slice_df)
    ef = EfficientFrontier(mu, S)
    raw_weights = ef.max_sharpe(risk_free_rate=-1)
    # raw_weights = ef.efficient_return(-1.0)
    cleaned_weights = pd.Series(ef.clean_weights())
    print(start_date + timedelta(days=10), (cleaned_weights*future_df).values.mean())
    start_date += timedelta(days=10)
    print('-' * 50)
