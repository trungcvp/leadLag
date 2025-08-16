import pandas as pd
import numpy as np

data = pd.read_csv("/home/trungnguyen21/Desktop/mlj-lead-lag-main/data/data1006ver2.csv", parse_dates=['date'])

#Lọc valid Ticker
df = data.copy()
ticker_counts = df.groupby('TICKER')['date'].nunique().sort_values(ascending=False)
total_days = df['date'].nunique()
valid_tickers = ticker_counts[ticker_counts >= 0.9 * total_days].index.tolist()

#Tạo returns data / volume data
def prepare_returns_filtered(df, valid_tickers,values_col = 'LOG_RET',top_n=500):
    df = df.copy()


    df = df.dropna(subset=['RET', 'PRC', 'VOL'])
    df = df[pd.to_numeric(df['RET'], errors='coerce').notnull()]
    df['RET'] = df['RET'].astype(float)
    df['LOG_RET'] = np.log1p(df['RET'])

    #Tính DOLLAR_VOLUME
    df['DOLLAR_VOLUME'] = df['PRC'].abs() * df['VOL']

    #Giữ ticker có trong valid_tickers
    df = df[df['TICKER'].isin(valid_tickers)]


    avg_volume = df.groupby('TICKER')['DOLLAR_VOLUME'].mean()
    top_tickers = avg_volume.nlargest(top_n).index.tolist()
    df = df[df['TICKER'].isin(top_tickers)]


    df['date'] = pd.to_datetime(df['date'])


    numeric_cols = df.select_dtypes(include='number').columns
    df = df[['date', 'TICKER'] + list(numeric_cols)].groupby(['date', 'TICKER'], as_index=False).mean()

    # 7. Pivot thành returns matrix
    returns = df.pivot(index='date', columns='TICKER', values=values_col)
    returns = returns.fillna(0)  # hoặc .dropna(thresh=...)

    return returns

#Lưu lại bảng Returns
returns = prepare_returns_filtered(data, valid_tickers, top_n=500)
returns.to_csv("/data/returns_matrix.csv")

# Lưu bảng Volume
volume_df = prepare_returns_filtered(data, valid_tickers,values_col='VOL', top_n=500)
volume_df_log = np.log(volume_df)
volume_df_log.replace(-np.inf, 0, inplace=True)
volume_df_log.to_csv("/data/volume.csv")