import pandas as pd
import numpy as np
from code.models.lead_lag_measure import get_lead_lag_matrix
import time

if __name__ == "__main__":
    # Load dữ liệu
    returns = pd.read_csv("returns_matrix.csv", index_col=0, parse_dates=True)
    volume_df = pd.read_csv("volume.csv", index_col=0, parse_dates=True)
    #Tính lead-lag matrix song song
    print("Calculating lead–lag matrix...")
    start = time.time()
    LL = get_lead_lag_matrix(
        data=returns,
        method='ccf_auc',
        volume_df = volume_df,
        correlation_method='distance',
        max_lag=5
    )
    end = time.time()
    print(f"Done in {end - start:.2f} seconds")

    # 3. Lưu lại kết quả
    LL.to_pickle("lead_lag_matrix.pkl")
    print("Saved to lead_lag_matrix.pkl")
