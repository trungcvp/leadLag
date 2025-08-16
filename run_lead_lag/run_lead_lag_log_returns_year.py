import pandas as pd
import numpy as np
import os
from code.models.lead_lag_measure import get_lead_lag_matrix
from code.models.herm_matrix_algo import get_ordered_clustering
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. Load returns matrix
returns = pd.read_csv("returns_matrix.csv", index_col=0, parse_dates=True)

# 2. Chia theo năm
returns_yearly = {year: df for year, df in returns.groupby(returns.index.year)}

# Chỉ giữ các năm đủ dữ liệu
returns_yearly = {
    year: df for year, df in returns_yearly.items()
    if df.shape[0] > 200 and df.shape[1] > 100
}

# 3. Tính clustering từng năm
clusterings = {}
for year, df_ret in tqdm(returns_yearly.items(), desc="Yearly clustering"):
    df_ret = df_ret.loc[:, ~(df_ret == 0).all() & ~(df_ret.isna().all())]
    df_ret = df_ret.fillna(0)
    if df_ret.shape[1] < 100:
        continue

    try:
        LL = get_lead_lag_matrix(
            data=df_ret,
            method='ccf_auc',
            correlation_method='distance',
            max_lag=5
        )
        skew = LL - LL.T
        clustering = get_ordered_clustering(skew, num_clusters=10)
        clusterings[year] = clustering
    except Exception as e:
        print(f"⚠️ Skipping {year}: {e}")

# 4. Tính Adjusted Rand Index giữa các năm
years = sorted(clusterings.keys())
n = len(years)
ari_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        ci = clusterings[years[i]]
        cj = clusterings[years[j]]
        common_tickers = ci.index.intersection(cj.index)
        if len(common_tickers) > 10:
            ari_matrix[i, j] = adjusted_rand_score(
                ci[common_tickers], cj[common_tickers]
            )
        else:
            ari_matrix[i, j] = np.nan

# 5. Vẽ Fig.14
plt.figure(figsize=(10, 8))
sns.heatmap(
    ari_matrix,
    xticklabels=years,
    yticklabels=years,
    cmap="magma",
    square=True,
    linewidths=0.3,
    vmin=0,
    vmax=1,
    cbar_kws={"label": "Adjusted Rand Index"}
)
plt.title("Fig. 14 — Adjusted Rand Index between yearly Hermitian RW clusters")
plt.tight_layout()
plt.savefig("fig14_rand_index.png", dpi=300)
plt.show()
