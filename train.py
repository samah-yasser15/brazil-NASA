import os
import json
import calendar
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILE = os.environ.get("CSV_FILE", "final_all_fires_Brazil.csv")
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

CHUNK_ROWS = 1_000_000
VALIDATION_DAYS = 365
BACKTEST_YEARS = 3
MAX_FORECAST_DAYS = 90
SEASON_MAX_YEARS = 15
REC_LIMIT = 90
MAX_ZONES_STORED = 200
LAGS = [1, 2, 3, 7, 14, 365]

FEATURES = [
    "month", "day_of_year", "day_of_week", "doy_sin", "doy_cos",
    *[f"lag_{lag}" for lag in LAGS],
    "roll_7", "roll_30",
]

LOCATION_CELL_DEG = 0.25
LOCATION_MIN_YEAR_SHARE = 0.7
LOCATION_WINDOW_DAYS = 7
LOCATION_MIN_ZONES = 20

# ==========================================
# 1. LOAD CSV & BUILD DAILY COUNTS + ZONES
# ==========================================
print("1. Reading CSV in chunks...")
header = pd.read_csv(CSV_FILE, nrows=0).columns
frp_col = "frp_aligned" if "frp_aligned" in header else "frp"
usecols = [c for c in ["acq_date", "latitude", "longitude", frp_col, "satellite", "confidence"] if c in header]

dtypes = {"latitude": "float32", "longitude": "float32", frp_col: "float32", "satellite": "str", "confidence": "str"}

chunks = []
for chunk in pd.read_csv(CSV_FILE, usecols=usecols, dtype=dtypes, chunksize=CHUNK_ROWS):
    chunk["acq_date"] = pd.to_datetime(chunk["acq_date"], errors="coerce")
    chunk = chunk.dropna(subset=["acq_date", "latitude", "longitude"])
    chunk["acq_date"] = chunk["acq_date"].dt.normalize()
    chunk["frp"] = chunk[frp_col].fillna(0).astype("float32")
    if frp_col != "frp":
        chunk.drop(columns=[frp_col], inplace=True)
    chunk["satellite"] = chunk["satellite"].fillna("Unknown").astype("category")
    chunk["confidence"] = chunk["confidence"].fillna("Unknown").astype("category")
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
del chunks

min_historical_date = df["acq_date"].min()
max_historical_date = df["acq_date"].max()
print(f"Loaded {len(df):,} rows ({min_historical_date.date()} to {max_historical_date.date()})")

# Precompute cell IDs for majority zones
lat_i = np.floor(df["latitude"].to_numpy(dtype="float32") / LOCATION_CELL_DEG).astype("int32")
lon_i = np.floor(df["longitude"].to_numpy(dtype="float32") / LOCATION_CELL_DEG).astype("int32")
CELL_ID = (lat_i + 1000) * 10000 + (lon_i + 5000)
del lat_i, lon_i

_dnp = df["acq_date"].to_numpy()
_min64 = min_historical_date.to_datetime64()
_day_all = ((_dnp - _min64) // np.timedelta64(1, "D")).astype("int32")
ORDER = np.argsort(_day_all, kind="stable").astype("int32")
_SORTED_DAYS = _day_all[ORDER]

_LAT = df["latitude"].to_numpy()
_LON = df["longitude"].to_numpy()
_FRP = df["frp"].to_numpy()
_SAT = df["satellite"].astype(str).to_numpy()
_CONF = df["confidence"].astype(str).to_numpy()


def rows_between(start, end):
    d0 = int((pd.Timestamp(start) - min_historical_date).days)
    d1 = int((pd.Timestamp(end) - min_historical_date).days)
    return int(np.searchsorted(_SORTED_DAYS, d0, side="left")), int(np.searchsorted(_SORTED_DAYS, d1, side="right"))


def majority_zones_for_date(month, day):
    slices = []
    for year in range(min_historical_date.year, max_historical_date.year + 1):
        d = min(day, calendar.monthrange(year, month)[1])
        center = pd.Timestamp(year=year, month=month, day=d)
        start = center - pd.Timedelta(days=LOCATION_WINDOW_DAYS)
        end = center + pd.Timedelta(days=LOCATION_WINDOW_DAYS)
        if start < min_historical_date or end > max_historical_date:
            continue
        lo, hi = rows_between(start, end)
        if hi > lo:
            slices.append((lo, hi))

    n_used = len(slices)
    if n_used < 2:
        return []

    idx = np.concatenate([ORDER[lo:hi] for lo, hi in slices]).astype(np.int64)
    year_id = np.concatenate([np.full(hi - lo, i, dtype=np.int64) for i, (lo, hi) in enumerate(slices)])
    cells = CELL_ID[idx].astype(np.int64)

    pairs = np.unique(cells * 64 + year_id)
    zone_ids, years_active = np.unique(pairs // 64, return_counts=True)

    keep = np.array([], dtype=np.int64)
    for share in (LOCATION_MIN_YEAR_SHARE, 0.5, 0.3):
        need = max(2, int(np.ceil(share * n_used)))
        keep = zone_ids[years_active >= need]
        if len(keep) >= LOCATION_MIN_ZONES:
            break
    if len(keep) == 0:
        return []

    mask = np.isin(cells, keep)
    rows = idx[mask]
    sub = pd.DataFrame({
        "cell": cells[mask],
        "latitude": _LAT[rows].astype("float64"),
        "longitude": _LON[rows].astype("float64"),
        "frp": _FRP[rows].astype("float64"),
        "sat": _SAT[rows],
        "conf": _CONF[rows],
    })

    years_map = pd.Series(years_active, index=zone_ids)
    stats = sub.groupby("cell").agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        avg_frp=("frp", "mean"),
        n=("latitude", "size"),
    )
    stats["years_active"] = years_map.reindex(stats.index).astype(int)
    stats["total_years_checked"] = int(n_used)
    stats["year_share"] = stats["years_active"] / n_used
    stats["per_year"] = stats["n"] / n_used
    stats["weight"] = (stats["year_share"] ** 2) * np.maximum(stats["per_year"], 0.1)
    stats["weight"] = stats["weight"] / stats["weight"].sum()
    stats = stats.sort_values("weight", ascending=False).head(MAX_ZONES_STORED)

    top = sub[sub["cell"].isin(stats.index)]
    sat_mode = top.groupby("cell")["sat"].agg(lambda s: s.value_counts().index[0])
    conf_mode = top.groupby("cell")["conf"].agg(lambda s: s.value_counts().index[0])

    zones = []
    for cell_id, r in stats.iterrows():
        zones.append({
            "latitude": round(float(r["latitude"]), 5),
            "longitude": round(float(r["longitude"]), 5),
            "avg_frp": round(float(r["avg_frp"]), 2),
            "years_active": int(r["years_active"]),
            "total_years_checked": int(r["total_years_checked"]),
            "weight": float(r["weight"]),
            "satellite": str(sat_mode[cell_id]),
            "confidence": str(conf_mode[cell_id]),
        })
    return zones


# ==========================================
# 2. TRAIN XGBOOST (90-DAY HORIZON)
# ==========================================
print("2. Training 90-day XGBoost model...")
counts = df.groupby("acq_date").size()
full_dates = pd.date_range(counts.index.min(), counts.index.max(), freq="D")
counts = counts.reindex(full_dates, fill_value=0).astype(float)
counts.index.name = "acq_date"


def build_feature_frame(s: pd.Series) -> pd.DataFrame:
    idx = s.index
    doy = idx.dayofyear.to_numpy()
    angle = 2 * np.pi * doy / 365.25
    feats = pd.DataFrame(index=idx)
    feats["month"] = idx.month.to_numpy()
    feats["day_of_year"] = doy
    feats["day_of_week"] = idx.dayofweek.to_numpy()
    feats["doy_sin"] = np.sin(angle)
    feats["doy_cos"] = np.cos(angle)
    for lag in LAGS:
        feats[f"lag_{lag}"] = s.shift(lag)
    prev = s.shift(1)
    feats["roll_7"] = prev.rolling(7, min_periods=1).mean()
    feats["roll_30"] = prev.rolling(30, min_periods=1).mean()
    return feats[FEATURES]


def feature_row(vals, date) -> list:
    n = len(vals)
    doy = date.dayofyear
    angle = 2 * np.pi * doy / 365.25
    row = [date.month, doy, date.dayofweek, np.sin(angle), np.cos(angle)]
    for lag in LAGS:
        row.append(vals[n - lag] if n - lag >= 0 else np.nan)
    row.append(float(np.mean(vals[max(0, n - 7):n])) if n > 0 else np.nan)
    row.append(float(np.mean(vals[max(0, n - 30):n])) if n > 0 else np.nan)
    return row


def recursive_forecast(model, history, last_date, n_days):
    vals = [float(v) for v in history]
    preds = []
    for i in range(1, min(int(n_days), MAX_FORECAST_DAYS) + 1):
        date = last_date + timedelta(days=i)
        x = np.array([feature_row(vals, date)], dtype=float)
        p = max(0.0, float(np.expm1(model.predict(x)[0])))
        vals.append(p)
        preds.append(p)
    return preds


def make_model(n_estimators, early_stopping_rounds=None):
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
    )


X_all = build_feature_frame(counts)
split = len(counts) - VALIDATION_DAYS
X_tr, X_va = X_all.iloc[:split].to_numpy(dtype=float), X_all.iloc[split:split + MAX_FORECAST_DAYS].to_numpy(dtype=float)
y_tr, y_va = counts.iloc[:split], counts.iloc[split:split + MAX_FORECAST_DAYS]

val_model = make_model(2000, early_stopping_rounds=50)
val_model.fit(X_tr, np.log1p(y_tr.to_numpy()), eval_set=[(X_va, np.log1p(y_va.to_numpy()))], verbose=False)
best_n = int(val_model.best_iteration) + 1
rec_preds = recursive_forecast(val_model, y_tr.to_numpy(), counts.index[split - 1], MAX_FORECAST_DAYS)

y_true_arr = y_va.to_numpy(dtype=float)
rec_arr = np.asarray(rec_preds, dtype=float)
mae = float(mean_absolute_error(y_true_arr, rec_arr))
rmse = float(np.sqrt(mean_squared_error(y_true_arr, rec_arr)))
r2 = float(r2_score(y_true_arr, rec_arr))
mean_actual = float(y_true_arr.mean())

X_full = X_all.dropna().to_numpy(dtype=float)
y_full = np.log1p(counts.loc[X_all.dropna().index].to_numpy(dtype=float))
full_model = make_model(max(best_n, 100))
full_model.fit(X_full, y_full, verbose=False)

# ==========================================
# 3. SAVE LIGHTWEIGHT ARTIFACTS FOR DEPLOYMENT
# ==========================================
print("3. Saving lightweight artifacts...")

future_preds = recursive_forecast(full_model, counts.to_numpy(dtype=float), counts.index[-1], MAX_FORECAST_DAYS)
future_dates = [counts.index[-1] + timedelta(days=i) for i in range(1, MAX_FORECAST_DAYS + 1)]

forecasts_out = []
for dt, pred_val in zip(future_dates, future_preds):
    zones = majority_zones_for_date(dt.month, dt.day)
    forecasts_out.append({
        "date": str(dt.date()),
        "predicted_fires": round(float(pred_val), 2),
        "zones": zones,
    })

with open(os.path.join(ARTIFACTS_DIR, "forecasts_90d.json"), "w", encoding="utf-8") as f:
    json.dump(forecasts_out, f)

metrics_out = {
    "validation_days": MAX_FORECAST_DAYS,
    "best_iteration": best_n,
    "last_historical_date": str(max_historical_date.date()),
    "min_historical_date": str(min_historical_date.date()),
    "total_records": int(len(df)),
    "chosen_method": "day-by-day XGBoost model (90-day horizon) + recurring majority lat/lon zones",
    "chosen_forecast": {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "mean_actual": mean_actual,
        "mae_pct_of_mean": round(100 * mae / mean_actual, 2) if mean_actual else None,
    },
}

with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics_out, f, indent=2)

# Save daily counts summary for historical lookup
daily_summary = counts.reset_index()
daily_summary.columns = ["date", "hotspots"]
daily_summary["date"] = daily_summary["date"].dt.strftime("%Y-%m-%d")
daily_summary.to_csv(os.path.join(ARTIFACTS_DIR, "daily_counts.csv"), index=False)

print("\nDONE! Artifacts saved in ./artifacts/:")
for fn in os.listdir(ARTIFACTS_DIR):
    size_mb = os.path.getsize(os.path.join(ARTIFACTS_DIR, fn)) / (1024 * 1024)
    print(f"  - {fn} ({size_mb:.2f} MB)")