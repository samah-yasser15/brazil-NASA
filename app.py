import os
import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
MAX_FORECAST_DAYS = 90
DEFAULT_ZONES = 50
MAX_ZONES_LIMIT = 200

with open(os.path.join(ARTIFACTS_DIR, "forecasts_90d.json"), "r", encoding="utf-8") as f:
    FORECASTS_90D = json.load(f)

with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "r", encoding="utf-8") as f:
    METRICS = json.load(f)

DAILY_COUNTS_DF = pd.read_csv(os.path.join(ARTIFACTS_DIR, "daily_counts.csv"))
DAILY_COUNTS_MAP = dict(zip(DAILY_COUNTS_DF["date"], DAILY_COUNTS_DF["hotspots"]))


def slice_and_allocate_zones(zones, predicted_fires, max_zones):
    if not zones:
        return []
    top = zones[:max_zones]
    weights = np.array([z["weight"] for z in top], dtype=float)
    weights = weights / weights.sum()

    out = []
    for z, w in zip(top, weights):
        out.append({
            "latitude": z["latitude"],
            "longitude": z["longitude"],
            "allocated_fires": round(float(predicted_fires * w), 2),
            "share_of_day": round(float(w * 100), 2),
            "years_active": f"{z['years_active']}/{z['total_years_checked']}",
            "avg_frp": z["avg_frp"],
            "satellite": z["satellite"],
            "confidence": z["confidence"],
        })
    return out


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "max_forecast_days": MAX_FORECAST_DAYS,
        "last_historical_date": METRICS.get("last_historical_date"),
    })


@app.route("/metrics", methods=["GET"])
def get_metrics():
    return jsonify(METRICS)


@app.route("/forecast", methods=["GET"])
def get_forecast():
    days = request.args.get("days", default=MAX_FORECAST_DAYS, type=int)
    max_zones = request.args.get("zones", default=DEFAULT_ZONES, type=int)

    if days is None or days < 1:
        return jsonify({"error": "Parameter 'days' must be a positive integer."}), 400
    if days > MAX_FORECAST_DAYS:
        return jsonify({
            "error": f"Forecast horizon is capped at {MAX_FORECAST_DAYS} days.",
            "max_forecast_days": MAX_FORECAST_DAYS,
        }), 400

    max_zones = max(1, min(int(max_zones), MAX_ZONES_LIMIT))
    records = []
    for item in FORECASTS_90D[:days]:
        pred_val = float(item["predicted_fires"])
        records.append({
            "date": item["date"],
            "predicted_fires": pred_val,
            "zones": slice_and_allocate_zones(item["zones"], pred_val, max_zones),
        })

    return jsonify({
        "horizon_days": days,
        "max_forecast_days": MAX_FORECAST_DAYS,
        "forecasts": records,
    })


@app.route("/hotspots", methods=["GET"])
def get_hotspots():
    date_str = request.args.get("date")
    limit = request.args.get("limit", default=DEFAULT_ZONES, type=int)
    limit = max(1, min(int(limit), MAX_ZONES_LIMIT))

    if not date_str:
        return jsonify({"error": "Query parameter 'date' (YYYY-MM-DD) is required."}), 400

    match = next((x for x in FORECASTS_90D if x["date"] == date_str), None)
    if not match:
        return jsonify({
            "error": "Date is outside the supported 90-day forecast window.",
            "forecast_start": FORECASTS_90D[0]["date"],
            "forecast_end": FORECASTS_90D[-1]["date"],
        }), 404

    pred_val = float(match["predicted_fires"])
    return jsonify({
        "date": date_str,
        "predicted_fires": pred_val,
        "zones": slice_and_allocate_zones(match["zones"], pred_val, limit),
    })


@app.route("/check-date", methods=["GET"])
def check_date():
    date_str = request.args.get("date")
    if not date_str:
        return jsonify({"error": "Query parameter 'date' (YYYY-MM-DD) is required."}), 400

    if date_str in DAILY_COUNTS_MAP:
        return jsonify({
            "type": "historical",
            "status": "success",
            "date": date_str,
            "hotspots": int(DAILY_COUNTS_MAP[date_str]),
            "fires": [],
        })

    return jsonify({"error": f"No historical record found for {date_str}"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)