#!/usr/bin/env python3
"""
score_generator.py — CLI scoring tool for Ecom Fraud model

Usage examples:
  python score_generator.py --input sample_shops.csv --output scored.csv
  python score_generator.py --single '{"Domain length": 20, "Top domain length":3, ... }' --json
"""

import argparse
import json
import sys
import os
import joblib
import numpy as np
import pandas as pd

# Remove FutureWarning about silent downcasting
pd.set_option('future.no_silent_downcasting', True)

# Candidate paths (relative and absolute)
CANDIDATE_MODEL_PATHS = "./train/model.pkl"
CANDIDATE_META_PATHS = "./train/model_metadata.json"

MODEL_PATH = CANDIDATE_MODEL_PATHS
META_PATH = CANDIDATE_META_PATHS

if MODEL_PATH is None or META_PATH is None:
    print(f"[ERROR] Could not find model or metadata. Looked at: {CANDIDATE_MODEL_PATHS} and {CANDIDATE_META_PATHS}")
    sys.exit(2)

model = joblib.load(MODEL_PATH)
with open(META_PATH, "r") as f:
    metadata = json.load(f)

EXPECTED_FEATURES = metadata.get("features_used", [])
MODEL_NAME = metadata.get("model_name", "unknown")

# print(f"[INFO] Loaded model: {MODEL_NAME}")
# print(f"[INFO] Expecting {len(EXPECTED_FEATURES)} features.")

# -------------------------
# Helpers
# -------------------------
def get_risk_tier(score: float) -> str:
    if score >= 80:
        return "High Risk"
    elif score >= 30:
        return "Medium Risk"
    else:
        return "Low Risk"

def normalize_and_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    COLUMN_MAP = {
        "Presence of prefix 'www'": "Presence of prefix 'www' "
    }
    return df.rename(columns=COLUMN_MAP)

def validate_and_prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_and_map_columns(df)
    df = df.replace({"": np.nan, None: np.nan})
    expected_set = set(EXPECTED_FEATURES)
    received_set = set(df.columns)
    missing = expected_set - received_set
    if missing:
        raise ValueError(f"Missing required columns ({len(missing)}): {sorted(list(missing))}\nExpected columns: {EXPECTED_FEATURES}")
    # Reindex to ensure consistent ordering for downstream code (although pipeline uses names)
    df = df.reindex(columns=EXPECTED_FEATURES)
    return df

# -------------------------
# CLI operations
# -------------------------
def predict_single(input_json: str, as_json: bool = False):
    try:
        data = json.loads(input_json)
    except Exception as e:
        print("[ERROR] Could not parse input JSON:", e)
        sys.exit(2)

    df = pd.DataFrame([data])
    try:
        df_prepped = validate_and_prepare_df(df)
    except ValueError as e:
        print("[ERROR] Validation failed:", e)
        sys.exit(2)

    proba = model.predict_proba(df_prepped)[0][1]
    score = float(proba * 100)
    tier = get_risk_tier(score)
    result = {"probability_fraud": float(proba), "risk_score_0_100": score, "risk_tier": tier}
    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print("----- Prediction Result -----")
        print(f"Fraud Probability: {result['probability_fraud']:.6f}")
        print(f"Risk Score (0-100): {result['risk_score_0_100']:.2f}")
        print(f"Risk Tier: {result['risk_tier']}")

def predict_batch(input_csv: str, output_csv: str = None, as_json: bool = False):
    if not os.path.exists(input_csv):
        print("[ERROR] Input CSV not found:", input_csv)
        sys.exit(2)
    df_raw = pd.read_csv(input_csv)
    try:
        df = validate_and_prepare_df(df_raw)
    except ValueError as e:
        print("[ERROR] Validation failed:", e)
        sys.exit(2)

    proba = model.predict_proba(df)[:, 1]
    scores = proba * 100

    df_raw["probability_fraud"] = proba
    df_raw["risk_score_0_100"] = scores
    df_raw["risk_tier"] = df_raw["risk_score_0_100"].apply(get_risk_tier)

    if as_json:
        print(df_raw.to_json(orient="records", indent=2))
    else:
        for idx, row in df_raw.iterrows():
            print(f"[Row {idx+1}]")
            print(f"probability_fraud: {row['probability_fraud']:.6f}")
            print(f"risk_score_0_100: {row['risk_score_0_100']:.2f}")
            print(f"risk_tier: {row['risk_tier']}")
            print("-" * 34)

    if output_csv:
        df_raw.to_csv(output_csv, index=False)
        print("[INFO] Saved predictions to:", output_csv)

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="E-commerce Fraud Risk Scoring Engine")
    parser.add_argument("--single", help="Provide JSON string for one record")
    parser.add_argument("--input", help="Input CSV for batch scoring")
    parser.add_argument("--output", help="Output CSV path for batch mode")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    if args.single:
        predict_single(args.single, as_json=args.json)
    elif args.input:
        predict_batch(args.input, output_csv=args.output, as_json=args.json)
    else:
        print("No input provided. Use --single or --input.")
        print("Examples:")
        print('  python score_generator.py --single "{\\"Domain length\\": 20, \\"Top domain length\\": 3, ... }"')
        print('  python score_generator.py --input sample_shops.csv --output scored.csv')
        sys.exit(0)

if __name__ == "__main__":
    main()
