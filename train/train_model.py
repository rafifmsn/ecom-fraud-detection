import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42

# Try to import xgboost, handle absence gracefully
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    print("xgboost not available in this environment. XGBoost training will be skipped. Install with `pip install xgboost` to enable it.")

# ---------------------------
# Load data
# ---------------------------
DATA_PATH = "./fraudulent_online_shops_dataset.csv"  # local path from your upload
assert os.path.exists(DATA_PATH), f"Data file not found at {DATA_PATH}"

df = pd.read_csv(DATA_PATH)
print("Loaded dataset shape:", df.shape)

# ---------------------------
# Basic cleaning & label
# ---------------------------
# Map target: fraudulent -> 1, legitimate -> 0
if 'Label' in df.columns:
    df['target'] = df['Label'].map({'fraudulent': 1, 'legitimate': 0})
else:
    raise ValueError("Expected 'Label' column in dataset.")

# ---------------------------
# Parse dates
# ---------------------------
# convert date fields to datetime where possible
date_cols_candidates = [col for col in df.columns if 'date' in col.lower() or 'registration' in col.lower() or 'expire' in col.lower()]
print("Detected date-like columns:", date_cols_candidates)

# Try parsing
for c in date_cols_candidates:
    try:
        df[c + "_dt"] = pd.to_datetime(df[c], errors='coerce')
        print(f"Parsed {c} -> {c + '_dt'}")
    except Exception:
        df[c + "_dt"] = pd.NaT

# Choose reference date: use max parsed date if available, else today
parsed_date_cols = [c + "_dt" for c in date_cols_candidates if c + "_dt" in df.columns]
if parsed_date_cols:
    # Use maximum non-null date as reference to compute relative values
    max_dates = [df[col].max() for col in parsed_date_cols if df[col].notnull().sum() > 0]
    if max_dates:
        ref_date = max(max_dates)
    else:
        ref_date = pd.to_datetime("2023-07-01")  # fallback
else:
    ref_date = pd.to_datetime("2023-07-01")
print("Reference date used for date diffs:", ref_date)

# Example engineered date features:
# days_until_ssl_expiry and days_since_registration (if available)
if "SSL certificate expire date_dt" in df.columns:
    df["days_until_ssl_expiry"] = (df["SSL certificate expire date_dt"] - ref_date).dt.days * -1
else:
    df["days_until_ssl_expiry"] = np.nan

# domain registration
reg_cols = [c for c in df.columns if 'registration' in c.lower() and c.endswith('_dt')]
if reg_cols:
    df["days_since_registration"] = (ref_date - df[reg_cols[0]]).dt.days
else:
    df["days_since_registration"] = np.nan

# ---------------------------
# Handle TrustPilot score (-1 -> missing)
# ---------------------------
if 'TrustPilot score' in df.columns:
    df['TrustPilot_score_clean'] = df['TrustPilot score'].replace(-1, np.nan)
else:
    df['TrustPilot_score_clean'] = np.nan

# ---------------------------
# Tranco rank handling
# ---------------------------
tranco_cols = [c for c in df.columns if 'tranco' in c.lower()]
if tranco_cols:
    tranco_col = tranco_cols[0]
    # create flag if listed and log transform for positives
    df['is_in_tranco'] = (df[tranco_col] > 0).astype(int)
    df['tranco_rank_log'] = np.where(df[tranco_col] > 0, np.log1p(df[tranco_col]), 0.0)
else:
    df['is_in_tranco'] = 0
    df['tranco_rank_log'] = 0.0

# ---------------------------
# URL-based ratio features
# ---------------------------
# Map candidate column names flexibly
def find_col(keywords):
    for key in df.columns:
        k = key.lower()
        if all(kw in k for kw in keywords):
            return key
    return None

col_domain_len = find_col(['domain', 'length'])
col_num_digits = find_col(['digit'])
col_num_letters = find_col(['letter'])
col_num_dots = find_col(['dot'])
col_num_hyphens = find_col(['hyphen', '-']) or find_col(['hyphens'])

# If column names differ, try common alternatives:
if col_domain_len is None:
    # try exact names from doc
    alt = ['Domain length', 'domain length']
    for a in alt:
        if a in df.columns:
            col_domain_len = a
            break

if col_num_digits is None:
    alt = ['Number  of digits', 'Number of digits in the URL', 'Number  of digits']
    for a in alt:
        if a in df.columns:
            col_num_digits = a
            break

if col_num_letters is None:
    alt = ['Number  of letters', 'Number of letters in the URL', 'Number  of letters']
    for a in alt:
        if a in df.columns:
            col_num_letters = a
            break

if col_num_dots is None:
    alt = ['Number  of dots (.)', 'Number of dots (.)', 'Number of dots']
    for a in alt:
        if a in df.columns:
            col_num_dots = a
            break

if col_num_hyphens is None:
    alt = ['Number  of hyphens (-)', 'Number of hyphens (-)', 'Number of hyphens']
    for a in alt:
        if a in df.columns:
            col_num_hyphens = a
            break

# Compute densities safely
def safe_div(numer, denom):
    return np.where((denom > 0) & (~np.isnan(denom)), numer / denom, 0.0)

if col_domain_len:
    domain_len = df[col_domain_len].astype(float)
else:
    domain_len = np.full(len(df), np.nan)

if col_num_digits:
    digits = df[col_num_digits].astype(float)
else:
    digits = np.zeros(len(df))

if col_num_letters:
    letters = df[col_num_letters].astype(float)
else:
    letters = np.zeros(len(df))

if col_num_dots:
    dots = df[col_num_dots].astype(float)
else:
    dots = np.zeros(len(df))

if col_num_hyphens:
    hyphens = df[col_num_hyphens].astype(float)
else:
    hyphens = np.zeros(len(df))

df['digit_density'] = safe_div(digits, domain_len)
df['hyphen_density'] = safe_div(hyphens, domain_len)
df['dot_density'] = safe_div(dots, domain_len)
df['letter_ratio'] = safe_div(letters, domain_len)
# also numeric domain length and top domain length keep as-is if present
# find top domain length column
col_top_domain_len = find_col(['top domain length']) or find_col(['top domain'])
if col_top_domain_len is None and 'Top domain length' in df.columns:
    col_top_domain_len = 'Top domain length'

# ---------------------------
# Payment/contact/logos features
# ---------------------------
# We'll keep payment flags as-is; create combined count
payment_cols = []
for candidate in ['credit card', 'Presence of credit card payment', 'Presence of credit card payment']:
    for c in df.columns:
        if candidate.lower() in c.lower():
            payment_cols.append(c)
# additional payment columns search
for c in df.columns:
    l = c.lower()
    if 'money back' in l or 'cash on delivery' in l or 'crypto' in l or 'payment' in l:
        if c not in payment_cols:
            payment_cols.append(c)

# Remove duplicates
payment_cols = list(dict.fromkeys(payment_cols))
# Create count of payment methods
if payment_cols:
    df['num_payment_methods'] = df[payment_cols].sum(axis=1)
else:
    df['num_payment_methods'] = 0

# Email free indicator (0..3)
email_col = None
for c in df.columns:
    if 'free contact' in c.lower() or 'free' in c.lower() and 'email' in c.lower() or 'presence of free contact emails' in c.lower():
        email_col = c
        break
if email_col is None:
    for c in df.columns:
        if 'email' in c.lower():
            email_col = c
            break

if email_col:
    df['has_free_email'] = (df[email_col] == 1).astype(int)
    # also one-hot will be used later
else:
    df['has_free_email'] = 0

# Young domain indicator (0=old,1=young,2=hidden)
young_col = None
for c in df.columns:
    if 'indication of young' in c.lower() or 'young' in c.lower():
        young_col = c
        break
if young_col:
    df['young_domain'] = df[young_col].astype(str)
else:
    df['young_domain'] = '0'

# Logo presence
logo_col = None
for c in df.columns:
    if 'logo' in c.lower() and 'presence' in c.lower():
        logo_col = c
        break
if logo_col:
    df['has_logo'] = df[logo_col].astype(int)
else:
    df['has_logo'] = 0

# Presence in Tranco list flag is already created as is_in_tranco

# TrustPilot presence already exists in dataset often as 'Presence of TrustPilot reviews'
trustpresence_col = None
for c in df.columns:
    if 'trustpilot' in c.lower() and 'presence' in c.lower():
        trustpresence_col = c
        break
if trustpresence_col is None:
    # fallback
    for c in df.columns:
        if 'trustpilot' in c.lower():
            trustpresence_col = c
            break

if trustpresence_col:
    df['trustpilot_has_reviews'] = df[trustpresence_col].astype(int)
else:
    df['trustpilot_has_reviews'] = 0

# SiteJabber presence
sitejabber_col = None
for c in df.columns:
    if 'sitejabber' in c.lower():
        sitejabber_col = c
        break
if sitejabber_col:
    df['sitejabber_has_reviews'] = df[sitejabber_col].astype(int)
else:
    df['sitejabber_has_reviews'] = 0

# ---------------------------
# Final feature list selection (keep engineered + a set of original useful)
# ---------------------------
# Candidate features (engineered + selected original columns)
features = []

# numeric raw features (if present)
candidates_raw = [
    col_domain_len,
    col_top_domain_len,
    col_num_digits,
    col_num_letters,
    col_num_dots,
    col_num_hyphens
]
for c in candidates_raw:
    if c and c not in features:
        features.append(c)

# Add engineered densities
features += [
    'digit_density', 'hyphen_density', 'dot_density', 'letter_ratio'
]

# Payments and counts
features += ['num_payment_methods', 'Presence of crypto currency' if any('crypto' in cc.lower() for cc in df.columns) else None]
# clean None
features = [f for f in features if f is not None]

# Add presence flags
# SSL issuer ID numeric column detection (list item)
ssl_issuer_id = None
for c in df.columns:
    if 'issuer organization list item' in c.lower() or 'issuer organization' in c.lower() and 'list' in c.lower() or 'ssl certificate issuer organization list item' in c.lower():
        ssl_issuer_id = c
        break
# Fallback detect column with values 1..11 as earlier: check ints with small unique set
if not ssl_issuer_id:
    for c in df.columns:
        if df[c].dtype in [np.int64, np.int32] and df[c].nunique() <= 20 and 'issuer' in c.lower():
            ssl_issuer_id = c
            break

if ssl_issuer_id:
    features.append(ssl_issuer_id)

# Add SSL days and registration days
features += ['days_until_ssl_expiry', 'days_since_registration']

# Tranco features
features += ['is_in_tranco', 'tranco_rank_log']

# Email / logo / young domain / trust flags / sitejabber
features += ['has_free_email', 'has_logo', 'young_domain', 'trustpilot_has_reviews', 'TrustPilot_score_clean', 'sitejabber_has_reviews']

# Add presence of prefix 'www' if exists
for c in df.columns:
    if "presence of prefix" in c.lower() or "www" in c.lower():
        features.append(c)
        break

# Deduplicate features and keep columns that exist in df
features = [f for i, f in enumerate(features) if f and f not in features[:i] and f in df.columns]
print("Final features planned ({}):".format(len(features)), features)

# ---------------------------
# Preprocessing pipeline
# ---------------------------
# We'll separate numeric vs categorical features
numeric_features = [f for f in features if df[f].dtype in [np.float64, np.int64] or df[f].dtype == 'float' or df[f].dtype == 'int']
# treat 'young_domain' as categorical; ensure any small-int categorical columns are listed as categorical if needed
categorical_features = [f for f in features if f not in numeric_features]

# Ensure TrustPilot_score_clean is numeric
if 'TrustPilot_score_clean' in df.columns and 'TrustPilot_score_clean' not in numeric_features:
    numeric_features.append('TrustPilot_score_clean')
    if 'TrustPilot_score_clean' in categorical_features:
        categorical_features.remove('TrustPilot_score_clean')

print("Numeric features:", numeric_features)
print("Categorical features (to encode):", categorical_features)

# Imputers and transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
], remainder='drop')

# ---------------------------
# Train/test split
# ---------------------------
X = df[features].copy()
y = df['target'].copy()

# For safety drop rows with missing target
mask = y.notnull()
X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print("Train/Test sizes:", X_train.shape, X_test.shape)

# ---------------------------
# Helper: function to fit, tune, evaluate a model pipeline
# ---------------------------
def evaluate_and_fit_model(model_name, estimator, param_distributions=None, n_iter=25):
    """
    estimator: estimator object (unfitted)
    param_distributions: dict for RandomizedSearchCV (parameters relative to the 'clf' step)
    returns: best_estimator (pipeline), best_score, cv_results
    """
    steps = [('preprocessor', preprocessor), ('clf', estimator)]
    pipe = Pipeline(steps=steps)

    if param_distributions:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        search = RandomizedSearchCV(
            pipe,
            param_distributions,
            n_iter=n_iter,
            scoring='f1',
            n_jobs=-1,
            cv=cv,
            verbose=1,
            random_state=RANDOM_STATE,
            return_train_score=False
        )
        print(f"\nRunning RandomizedSearchCV for {model_name} ...")
        search.fit(X_train, y_train)
        best = search.best_estimator_
        print(f"{model_name} best params:", search.best_params_)
        cv_results = search.cv_results_
    else:
        print(f"\nFitting {model_name} without hyperparam search ...")
        pipe.fit(X_train, y_train)
        best = pipe
        cv_results = None

    # Evaluate on test set
    y_pred = best.predict(X_test)
    y_proba = None
    if hasattr(best, "predict_proba"):
        y_proba = best.predict_proba(X_test)[:, 1]
    elif hasattr(best, "decision_function"):
        try:
            # scale decision function to (0,1) using logistic
            import scipy.special
            y_proba = scipy.special.expit(best.decision_function(X_test))
        except Exception:
            y_proba = None

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None
    }

    print(f"\n=== {model_name} Test Metrics ===")
    print("Accuracy:", metrics['accuracy'])
    print("F1:", metrics['f1'])
    print("Precision:", metrics['precision'])
    print("Recall:", metrics['recall'])
    if metrics['roc_auc'] is not None:
        print("ROC AUC:", metrics['roc_auc'])
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return best, metrics, cv_results

# ---------------------------
# Train Logistic Regression
# ---------------------------
lr = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, max_iter=1000)
lr_params = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l2']
}
best_lr, metrics_lr, _ = evaluate_and_fit_model("LogisticRegression", lr, param_distributions=lr_params, n_iter=8)

# ---------------------------
# Train Random Forest
# ---------------------------
rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
rf_params = {
    'clf__n_estimators': [200, 500, 800],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__max_features': ['sqrt', 'log2', None]
}
best_rf, metrics_rf, _ = evaluate_and_fit_model("RandomForest", rf, param_distributions=rf_params, n_iter=12)

# ---------------------------
# Train XGBoost (if available)
# ---------------------------
best_xgb = None
metrics_xgb = None
if XGBOOST_AVAILABLE:
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1)
    xgb_params = {
        'clf__n_estimators': [300, 500],
        'clf__max_depth': [4, 6, 8],
        'clf__learning_rate': [0.02, 0.05, 0.1],
        'clf__subsample': [0.8, 1.0],
        'clf__colsample_bytree': [0.8, 1.0]
    }
    best_xgb, metrics_xgb, _ = evaluate_and_fit_model("XGBoost", xgb, param_distributions=xgb_params, n_iter=12)
else:
    print("Skipping XGBoost training because xgboost is not installed.")

# ---------------------------
# Compare models and pick best by F1
# ---------------------------
all_metrics = {
    'LogisticRegression': metrics_lr,
    'RandomForest': metrics_rf,
}
if metrics_xgb:
    all_metrics['XGBoost'] = metrics_xgb

# Print summary
print("\n=== Model comparison (test F1) ===")
for name, m in all_metrics.items():
    print(f"{name}: F1 = {m['f1']:.4f}, Precision = {m['precision']:.4f}, Recall = {m['recall']:.4f}, ROC_AUC = {m.get('roc_auc')}")

# Select best
best_model_name = max(all_metrics.items(), key=lambda x: x[1]['f1'])[0]
print("\nBest model based on test F1:", best_model_name)

if best_model_name == 'LogisticRegression':
    final_model = best_lr
elif best_model_name == 'RandomForest':
    final_model = best_rf
elif best_model_name == 'XGBoost':
    final_model = best_xgb
else:
    final_model = best_rf  # fallback

# ---------------------------
# Save the final pipeline (includes preprocessor)
# ---------------------------
OUT_DIR = "./"
os.makedirs(OUT_DIR, exist_ok=True)
model_path = os.path.join(OUT_DIR, "model.pkl")
joblib.dump(final_model, model_path)
print("Saved final model pipeline to:", model_path)

# Save metadata (feature names, model type, metrics)
metadata = {
    'model_name': best_model_name,
    'features_used': features,
    'metrics': all_metrics,
    'training_date': datetime.utcnow().isoformat() + "Z"
}
meta_path = os.path.join(OUT_DIR, "model_metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2)
print("Saved metadata to:", meta_path)

# ---------------------------
# Feature importance (if tree model)
# ---------------------------
def get_feature_names_from_preprocessor(preproc):
    """
    Extracts feature names from ColumnTransformer after fit.
    """
    # numeric feature names
    numeric_names = numeric_features
    # categorical feature names after onehot
    cat_names = []
    # find the ohe within the transformer
    for name, transformer, cols in preproc.transformers_:
        if name == 'cat':
            # transformer is a pipeline: imputer -> onehot
            ohe = transformer.named_steps['onehot']
            cat_cols = cols
            try:
                ohe_features = list(ohe.get_feature_names_out(cat_cols))
            except Exception:
                # fallback for older sklearn
                ohe_features = []
                for i, c in enumerate(cat_cols):
                    # approximate names
                    ohe_features.append(f"{c}_ohe_{i}")
            cat_names = ohe_features
    return numeric_names + cat_names

try:
    preproc = final_model.named_steps['preprocessor']
    feat_names = get_feature_names_from_preprocessor(preproc)
    print("Total features after preprocessing:", len(feat_names))
except Exception:
    feat_names = features
    print("Could not extract feature names from pipeline; using original feature list length:", len(feat_names))

# If final model is tree-based, show importances
if best_model_name in ['RandomForest', 'XGBoost']:
    try:
        clf = final_model.named_steps['clf']
        importances = getattr(clf, 'feature_importances_', None)
        if importances is not None:
            # align with names
            fi = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
            print("\nTop 15 feature importances (model):")
            for name, val in fi[:15]:
                print(f"{name}: {val:.5f}")
    except Exception as e:
        print("Could not extract feature importances:", e)

print("\nTraining script finished. Files generated in ./:")
print("- model.pkl (final pipeline)")
print("- model_metadata.json (training summary)")
