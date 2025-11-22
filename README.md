
# E-commerce Fraud Risk Scoring

This repository contains the code, trained model, sample data, and documentation for the project **"Understanding Fraud Signals in E-Commerce: An Interpretable Machine Learning Study."** The study analyzes structural, security, and reputation signals of online shops to detect fraudulent sites using interpretable machine learning. The final deliverable includes a production-ready scoring tool that outputs a fraud probability, a 0–100 risk score, and a tiered risk label (Low / Medium / High).

Full article: [Scribd](https://www.scribd.com/embeds/953317510/content?start_page=1&view_mode=scroll&access_key=key-zKFVa8fKWWQdvcgvxrp1)

---

## 🚩 Table of Contents

- [Repository structure](#repository-structure)  
- [Dataset & attribution](#dataset--attribution)  
- [Quick start](#quick-start)  
  - [Requirements](#requirements)  
  - [Installation](#installation)  
  - [Usage: Fraud Risk Scoring CLI](#usage-fraud-risk-scoring-cli)  
- [Model & artifacts](#model--artifacts)  
- [How scoring works](#how-scoring-works)  
- [Ethics & limitations](#ethics--limitations)  
- [License & citation](#license--citation)  
- [Contact](#contact)

---

## Repository structure

├── train/  
│ ├── fraudulent_online_shops_dataset.csv # original dataset
│ ├── model.pkl # serialized pipeline (preprocessor + model)  
│ ├── model_metadata.json # features_used, model_name, metrics, timestamp  
│ └── train_model.py # training file used to reproduce models  
│  
├── score_generator.py # CLI scoring tool (single & batch mode)  
├── sample_shops.csv # example CSV input (feature order template)  
├── requirements.txt # python dependencies
└── README.md

---

## Dataset & attribution

**Dataset used:**  
Fraudulent and Legitimate Online Shops Dataset (Mendeley Data)  
URL: https://data.mendeley.com/datasets/m7xtkx7g5m/1  
**Original contributors:** Audronė Janavičiūtė, Agnius Liutkevičius  
**Published:** 22 December 2023  
**License:** CC BY 4.0

---

## Quick start

### Requirements

Python 3.8+ is required (3.10 recommended).
```bash
pip install -r requirements.txt
```

### Installation

Clone the repository:
```bash
git clone https://github.com/your-repo/ecom-fraud-detection.git
cd ecom-fraud-detection
```

### Usage: Fraud Risk Scoring CLI

**Development & Training**
To reproduce or retrain the model, execute the `train_model.py` file.
```bash
cd train
python train_model.py
```
The notebook includes:
-   EDA steps
-   Feature engineering code
-   Model training & hyperparameter search
-   Model export (`model.pkl`) and metadata generation

If you retrain and change `features_used`, update `model_metadata.json` accordingly so scoring remains consistent.

**Scoring Tool**

1. Single Record Scoring (JSON input)
Use single quotes outside, double quotes inside — no escaping needed:
	```bash
	python score_generator.py --single '{"Domain length": 20, "Top domain length": 3, "Number  of digits": 1, "Number  of letters": 22, "Number  of dots (.)": 2, "Number  of hyphens (-)": 0, "digit_density": 0.05, "hyphen_density": 0.0, "dot_density": 0.1, "letter_ratio": 1.1, "num_payment_methods": 2, "Presence of crypto currency": 0, "SSL certificate issuer organization list item": 2, "days_until_ssl_expiry": 45, "days_since_registration": 380, "is_in_tranco": 0, "tranco_rank_log": 0.0, "has_free_email": 1, "has_logo": 1, "young_domain": "1", "trustpilot_has_reviews": 0, "TrustPilot_score_clean": null, "sitejabber_has_reviews": 0, "Presence of prefix 'www' ": 1}' --json
	```
	> Windows CMD: escape double quotes with \\" or use PowerShell single-quote variant.
	
	Example JSON output:
	```json
	{
	  "probability_fraud": 0.9435,
	  "risk_score_0_100": 94.35,
	  "risk_tier": "High Risk"
	}
	```

2. Batch scoring (CSV)
Prepare sample_shops.csv matching feature names (see Input format below). Then:
	```bash
	python score_generator.py --input sample_shops.csv --output scored_results.csv
	```
	- This prints a concise per-row summary to the console.
	- If --output is provided, a CSV with appended columns probability_fraud, risk_score_0_100, and risk_tier is saved.

---

## Model & artifacts

-   **Trained model:** `model.pkl`  
    Contains the full scikit-learn pipeline: preprocessing (imputation, scaling, encoding) + trained classifier.
    
-   **Metadata:** `model_metadata.json`  
    Includes:
    
    -   `model_name` (e.g., `"RandomForest"`)
    -   `features_used` (list of required features)
    -   `metrics` (test-set evaluation)
    -   `training_date`

Keep `model_metadata.json` alongside `model.pkl` so `score_generator.py` can validate input schema and report expected features.

---

## How scoring works

1.  **Input validation & normalization**  
    The CLI normalizes header whitespace and maps known header variants (for robust ingestion).
    
2.  **Schema validation**  
    The script checks that all features in `model_metadata.json` are present; if missing, it fails with a helpful message.
    
3.  **Preprocessing + prediction**  
    The `model.pkl` pipeline performs imputation, encoding, scaling (where needed), and prediction.
    
4.  **Risk mapping**
    
    -   `probability_fraud = model.predict_proba(...)[:,1]`
    -   `risk_score_0_100 = probability_fraud * 100`
    -   `risk_tier`:
        -   High Risk: `>= 80`
        -   Medium Risk: `30–79`
        -   Low Risk: `< 30`

---

## Ethics & limitations

**Limitations**

-   Dataset size is modest; results are strong but should be validated on additional, more recent data.
-   The model uses static metadata (domain registration, SSL, TrustPilot) and does not inspect site content or transactional behavior.
-   Fraud tactics evolve; periodic retraining and monitoring are required.
    

**Ethical use**

-   Do not use the tool as the sole, final arbiter for punitive actions. Always include manual review paths.
-   Provide transparency to affected parties and allow appeal or verification steps.
-   Respect robots.txt and legal constraints when collecting additional data.

---

## License & citation

-   **Repository code:** MIT License (place `LICENSE` file in repo if publishing).
-   **Dataset:** CC BY 4.0
    
**Suggested citation (APA-like):**  
Janavičiūtė, A., & Liutkevičius, A. (2023). _Fraudulent and Legitimate Online Shops Dataset_ (Version 1) [Data set]. Mendeley Data. [https://data.mendeley.com/datasets/m7xtkx7g5m/1](https://data.mendeley.com/datasets/m7xtkx7g5m/1)

---

## Contact

If you have questions, issues, or contributions, open an issue or contact the repository maintainer.