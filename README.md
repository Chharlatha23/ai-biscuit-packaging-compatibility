# AI-Based Biscuit Packaging Compatibility Prediction System

## 1. Project Title
AI-Based Intelligent Food Packaging Material Recommendation System for Biscuits

## 2. Project Overview
This project is an ML-powered decision-support application designed to estimate and predict the compatibility of various food packaging materials for biscuits based on their properties and storage conditions.

## 3. Problem Statement
Selecting the optimal packaging material for biscuits is critical to prevent moisture gain and maintain shelf-life. Inappropriate packaging leads to spoilage and economic loss. A systematic, data-driven approach is needed to match biscuit characteristics with appropriate packaging barriers.

## 4. Objectives
- Build a software-only AI-based decision-support system to evaluate biscuit-packaging compatibility.
- Utilize machine learning to predict compatibility using moisture and barrier properties.
- Provide a professional, interactive dashboard for easy assessment.

## 5. Key Features
- **Machine Learning Inference:** Predicts compatibility (Recommended, Conditionally Recommended, Not Recommended).
- **Interactive UI:** Professional Streamlit dashboard for real-time assessments.
- **Explainable Results:** Displays full class probability distribution.
- **Assessment History:** Session-based tracking and CSV export.
- **Reporting:** Downloadable assessment reports.

## 6. System Architecture
User Input → Input Validation → Feature Preparation / Preprocessing → Trained Random Forest Classifier → Compatibility Prediction + Class Probabilities → Streamlit Professional Dashboard

## 7. Dataset
The data includes biscuit properties (initial moisture, critical moisture), packaging properties (thickness, WVTR, OTR, material type), and evidence quality metrics. 35 biscuit products and 210 biscuit-packaging pairings were evaluated. 

## 8. Feature Engineering
Features include `initial_moisture`, `final_moisture`, derived `moisture_gain`, `thickness_micron`, `wvtr`, `otr`, `packaging_material`, and `evidence_quality`. Numeric and categorical features are processed through a scikit-learn Pipeline using `OneHotEncoder` and standard scaling.

## 9. Rule-Generated Training Labels
The deterministic compatibility framework is used to generate the training labels, while the trained Random Forest classifier performs the final prediction.

## 10. ML Model
The system uses a Random Forest Classifier to predict the compatibility category. The model outputs class probabilities along with the final recommendation status.

## 11. Grouped Evaluation Methodology
The model was validated using a 5-Fold Product-Grouped Cross-Validation strategy. This ensures that observations from the same biscuit product (`product_id`) are not split across training and validation folds, preventing data leakage.

## 12. Results
- **Mean CV Accuracy:** Approximately 96.17%
- **Fixed Grouped Holdout Test Accuracy:** 100%
- **Total Pairings:** 210 pairings across 35 products
- **Usable ML Rows:** 162 rows (excluding 'Insufficient Data' class)

## 13. Streamlit Dashboard
The UI includes:
- KPI cards and process flow
- Compatibility Assessment with live moisture gain calculations
- Model Insights (CV accuracy, confusion matrix)
- Packaging Materials reference dataset
- Assessment History (Session-based with CSV export)
- Detailed About section

## 14. Project Structure
```text
.
├── app.py                      # Streamlit application
├── ml/                         # ML pipeline (train, predict, evaluate, preprocess)
├── models/                     # Saved models (joblib files, metadata)
├── processed_data/             # Processed datasets
├── reports/                    # Generated EDA and evaluation reports
├── src/                        # Source integration modules
├── tests/                      # pytest test suite
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## 15. Installation
Ensure Python 3.12+ is installed. Clone the repository and install requirements:
```bash
git clone https://github.com/Chharlatha23/ai-biscuit-packaging-compatibility.git
cd ai-biscuit-packaging-compatibility
pip install -r requirements.txt
```

## 16. Running Instructions
To run the Streamlit UI dashboard locally:
```bash
python -m streamlit run app.py
```
To run the training pipeline:
```bash
python ml/train_model.py
```

## 17. Testing
Tests are implemented using `pytest`. To run the test suite:
```bash
pytest -q
```
The suite confirms expected exception handling, probability bounds, deterministic reproducibility, and correct categorical encoding handling.

## 18. Limitations
- The model is trained on deterministically generated rules (approximating compatibility rules) and lacks independent empirical real-world shelf-life experimental ground truth.
- The prediction reflects the classifier's estimated probability for the available training labels, not a real-world failure rate.

## 19. Future Scope
- Validation against independent, empirical real-world shelf-life experiments.
- Expansion of the packaging knowledge base to include more complex laminate structures and sustainability metrics.
- Incorporation of accelerated shelf-life testing results directly into the ML training pipeline.
