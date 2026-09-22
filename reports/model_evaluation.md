# Model Evaluation Report

**Number of training samples:** 126
**Number of test samples:** 36

**Features Used:**
- initial_moisture
- final_moisture
- moisture_gain
- thickness_micron
- wvtr
- otr
- evidence_quality
- packaging_material

**Target Used:** recommendation_status
**Final Model Selected:** Random Forest

## Metrics
```text
                           precision    recall  f1-score   support

Conditionally Recommended       1.00      1.00      1.00         1
          Not Recommended       1.00      1.00      1.00         3
              Recommended       1.00      1.00      1.00        32

                 accuracy                           1.00        36
                macro avg       1.00      1.00      1.00        36
             weighted avg       1.00      1.00      1.00        36
```

## Model Performance & Limitations
**The Random Forest achieved 96.17% mean accuracy under 5-fold product-grouped cross-validation. On the fixed grouped holdout test set, it achieved 100% accuracy.**

**Both metrics evaluate agreement with deterministic rule-generated labels, not experimentally measured packaging compatibility.**

The model achieved the reported evaluation performance on the available compatibility-labelled dataset. Because the labels were generated using deterministic rules rather than independent experimental ground truth, these metrics indicate agreement with the existing decision logic and should not be interpreted as independently validated real-world packaging compatibility accuracy.

1. **Rule-Based Ground Truth:** The model approximates deterministically generated labels from the original rule engine.
2. **Small Dataset Size:** The dataset has limited representation of the `Conditionally Recommended` class, and is evaluated using a grouped split to avoid product leakage.
3. **Missing Features:** Important features like `storage_temperature_c` and `relative_humidity_percent` were completely unavailable in the training dataset and thus are excluded from the model.

## Conclusion
The model effectively replicates the deterministic rules, making it suitable for a **prototype demonstration** of an AI-assisted decision-support system. It is not ready for production deployment without real-world empirical validation.
