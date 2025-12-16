# Census Income Classification & Customer Segmentation

A machine learning project that predicts high-income individuals (≥$50K annual income) and segments customers for targeted marketing campaigns using 1994-1995 US Census Bureau data. The analysis combines a Random Forest classifier for precision targeting with outcome-driven segmentation for interpretable market segments.

---

## Files Included

- **`income_classification_model.ipynb`** - Random Forest classifier for predicting high-income individuals (≥$50K). Achieves 0.919 ROC-AUC with 86.2% recall.

- **`outcome_driven_segmentation.ipynb`** - Customer segmentation model identifying 6 distinct market segments based on education, occupation, age, and work patterns.

- **`comprehensive_census_analysis.ipynb`** - Full exploratory data analysis (optional reference). Documents data quality issues, interaction effects, and key patterns.

- **`census-bureau.data`** - Census dataset containing 199,523 individual records with 40+ demographic and employment variables.

- **`census-bureau.columns`** - Column name definitions for the census dataset.

- **`Executive_Summary_UPDATED.docx`** - Executive summary with analysis approach, key findings, model performance, and deployment recommendations.

- **`README.md`** - This file.

---

## Requirements

**Python Version:** 3.8 or higher

**Required Packages:**
```
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
python-docx>=0.8.11
```

---

## Installation Instructions

1. **Clone or download this project** to your local machine

2. **Verify Python installation:**
   ```bash
   python --version  # Should show Python 3.8 or higher
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

---

## How to Run

Each notebook is self-contained and can be executed independently. Run cells sequentially from top to bottom.

**Recommended execution order:**

1. **`comprehensive_census_analysis.ipynb`** (Optional)
   - Exploratory data analysis showing data quality issues, interaction effects, and key patterns
   - Execution time: ~2-3 minutes
   - Output: Data profiling statistics, visualizations, pattern analysis

2. **`income_classification_model.ipynb`** (Required)
   - Trains Random Forest classifier and evaluates performance
   - Execution time: ~1-2 minutes
   - Output: Model metrics, confusion matrix, ROC curve, feature importance chart

3. **`outcome_driven_segmentation.ipynb`** (Required)
   - Generates 6 customer segments with behavioral profiles
   - Execution time: ~1-2 minutes
   - Output: Segment definitions, size distributions, targeting recommendations

**Note:** All notebooks include detailed markdown explanations and inline comments documenting analytical decisions.

---

## Key Findings Summary

- **Sample weight bias:** Analysis revealed 2.3x underrepresentation of high-income individuals in the dataset, requiring weighted training to correct sampling bias.

- **Strongest predictor:** "Weeks worked in year" emerged as the most important feature (31.6% importance), followed by education (25.4%) and occupation (18.9%), validating the work intensity hypothesis.

- **Classifier performance:** Random Forest achieves 0.919 ROC-AUC with 86.2% recall, but precision is only 23.7% at default threshold. Requires threshold tuning based on campaign costs before deployment.

- **Segmentation results:** Identified 6 customer segments with "Early Career Builders" (13% of market, 15.8% high-income rate) and "Credentialed Professionals" (9% of market, 28.4% high-income rate) as primary targets.

---

## Model Performance Summary

### Classification Model (Random Forest)
- **ROC-AUC:** 0.919 (strong discrimination ability)
- **Recall (Sensitivity):** 86.2% (captures majority of high-income individuals)
- **Precision:** 23.7% (high false positive rate at default 0.5 threshold)
- **Recommendation:** Adjust threshold based on campaign costs (0.7-0.8 for expensive campaigns, 0.3-0.4 for low-cost digital campaigns)

### Segmentation Model (Rule-Based)
- **Segments:** 6 outcome-driven customer groups
- **Approach:** Education × occupation × age × work consistency interactions
- **Top segments:**
  - Credentialed Professionals: 28.4% high-income rate (9% of market)
  - Early Career Builders: 15.8% high-income rate (13% of market)
  - Blue-Collar Majority: 2.1% high-income rate (47% of market, avoid targeting)

---

## Project Structure

```
JPMC_Takehome/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── Executive_Summary_UPDATED.docx         # Analysis report
├── census-bureau.data                     # Dataset (199,523 rows)
├── census-bureau.columns                  # Column definitions
├── income_classification_model.ipynb      # Classifier notebook
├── outcome_driven_segmentation.ipynb      # Segmentation notebook
└── comprehensive_census_analysis.ipynb    # EDA notebook
```

---

## Contact Information

**Author:** Siddhant Evre
**Email:** sidevre@gmail.com
**Date Completed:** December 2025


---

## Notes and Limitations

- **Data age:** Census data from 1994-1995 is 30+ years old. Requires retraining on current data for production deployment.

- **Threshold calibration:** Classification model uses default 0.5 probability threshold. Recommend business cost analysis to determine optimal precision/recall tradeoff.

- **Recall vs. Precision tradeoff:** Current model prioritizes recall (86.2%) over precision (23.7%), meaning it captures most high-income individuals but with significant false positives. Acceptable for low-cost campaigns, requires tuning for expensive targeting.

- **Feature drift monitoring:** Occupation codes and educational categories evolve over time. Set up monitoring for distribution drift before deployment.

- **Fairness audit:** Conduct bias assessment across protected attributes (race, sex, age) before production use.

- **Deployment recommendation:** Combine segmentation (interpretability) + classification (precision) in two-stage workflow. See Executive Summary for detailed deployment plan.

---

## License

This project was created as part of a JPMC data science take-home assessment. Data sourced from US Census Bureau public records.
