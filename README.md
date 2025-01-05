# Hospital-Readmission-Prediction

Healthcare Analytics Dashboard

# Hospital Readmissions Analysis and Prediction

This project explores healthcare analytics, specifically focusing on hospital readmissions. The dataset is analyzed using exploratory data analysis (EDA) and machine learning models, with interactive visualizations and predictions showcased using Streamlit.

## Project Features

### 1. **Exploratory Data Analysis (EDA)**

- Dynamic visualizations such as line plots, bar charts, and pie charts to explore:
  - Average time spent in the hospital by age group.
  - Proportion of readmissions.
- Filtering data dynamically based on age ranges for better insights.

### 2. **Machine Learning Models**

- Logistic Regression: A classification model to predict hospital readmissions.
- Decision Tree: An interpretable model with adjustable depth and criteria (Gini or Entropy).
- Random Forest: A robust ensemble model with feature importance analysis.

### 3. **Interactive Features**

- Input custom parameters for predictions (e.g., inpatient visits, outpatient visits, emergency visits, and time in the hospital).
- Toggle between different machine learning models and their evaluations.

### 4. **Visualizations**

- Heatmaps to show correlation among features.
- Decision tree visualization.
- ROC Curve for model evaluation.
- Feature importance plot for Random Forest.

---
##  Dashboard
 https://hospital-readmission-prediction-bq6knu6c6ttivx62umg7sk.streamlit.app/

## Project Structure

```
├── BackEnd.py             # Backend logic for data preprocessing and machine learning models
├── frontend.py            # Streamlit application for interactive visualizations and predictions
├── hospital_readmissions.csv # Dataset for analysis
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Usage

### Dataset

The dataset used is publicly available on Kaggle:
[Hospital Readmissions Dataset](https://www.kaggle.com/datasets/dubradave/hospital-readmissions)

### Features for Prediction

- `n_inpatient`: Number of inpatient visits.
- `n_outpatient`: Number of outpatient visits.
- `n_emergency`: Number of emergency visits.
- `time_in_hospital`: Number of days spent in the hospital.

### Dashboard

The dashboard is divided into the following sections:

1. **Introduction**: Overview of EDA with interactive filtering.
2. **Logistic Regression**: Model evaluation, classification report, and prediction inputs.
3. **Decision Tree**: Interactive depth and criterion selection, tree visualization, and predictions.
4. **Random Forest**: Feature importance visualization and predictions.

---

## Key Dependencies

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

---

## Future Enhancements

- Include additional datasets for multi-hospital analysis.
- Implement additional machine learning models (e.g., Gradient Boosting).
- Add support for hyperparameter tuning directly from the dashboard.

---

---

## Contact

For any inquiries or contributions, please reach out to:

- **Saiem Ahmed Amin**
- saiemamin708@gmail.com
