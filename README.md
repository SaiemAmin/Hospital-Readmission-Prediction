# Hospital-Readmission-Prediction

## Healthcare Analytics Dashboard
https://hospital-readmission-prediction-bq6knu6c6ttivx62umg7sk.streamlit.app/

### Overview
This project focuses on **hospital readmissions** through comprehensive **Exploratory Data Analysis (EDA)** and **Machine Learning** models. Using **Streamlit**, the application provides interactive visualizations, filtering capabilities, and predictive functionalities for healthcare professionals and data enthusiasts alike.

### Project Features

1. **Exploratory Data Analysis (EDA)**  
   - **Dynamic Visualizations**: Line plots, bar charts, and pie charts to understand hospital stay durations and readmission rates.  
   - **Age-Based Filtering**: Interactively select age ranges for targeted insights.

2. **Machine Learning Models**  
   - **Logistic Regression**: A straightforward classification approach for predicting readmissions.  
   - **Decision Tree**: An interpretable model with adjustable depth and criterion (Gini or Entropy).  
   - **Random Forest**: An ensemble method offering feature importance insights.

3. **Interactive Features**  
   - **Custom Input Parameters**: Provide inpatient, outpatient, emergency visits, and hospital stay duration for on-the-fly predictions.  
   - **Model Evaluation**: Seamlessly switch between models to compare metrics and results.

4. **Visualizations**  
   - **Correlation Heatmap**: Identify relationships among variables.  
   - **Decision Tree Visualization**: Graphical interpretation of decision splits.  
   - **ROC Curve**: Evaluate model performance.  
   - **Feature Importance**: Understand which factors most influence the Random Forest model.

### Live Dashboard
[Hospital-Readmission-Prediction](https://hospital-readmission-prediction-bq6knu6c6ttivx62umg7sk.streamlit.app/)  

### Project Structure


## Project Structure

```
├── BackEnd.py             # Backend logic for data preprocessing and machine learning models
├── frontend.py            # Streamlit application for interactive visualizations and predictions
├── hospital_readmissions.csv # Dataset for analysis
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---


### Usage

1. **Dataset**  
   - Source: [Hospital Readmissions Dataset (Kaggle)](https://www.kaggle.com/).  
   - **Features**:  
     - `n_inpatient`: Number of inpatient visits  
     - `n_outpatient`: Number of outpatient visits  
     - `n_emergency`: Number of emergency visits  
     - `time_in_hospital`: Days spent in hospital  

2. **Dashboard Sections**  
   - **Introduction**: EDA overview with interactive age filtering.  
   - **Logistic Regression**: Model performance, classification reports, and user inputs for prediction.  
   - **Decision Tree**: Hyperparameter tuning (depth, criterion), tree visualization, and predictions.  
   - **Random Forest**: Feature importance analysis and prediction interface.

### Key Dependencies
- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

### Future Enhancements
- **Multi-Hospital Analysis**: Incorporate datasets from multiple hospitals.
- **Additional Models**: Experiment with Gradient Boosting, XGBoost, etc.
- **Hyperparameter Tuning**: Integrate direct tuning from the dashboard interface.

### Contact
For questions, feedback, or contributions, please reach out:
- **Name**: Saiem Ahmed Amin  
- **Email**: [saiemamin708@gmail.com](mailto:saiemamin708@gmail.com)
