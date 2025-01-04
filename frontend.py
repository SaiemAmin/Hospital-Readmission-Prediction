import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from BackEnd import (
    get_preprocessed_data,
    line_plt,
    barplot,
    pie_plot,
    logistic_regression,
    decision_tree,
    random_forest_model,
    plot_feature_importance,
    decision_tree,
    random_forest_model
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree

@st.cache_data
def load_data():
    """
    Load the dataset.
    """
    df = pd.read_csv("hospital_readmissions.csv")
    return df

def convert_age_to_numeric(df):
    """
    Convert age ranges like '[70-80)' to their midpoints.
    """
    df['age'] = df['age'].str.extract(r'(\d+)-(\d+)').astype(float).mean(axis=1)
    return df

# Title
st.title("Healthcare Analytics Dashboard")

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = st.sidebar.radio("Go to", ["Introduction", "Logistic Regression", "Decision Tree", "Random Forest"])

# Load and preprocess data
df = load_data()
df = convert_age_to_numeric(df)

# Split data
selected_features = ["n_inpatient", "n_outpatient", "n_emergency", "time_in_hospital"]
X = df[selected_features]
Y = df["readmitted"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

if pages == "Introduction":
    st.header("Exploratory Data Analysis (EDA)")
    
    # Add filters for dynamic data exploration
    age_range = st.slider("Select Age Range:",
                          min_value=int(df['age'].min()),
                          max_value=int(df['age'].max()),
                          value=(30, 60), step=5)
    filtered_df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
    st.write(f"Data for Age Range {age_range[0]} to {age_range[1]}:")

    # Line Plot for Age Group
    st.subheader("Avg Time Spent in Hospital By Age Group (Line Plot)")
    st.pyplot(line_plt(filtered_df))

    # Bar Plot for Readmissions
    st.subheader("BarPlot of Readmissions (Pie Chart)")
    st.pyplot(barplot(filtered_df))

    # Pie Plot for readmission
    st.subheader("Readmission Proportion (Pie Chart)")
    st.pyplot(pie_plot(filtered_df))


# LOGISTIC REGRESSION


elif pages == "Logistic Regression":
    st.markdown("<h2 style='color:#4CAF50;'>Logistic Regression Model</h2>", unsafe_allow_html=True)
    
    # Train the logistic regression model
    model, train_accuracy, test_accuracy, cm, report = logistic_regression()
    st.metric(label="Training Accuracy", value=f"{train_accuracy:.3f}")
    st.metric(label="Test Accuracy", value=f"{test_accuracy:.3f}")

    from sklearn.preprocessing import LabelEncoder

    # Encode Y_test labels
    label_encoder = LabelEncoder()
    Y_test_encoded = label_encoder.fit_transform(Y_test)  # Transform 'yes'/'no' to 1/0

    # Predict labels using the model
    Y_pred = model.predict(X_test)

    # Display Classification Report in Table Format
    st.subheader("Classification Report")
    report_dict = classification_report(Y_test_encoded, Y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df)

    # Dynamic Input for Predictions
    st.subheader("Make a Prediction")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_inpatient = st.number_input("Inpatient Visits", min_value=0, max_value=20, value=1)
    with col2:
        n_outpatient = st.number_input("Outpatient Visits", min_value=0, max_value=20, value=1)
    with col3:
        n_emergency = st.number_input("Emergency Visits", min_value=0, max_value=20, value=1)
    with col4:
        time_in_hospital = st.number_input("Time in Hospital (Days)", min_value=1, max_value=30, value=3)
    if st.button("Predict Readmission"):
        prediction = model.predict([[n_inpatient, n_outpatient, n_emergency, time_in_hospital]])
        result = "Readmitted" if prediction[0] == 1 else "Not Readmitted"
        st.success(f"Prediction: {result}")



elif pages == "Decision Tree":
    st.markdown("<h2 style='color:#4CAF50;'>Decision Tree Model</h2>", unsafe_allow_html=True)

    # Add interactivity with hyperparameters
    max_depth = st.slider("Max Depth", min_value=1, max_value=10, value=3)
    criterion = st.selectbox("Criterion", options=["gini", "entropy"], index=0)

    # Call the updated decision_tree function
    dt_model, train_accuracy, test_accuracy, report = decision_tree(
        max_depth=max_depth, criterion=criterion #crtierion is entropy or gini
    )

    st.metric(label="Training Accuracy", value=f"{train_accuracy:.2f}")
    st.metric(label="Test Accuracy", value=f"{test_accuracy:.2f}")

    # Display Classification Report
    st.subheader("Classification Report")
    # Convert the report dictionary into a DataFrame
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # Visualize the Decision Tree
    st.subheader("Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        dt_model,
        feature_names=selected_features,
        class_names=["Not Readmitted", "Readmitted"],
        filled=True,
        ax=ax,
    )
    st.pyplot(fig)

    st.subheader("Make a Prediction")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_inpatient = st.number_input("Inpatient Visits", min_value=0, max_value=20, value=1)
    with col2:
        n_outpatient = st.number_input("Outpatient Visits", min_value=0, max_value=20, value=1)
    with col3:
        n_emergency = st.number_input("Emergency Visits", min_value=0, max_value=20, value=1)
    with col4:
        time_in_hospital = st.number_input("Time in Hospital (Days)", min_value=1, max_value=30, value=3)
   
    if st.button("Predict Readmission"):
        prediction = dt_model.predict([[n_inpatient, n_outpatient, n_emergency, time_in_hospital]])
        result = "Readmitted" if prediction[0] == 1 else "Not Readmitted"
        st.success(f"Prediction: {result}")


# RANDOM FOREST

elif pages == "Random Forest":
    st.markdown("<h2 style='color:#4CAF50;'>Random Forest Model</h2>", unsafe_allow_html=True)
    rf_model, train_accuracy, test_accuracy, report = random_forest_model()
    st.metric(label="Training Accuracy", value=f"{train_accuracy:.2f}")
    st.metric(label="Test Accuracy", value=f"{test_accuracy:.2f}")

    st.subheader("Classification Report")
    # Convert the report dictionary into a DataFrame
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # Prediction Section
    st.subheader("Make a Prediction")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_inpatient = st.number_input("Inpatient Visits", min_value=0, max_value=20, value=1)
    with col2:
        n_outpatient = st.number_input("Outpatient Visits", min_value=0, max_value=20, value=1)
    with col3:
        n_emergency = st.number_input("Emergency Visits", min_value=0, max_value=20, value=1)
    with col4:
        time_in_hospital = st.number_input("Time in Hospital (Days)", min_value=1, max_value=30, value=3)

    if st.button("Predict Readmission"):
        prediction = rf_model.predict([[n_inpatient, n_outpatient, n_emergency, time_in_hospital]])
        result = "Readmitted" if prediction[0] == 1 else "Not Readmitted"
        st.success(f"Prediction: {result}")


