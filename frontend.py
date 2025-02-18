import streamlit as st

# Must be the very first Streamlit command
st.set_page_config(
    page_title="Healthcare Analytics Dashboard",
    page_icon=":hospital:",
    layout="wide"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------------
# 1) Data Loading and Preprocessing
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Load the hospital readmissions CSV file and convert certain columns to numeric.
    """
    df = pd.read_csv("hospital_readmissions.csv")
    df["readmitted"] = df["readmitted"].map({"yes": 1, "no": 0})
    df["diabetes_med"] = df["diabetes_med"].map({"yes": 1, "no": 0})
    return df

def convert_age_to_numeric(df):
    """
    Convert age ranges like '70-80' to their numeric midpoints.
    """
    age_range = df['age'].str.extract(r'(\d+)-(\d+)')
    df['age'] = age_range.astype(float).mean(axis=1)
    return df

# -------------------------------------------------------------------
# 2) Plotly-based EDA Functions
# -------------------------------------------------------------------
def plot_line(df, x_col, y_col, title="Line Plot", width=400, height=300):
    fig = px.line(df, x=x_col, y=y_col, title=title, markers=True)
    fig.update_layout(width=width, height=height)
    return fig

def plot_bar(df, x_col, y_col, title="Bar Plot", width=400, height=300):
    fig = px.bar(df, x=x_col, y=y_col, title=title)
    fig.update_layout(width=width, height=height)
    return fig

def plot_pie(df, names_col, title="Pie Chart", width=400, height=300):
    fig = px.pie(df, names=names_col, title=title)
    fig.update_layout(width=width, height=height)
    return fig

def plot_hist(df, x_col, bins=20, title="Histogram", width=400, height=300):
    fig = px.histogram(df, x=x_col, nbins=bins, title=title)
    fig.update_layout(width=width, height=height)
    return fig

def plot_scatter(df, x_col, y_col, color_col=None, title="Scatter Plot", width=400, height=300):
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
    fig.update_layout(width=width, height=height)
    return fig

def plot_correlation_heatmap(df, width=400, height=300):
    """
    Create a Plotly heatmap for the correlation matrix.
    Only numeric columns are used to avoid conversion errors.
    """
    df_numeric = df.select_dtypes(include=["number"])
    corr = df_numeric.corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="RdBu",
            zmid=0
        )
    )
    fig.update_layout(title="Correlation Heatmap", width=width, height=height)
    return fig

# -------------------------------------------------------------------
# 3) Machine Learning Model Functions
# -------------------------------------------------------------------
def logistic_regression_model(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, train_acc, test_acc, cm, report

def decision_tree_model(X_train, y_train, X_test, y_test, max_depth=3, criterion="gini"):
    dt_model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    train_acc = dt_model.score(X_train, y_train)
    test_acc = dt_model.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return dt_model, train_acc, test_acc, cm, report

def random_forest_model(X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    train_acc = rf_model.score(X_train, y_train)
    test_acc = rf_model.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return rf_model, train_acc, test_acc, cm, report

def plot_confusion_matrix_plotly(cm, labels, title="Confusion Matrix", width=400, height=300):
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size":10}
        )
    )
    fig.update_layout(title=title, width=width, height=height)
    return fig

def plot_feature_importance(model, feature_names, title="Feature Importance", width=400, height=300):
    importances = model.feature_importances_
    fig = px.bar(x=feature_names, y=importances, title=title)
    fig.update_layout(width=width, height=height)
    return fig

# -------------------------------------------------------------------
# 4) Main App Setup
# -------------------------------------------------------------------
df = load_data()
df = convert_age_to_numeric(df)
selected_features = ["n_inpatient", "n_outpatient", "n_emergency", "time_in_hospital"]
X = df[selected_features]
y = df["readmitted"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar Navigation
pages = st.sidebar.radio(
    "Navigation",
    ["Introduction", "Data Overview", "Model Comparison", "Logistic Regression", "Decision Tree", "Random Forest", "About"]
)

# -------------------------------------------------------------------
# 5) Pages
# -------------------------------------------------------------------
if pages == "Introduction":
    st.title("Healthcare Analytics Dashboard (Plotly Edition)")
    st.markdown("## Exploratory Data Analysis")
    
    st.subheader("Filter Data by Age")
    min_age, max_age = int(df['age'].min()), int(df['age'].max())
    age_range = st.slider("Select Age Range", min_value=min_age, max_value=max_age, value=(30, 60), step=5)
    filtered_df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
    st.write(f"Displaying rows for age between {age_range[0]} and {age_range[1]}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Line Plot", "Bar Plot", "Pie Chart", "Scatter Plot", "Histogram"])
    
    with tab1:
        st.subheader("Line Plot: Avg Time in Hospital by Age")
        line_df = filtered_df.groupby("age", as_index=False)["time_in_hospital"].mean()
        fig_line = plot_line(line_df, x_col="age", y_col="time_in_hospital", title="Avg Time in Hospital", width=500, height=350)
        st.plotly_chart(fig_line, use_container_width=False)
    
    with tab2:
        st.subheader("Bar Plot: Avg Time in Hospital by Age")
        bar_df = filtered_df.groupby("age", as_index=False)["time_in_hospital"].mean()
        fig_bar = plot_bar(bar_df, x_col="age", y_col="time_in_hospital", title="Avg Time in Hospital", width=500, height=350)
        st.plotly_chart(fig_bar, use_container_width=False)

    with tab3:
        st.subheader("Pie Chart: Readmission Proportions")
        pie_df = filtered_df["readmitted"].value_counts().reset_index()
        pie_df.columns = ["readmitted", "count"]
        fig_pie = px.pie(pie_df, names="readmitted", values="count", title="Readmission Distribution", width=500, height=350)
        fig_pie.update_traces(textinfo='percent+label', hovertemplate='Readmitted: %{label} <br>Count: %{value}')
        st.plotly_chart(fig_pie, use_container_width=False)

    with tab4:
        st.subheader("Scatter Plot")
        x_axis = st.selectbox("Select X-axis", options=selected_features, index=0)
        y_axis = st.selectbox("Select Y-axis", options=selected_features, index=1)
        fig_scatter = plot_scatter(filtered_df, x_col=x_axis, y_col=y_axis, color_col="readmitted", title="Scatter Plot", width=500, height=350)
        st.plotly_chart(fig_scatter, use_container_width=False)

    with tab5:
        st.subheader("Histogram")
        col_choice = st.selectbox("Select Column", ["age", "time_in_hospital", "n_inpatient", "n_outpatient", "n_emergency"])
        bins = st.slider("Number of bins", 5, 50, 20)
        fig_hist = plot_hist(filtered_df, x_col=col_choice, bins=bins, title="Histogram", width=500, height=350)
        st.plotly_chart(fig_hist, use_container_width=False)

elif pages == "Data Overview":
    st.title("Data Overview")
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10))
    
    st.markdown("### Dataset Summary")
    if st.checkbox("Show Summary Statistics"):
        st.write(df.describe())
    
    st.markdown("### Missing Values")
    missing = df.isnull().sum()
    st.bar_chart(missing)
    
    st.markdown("### Correlation Heatmap")
    fig_corr = plot_correlation_heatmap(df, width=500, height=350)
    st.plotly_chart(fig_corr, use_container_width=False)
    
    st.markdown("### Download Data")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download CSV", data=csv_data, file_name="hospital_readmissions.csv", mime="text/csv")

elif pages == "Model Comparison":
    st.title("Model Comparison")
    st.markdown("Train and compare multiple models on the same data split.")
    
    lr_model, lr_train, lr_test, lr_cm, lr_report = logistic_regression_model(X_train, y_train, X_test, y_test)
    dt_model, dt_train, dt_test, dt_cm, dt_report = decision_tree_model(X_train, y_train, X_test, y_test)
    rf_model, rf_train, rf_test, rf_cm, rf_report = random_forest_model(X_train, y_train, X_test, y_test)
    
    summary_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Train Accuracy": [lr_train, dt_train, rf_train],
        "Test Accuracy": [lr_test, dt_test, rf_test]
    })
    st.table(summary_df)
    
    st.markdown("### Detailed Classification Reports")
    with st.expander("Logistic Regression Report"):
        st.dataframe(pd.DataFrame(lr_report).transpose())
    with st.expander("Decision Tree Report"):
        st.dataframe(pd.DataFrame(dt_report).transpose())
    with st.expander("Random Forest Report"):
        st.dataframe(pd.DataFrame(rf_report).transpose())

elif pages == "Logistic Regression":
    st.title("Logistic Regression")
    model, train_acc, test_acc, cm, report = logistic_regression_model(X_train, y_train, X_test, y_test)
    st.metric("Train Accuracy", f"{train_acc:.3f}")
    st.metric("Test Accuracy", f"{test_acc:.3f}")
    
    st.markdown("### Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())
    
    st.markdown("### Confusion Matrix")
    fig_cm = plot_confusion_matrix_plotly(cm, labels=["Not Readmitted", "Readmitted"], width=400, height=300)
    st.plotly_chart(fig_cm, use_container_width=False)
    
    st.markdown("### Make a Prediction")
    with st.form("lr_pred"):
        col1, col2, col3, col4 = st.columns(4)
        n_inpatient = col1.number_input("Inpatient Visits", 0, 20, 1)
        n_outpatient = col2.number_input("Outpatient Visits", 0, 20, 1)
        n_emergency = col3.number_input("Emergency Visits", 0, 20, 1)
        time_in_hospital = col4.number_input("Time in Hospital", 1, 30, 3)
        submit_lr = st.form_submit_button("Predict")
        if submit_lr:
            pred = model.predict([[n_inpatient, n_outpatient, n_emergency, time_in_hospital]])
            result = "Readmitted" if pred[0] == 1 else "Not Readmitted"
            st.success(f"Prediction: {result}")

elif pages == "Decision Tree":
    st.title("Decision Tree")
    st.markdown("#### Tune Hyperparameters")
    max_depth = st.slider("Max Depth", 1, 10, 3)
    criterion = st.selectbox("Criterion", ["gini", "entropy"])
    
    dt_model, train_acc, test_acc, cm, report = decision_tree_model(X_train, y_train, X_test, y_test, max_depth, criterion)
    st.metric("Train Accuracy", f"{train_acc:.3f}")
    st.metric("Test Accuracy", f"{test_acc:.3f}")
    
    st.markdown("### Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())
    
    st.markdown("### Confusion Matrix")
    fig_cm = plot_confusion_matrix_plotly(cm, labels=["Not Readmitted", "Readmitted"], width=400, height=300)
    st.plotly_chart(fig_cm, use_container_width=False)
    
    st.markdown("### Make a Prediction")
    with st.form("dt_pred"):
        col1, col2, col3, col4 = st.columns(4)
        n_inpatient = col1.number_input("Inpatient Visits", 0, 20, 1, key="dt1")
        n_outpatient = col2.number_input("Outpatient Visits", 0, 20, 1, key="dt2")
        n_emergency = col3.number_input("Emergency Visits", 0, 20, 1, key="dt3")
        time_in_hospital = col4.number_input("Time in Hospital", 1, 30, 3, key="dt4")
        submit_dt = st.form_submit_button("Predict")
        if submit_dt:
            pred = dt_model.predict([[n_inpatient, n_outpatient, n_emergency, time_in_hospital]])
            result = "Readmitted" if pred[0] == 1 else "Not Readmitted"
            st.success(f"Prediction: {result}")

elif pages == "Random Forest":
    st.title("Random Forest")
    rf_model, train_acc, test_acc, cm, report = random_forest_model(X_train, y_train, X_test, y_test)
    st.metric("Train Accuracy", f"{train_acc:.3f}")
    st.metric("Test Accuracy", f"{test_acc:.3f}")
    
    st.markdown("### Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())
    
    st.markdown("### Confusion Matrix")
    fig_cm = plot_confusion_matrix_plotly(cm, labels=["Not Readmitted", "Readmitted"], width=400, height=300)
    st.plotly_chart(fig_cm, use_container_width=False)
    
    st.markdown("### Feature Importance")
    if hasattr(rf_model, "feature_importances_"):
        fig_fi = plot_feature_importance(rf_model, selected_features, width=400, height=300)
        st.plotly_chart(fig_fi, use_container_width=False)
    
    st.markdown("### Make a Prediction")
    with st.form("rf_pred"):
        col1, col2, col3, col4 = st.columns(4)
        n_inpatient = col1.number_input("Inpatient Visits", 0, 20, 1, key="rf1")
        n_outpatient = col2.number_input("Outpatient Visits", 0, 20, 1, key="rf2")
        n_emergency = col3.number_input("Emergency Visits", 0, 20, 1, key="rf3")
        time_in_hospital = col4.number_input("Time in Hospital", 1, 30, 3, key="rf4")
        submit_rf = st.form_submit_button("Predict")
        if submit_rf:
            pred = rf_model.predict([[n_inpatient, n_outpatient, n_emergency, time_in_hospital]])
            result = "Readmitted" if pred[0] == 1 else "Not Readmitted"
            st.success(f"Prediction: {result}")

elif pages == "About":
    st.title("About This Dashboard")
    st.markdown("""
    **Plotly-based Healthcare Analytics Dashboard**  
    
    This dashboard explores a hospital readmissions dataset using:
    
    - **Plotly** for all EDA and model visualizations (line, bar, scatter, pie, hist, correlation heatmap, confusion matrix, feature importance).
    - **scikit-learn** for Logistic Regression, Decision Tree, and Random Forest models.
    
    **Features**  
    - **Introduction (EDA):** Filter by age and see various Plotly charts.  
    - **Data Overview:** View summary statistics, missing values, correlation heatmap, and download the CSV.
    - **Model Comparison:** Compare multiple machine learning models on the same data split.  
    - **Individual Model Pages:** Check performance metrics and make new predictions.
    
    ---
    *Developed for educational and healthcare analytics demonstrations.*
    """)

