#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import pandas as ps
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import pandas as pdf


def get_preprocessed_data():
    """
    Fetch, load, and preprocess the dataset.
    Returns a pandas DataFrame.
    """


        # Load the dataset into a pandas DataFrame
    df = pd.read_csv("hospital_readmissions.csv")

        #Converting categorical values to numerical
        #Coverting readmitted column to identifiers 1/0
    df["readmitted"] = df["readmitted"].map({"yes":1, "no" : 0})
    df["diabetes_med"] = df["diabetes_med"].map({"yes": 1, "no": 0})
    df.head()

    return df  # Return the processed DataFrame


# In[3]:


df = get_preprocessed_data()

if df is not None:
    print("Dataset Preview:")
    print(df.head(5)) # Preview the first few rows
    print("Dataset Shape:", df.shape)  # Get the number of rows and columns
else:
    print("Failed to load the dataset.")


# In[4]:


numeric_df = df.select_dtypes(include= ["int64", "float64"])
binary_df = df.select_dtypes(include = ["object"])


# In[5]:


# Scaling the numerical columns
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_df = df.select_dtypes(include= ["int64", "float64"])
numerical_df = scaler.fit_transform(numerical_df)


# # EDA

# In[17]:


def line_plt(df):
    age_hospitals = df.groupby("age")["time_in_hospital"].mean()
    fig, ax = plt.subplots()
    ax.plot(age_hospitals.index, age_hospitals.values, marker="o")
    ax.set_xlabel("Age")
    ax.set_ylabel("Avg Time in Hospital")
    ax.set_title("Avg Time Spent in Hospital by Age Group")
    return fig  # Return the figure

line_plt(df)


# In[7]:


def barplot(df):
    age_hospitals = df.groupby("age")["time_in_hospital"].mean()
    fig, ax = plt.subplots()
    ax.bar(age_hospitals.index, age_hospitals.values)
    ax.set_xlabel("Age")
    ax.set_ylabel("Avg time in Hospital")
    ax.set_title("Avg time spent in hospital by Age Group")
    return fig

barplot(df)


# Time in hospital increments as age group increases. The older you get the more time you spend in hospital; however, after 80-90 age the readmission declines.

# In[8]:


def pie_plot(df):
    # Calculate readmission proportions
    readmission_counts = df['readmitted'].value_counts()

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(
        readmission_counts,
        autopct='%1.1f%%',
        startangle=90,
        labels=["No", "Yes"],
        colors=["skyblue", "lightcoral"],
        wedgeprops={"edgecolor": "black"}
    )
    ax.set_title("Proportion of Readmissions", fontsize=14)
    return fig  # Return the figure object

pie_plot(df)



# In[9]:


print(df.columns)


# In[10]:


#Using Label Encoder to convert  categorical data into numerical
from sklearn.preprocessing import LabelEncoder

for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])
        print(f"{col}:{df[col].unique()}")  


# In[11]:


def correlation_matrix(df):
    # Compute the correlation matrix
    corr_matrix = df.corr()
    
    # Plot the heatmap
    sns.heatmap(corr_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                annot_kws={"size": 7})
    plt.title("Correlation Matrix")
    plt.show()
    
    # Return the correlation matrix
    return corr_matrix

# Call the function and capture the correlation matrix
corr_matrix = correlation_matrix(df)


# In[12]:


threshold = 0.1

# Selecting highly correlated features
readmitted_features = corr_matrix["readmitted"].abs()
significant_corr = readmitted_features[readmitted_features >= threshold]

print(significant_corr.sort_values(ascending= False))


# In[13]:


# Removing Outliers using IQR method

def remove_outliers(df, cols):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR

    return df
# specify particular columns to remove outliers
cols_to_check = ['n_medications', 'n_outpatient', 'n_inpatient', 'n_emergency']

df_cleaned = remove_outliers(df, cols_to_check)


# # Train/Test Split for X,Y variables that can be used for multiple models 

# In[14]:


from sklearn.model_selection import train_test_split
selected_features = ["n_inpatient", "n_outpatient", "n_emergency", "time_in_hospital"]
X = df_cleaned[selected_features]
Y = df_cleaned["readmitted"]

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# # Logistic Regression

# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def logistic_regression():
    # selected_features = ["n_inpatient", "n_outpatient", "n_emergency", "time_in_hospital"]
    # X = df_cleaned[selected_features]
    # Y = df_cleaned["readmitted"]

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    
    train_accuracy = model.score(X_train, Y_train)
    test_accuracy = model.score(X_test, Y_test)
    cm  = confusion_matrix(Y_test, Y_pred) # Confusion Matrix
    report = classification_report(Y_test, Y_pred)
    print("Training Accuracy:",train_accuracy)
    print("Test Accuracy:", test_accuracy)

    #Classification Report
    Y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = model.classes_)
    disp.plot(cmap = "Blues")
    plt.title("Confusion Matrix")
    plt.show()

    return model, train_accuracy, test_accuracy, cm, report

# Call the function and capture the model, train and test accuracy
model, train_accuracy, test_accuracy, cm, report  = logistic_regression()


# # Decision Tree Model

# In[20]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

def decision_tree(max_depth=3, criterion="gini"):
    dt_model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
    dt_model.fit(X_train, Y_train)

    train_accuracy = dt_model.score(X_train, Y_train)
    test_accuracy = dt_model.score(X_test, Y_test)

    Y_pred = dt_model.predict(X_test)

    report = classification_report(Y_test, Y_pred, output_dict=True)

    return dt_model, train_accuracy, test_accuracy, report

# Call the function and capture the model, train and test accuracy

dt_model, train_accuracy, test_accuracy, report = decision_tree(max_depth=5)



# In[160]:

# # Random Forest Model
#importing libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def random_forest_model():
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf_model.fit(X_train, Y_train)

    # Evaluating the model
    Y_pred = rf_model.predict(X_test)
    train_accuracy = float(rf_model.score(X_train, Y_train))
    test_accuracy = float(rf_model.score(X_test, Y_test))

    # Classification Report
    report = classification_report(Y_test, Y_pred, output_dict=True)  # Use output_dict=True
    return rf_model, train_accuracy, test_accuracy, report


# Call the function and capture the model, train and test accuracy

rf_model,report, train_accuracy, test_accuracy = random_forest_model()



# In[161]:


#Important featres

def plot_feature_importance(model, feature_names, title="Feature Importance"):
    """
    Plot the feature importance from a trained Random Forest model.

    Parameters:
    - model: Trained model with `feature_importances_` attribute (e.g., RandomForestClassifier).
    - feature_names: List or array of feature names corresponding to the input data.
    - title: Title for the plot (default: "Feature Importance").

    Returns:
    - feature_importance: DataFrame of features and their importance scores.
    """
    # Create a DataFrame for feature importance
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_  # Use the model parameter
    }).sort_values(by="Importance", ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return feature_importance


# Assuming rf_model is trained and X.columns contains feature names
feature_importance_df = plot_feature_importance(rf_model, X.columns, title="Random Forest Feature Importance")

# Print the DataFrame
print(feature_importance_df)



