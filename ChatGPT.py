import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the data
data = pd.read_csv("Diabetes.csv")

# Preprocess the data
data = data.fillna(data.mean())

# Split the data into training and testing sets
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

try:
    # Create the logistic regression model
    model = LogisticRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Create the app interface
    st.sidebar.header("Logistic Regression App")
    st.sidebar.markdown("Select the features to include in the model:")
    feature_names = X.columns.tolist()
    selected_features = st.sidebar.multiselect("Features", feature_names)

    st.subheader("Logistic Regression Results")

    # Predict on the test data using the selected features
    X_test_selected = X_test[selected_features]
    y_pred = model.predict(X_test_selected)

    # Display the predicted values and the model performance
    st.write("Predicted values:", y_pred)
    st.write("Model performance:")
    st.write(model.score(X_test_selected, y_test))
except Exception as e:
        print(e)
        st.write("Please upload file to the application.")
