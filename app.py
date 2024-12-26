import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('breast_cancer_model.pkl')

st.title("Breast Cancer Classification")

# Input form for the features
st.write("Enter the values for the features:")
mean_radius = st.number_input("Mean Radius")
mean_texture = st.number_input("Mean Texture")
mean_perimeter = st.number_input("Mean Perimeter")
mean_area = st.number_input("Mean Area")
mean_smoothness = st.number_input("Mean Smoothness")
mean_compactness = st.number_input("Mean Compactness")
mean_concavity = st.number_input("Mean Concavity")
mean_concave_points = st.number_input("Mean Concave Points")
mean_symmetry = st.number_input("Mean Symmetry")
mean_fractal_dimension = st.number_input("Mean Fractal Dimension")

# Predict button
if st.button("Predict"):
    # Create a DataFrame with the input values
    input_data = pd.DataFrame([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
                                mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension]],
                              columns=["mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness",
                                       "mean_compactness", "mean_concavity", "mean_concave_points", "mean_symmetry", "mean_fractal_dimension"])
    # Make prediction
    prediction = model.predict(input_data)
    result = "Malignant" if prediction[0] == 1 else "Benign"
    st.write(f"The predicted class is: {result}")