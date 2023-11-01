import streamlit as st
from DataGenerator import DataGenerator
from DataPreprocessor import DataPreprocessor
from GLMModel import GLMModel
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    st.title('GLM for House Price Prediction')

    # Step 1: Data Generation
    st.header('1. Data Generation')
    num_samples = st.slider('Number of Samples', 100, 1000, 500)
    data_gen = DataGenerator(num_samples=num_samples)
    data = data_gen.generate_data()
    st.dataframe(data.head())

    # Step 2: Data Preprocessing
    st.header('2. Data Preprocessing')
    preprocessor = DataPreprocessor()
    preprocessed_data = preprocessor.preprocess(data)
    st.dataframe(preprocessed_data.head())

    # Step 3: Data Splitting
    st.header('3. Data Splitting')
    X = preprocessed_data.iloc[:, :-1]
    y = data['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write(f"Training set size: {len(X_train)}")
    st.write(f"Test set size: {len(X_test)}")

    # Step 4: Model Fitting
    st.header('4. Model Fitting')
    glm_model = GLMModel()
    glm_model.fit(X_train, y_train)
    summary_str = str(glm_model.summary())
    st.text_area("Model Summary:", value=summary_str, height=200)

    # Step 5: Model Evaluation
    st.header('5. Model Evaluation')
    mae, mse, r2 = glm_model.evaluate(X_test, y_test)
    st.write(f"Mean Absolute Error: {mae}")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared: {r2}")

if __name__ == "__main__":
    main()