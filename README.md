# Generalized Linear Model Housing Prices Prediction
Using a Generalized Linear Model to spoof housing prices and generate predictions

A Generalized Linear Model takes simple linear regression and generalizes it using a link function. Here's the overview of the app process, a kind of flowchart:

Data Preparation:
-generated synthetic data to simulate housing prices based on features like square footage, number of bedrooms, and location using numpy.

Data Exploration and Visualization:
-using Seaborn, created various plots to understand the relationships between different variables. This helped us get a visual sense of the data and its structure.

Feature Selection:
- chose the variables that would be part of our model. This is crucial for making accurate predictions.

Model Building:
-used GLM to build a model that describes the relationship between the chosen features and the house price. This is where the "fitting" of the model happens, essentially teaching it to make predictions.

Evaluation:
-once the model was trained, used statistical methods to evaluate its performance. This tells us how well the model is likely to perform on unseen data.

Prediction:
-used the trained model to make price predictions on new, unseen data.

In order to look at the streamlit visualization in your local browser, run in the terminal: "streamlit run app.py"; this cool visualization tool allows you to fiddle around with some of the model settings.
