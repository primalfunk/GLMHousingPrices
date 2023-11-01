from DataGenerator import DataGenerator
from DataPreprocessor import DataPreprocessor
from GLMModel import GLMModel
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Data generation
    data_gen = DataGenerator(num_samples=1000)
    data = data_gen.generate_data()

    # Data preprocessing
    preprocessor = DataPreprocessor()
    preprocessed_data = preprocessor.preprocess(data)

    # Prepare target variable and features
    X = preprocessed_data.iloc[:, :-1]
    y = data['Price']

    # Fit GLM model
    glm_model = GLMModel()
    glm_model.fit(X, y)
    print(glm_model.summary())

    # Split data into training and test sets; 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit GLM model
    glm_model = GLMModel()
    glm_model.fit(X_train, y_train)
    
    # Evaluate the model
    mae, mse, r2 = glm_model.evaluate(X_test, y_test)
    print(f'MAE: {mae}, MSE: {mse}, R-squared: {r2}')

