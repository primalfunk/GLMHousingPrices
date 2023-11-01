from sklearn import metrics
import statsmodels.api as sm

class GLMModel:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        X = sm.add_constant(X)  # Adds a constant term to the predictor
        self.model = sm.GLM(y, X, family=sm.families.Gaussian()).fit()

    def summary(self):
        if self.model:
            return self.model.summary()
        else:
            return "The model is not fitted yet"

    def predict(self, X):
        X = sm.add_constant(X)
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)

        return mae, mse, r2


