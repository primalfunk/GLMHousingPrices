import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    def __init__(self):
        # Column transformer setup
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['SquareFootage', 'NumBedrooms', 'Age']),
                ('cat', OneHotEncoder(), ['Location'])
            ])

    def preprocess(self, data):
        return pd.DataFrame(self.preprocessor.fit_transform(data))