# DataGenerator.py

import pandas as pd
import numpy as np

class DataGenerator:
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def generate_data(self):
        square_footage = np.random.normal(1500, 500, self.num_samples).astype(int)
        num_bedrooms = np.random.randint(1, 5, self.num_samples)
        location = np.random.choice(['urban', 'suburban', 'rural'], self.num_samples)
        age = np.random.randint(0, 100, self.num_samples)

        # Generating price with some random noise
        price = square_footage * 200 + num_bedrooms * 10000 + np.random.normal(0, 10000, self.num_samples)

        # Creating a DataFrame
        df = pd.DataFrame({
            'SquareFootage': square_footage,
            'NumBedrooms': num_bedrooms,
            'Location': location,
            'Age': age,
            'Price': price
        })

        return df