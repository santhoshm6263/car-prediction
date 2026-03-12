import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pickle

def main():
    print("Generating synthetic dataset for car prices...")
    np.random.seed(42)
    n_samples = 200

    # Generate synthetic data
    data = {
        'wheelbase': np.random.uniform(85, 120, n_samples),
        'carlength': np.random.uniform(140, 210, n_samples),
        'carwidth': np.random.uniform(60, 75, n_samples),
        'curbweight': np.random.uniform(1400, 4000, n_samples),
        'enginesize': np.random.uniform(60, 300, n_samples),
        'horsepower': np.random.uniform(48, 288, n_samples)
    }

    # Creating a linear-ish relationship for price with some noise
    price = (
        data['enginesize'] * 120 +
        data['horsepower'] * 60 +
        data['curbweight'] * 1.5 +
        data['carwidth'] * 80 +
        np.random.normal(0, 1500, n_samples)
    )

    df = pd.DataFrame(data)
    df['price'] = price

    # Features and Target
    X = df[['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize', 'horsepower']]
    y = df['price']

    # Train the Decision Tree Regressor
    print("Training the DecisionTreeRegressor model...")
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)

    # Save the model
    model_filename = 'model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    print(f"Model trained successfully and saved as {model_filename}")

if __name__ == "__main__":
    main()
