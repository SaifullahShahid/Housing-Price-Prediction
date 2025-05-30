import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import streamlit as st

# Loading dataset
housing_raw = fetch_california_housing(as_frame=True)
housing = housing_raw.frame  # pandas DataFrame with data + target

# Renaming target column for clarity
housing = housing.rename(columns={"MedHouseVal": "median_house_value"})

# 2. Exploratory Data Analysis

def plot_histograms(data):
    data.hist(bins=50, figsize=(20,15))
    plt.tight_layout()
    plt.show()

def plot_scatter_matrix(data, attributes):
    pd.plotting.scatter_matrix(data[attributes], figsize=(12,8))
    plt.show()



# Preprocessing

# Custom transformer to add combined attributes
rooms_ix, households_ix = 3, 5  # indices of AveRooms, AveOccup in dataset columns
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        return np.c_[X, rooms_per_household]

# Split dataset
X = housing.drop("median_house_value", axis=1)
y = housing["median_house_value"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline for numerical features
num_attribs = list(X.columns)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

# Applying pipeline to training and test data
X_train_prepared = num_pipeline.fit_transform(X_train)
X_test_prepared = num_pipeline.transform(X_test)

#Train and compare models

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
}

results = {}

for name, model in models.items():
    model.fit(X_train_prepared, y_train)
    preds_train = model.predict(X_train_prepared)
    preds_test = model.predict(X_test_prepared)
    
    rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
    
    mae_train = mean_absolute_error(y_train, preds_train)
    mae_test = mean_absolute_error(y_test, preds_test)
    
    results[name] = {
        "model": model,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "mae_train": mae_train,
        "mae_test": mae_test,
    }

# Printing model performance
for name, metrics in results.items():
    print(f"Model: {name}")
    print(f"  Training RMSE: {metrics['rmse_train']:.2f}")
    print(f"  Test RMSE: {metrics['rmse_test']:.2f}")
    print(f"  Training MAE: {metrics['mae_train']:.2f}")
    print(f"  Test MAE: {metrics['mae_test']:.2f}\n")

# Selecting best model by test RMSE
best_model_name = min(results, key=lambda x: results[x]["rmse_test"])
best_model = results[best_model_name]["model"]
print(f"Best model based on Test RMSE: {best_model_name}")

def display_model_comparison():
    # Data dictionary of model metrics
    model_data = {
        "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
        "Training RMSE": [0.68, 0.00, 0.19],
        "Test RMSE": [0.69, 0.73, 0.51],
        "Training MAE": [0.49, 0.00, 0.12],
        "Test MAE": [0.50, 0.47, 0.33],
    }

    # Create a DataFrame
    df = pd.DataFrame(model_data)

    # Display title
    st.write("### Model Performance Comparison")

    # Show the table in Streamlit
    st.table(df)


# Streamlit App to Deploy Best Model

def run_streamlit_app():
    import streamlit as st

    st.title("California Housing Price Prediction")

    st.write("Using the best model:", best_model_name)

    # Simple sliders for each feature with min/max values taken from your X_train data
    MedInc = st.slider("Median Income", float(X["MedInc"].min()), float(X["MedInc"].max()), float(X["MedInc"].mean()))
    HouseAge = st.slider("House Age", float(X["HouseAge"].min()), float(X["HouseAge"].max()), float(X["HouseAge"].mean()))
    AveRooms = st.slider("Average Rooms", float(X["AveRooms"].min()), float(X["AveRooms"].max()), float(X["AveRooms"].mean()))
    AveBedrms = st.slider("Average Bedrooms", float(X["AveBedrms"].min()), float(X["AveBedrms"].max()), float(X["AveBedrms"].mean()))
    Population = st.slider("Population", float(X["Population"].min()), float(X["Population"].max()), float(X["Population"].mean()))
    AveOccup = st.slider("Average Occupancy", float(X["AveOccup"].min()), float(X["AveOccup"].max()), float(X["AveOccup"].mean()))
    Latitude = st.slider("Latitude", float(X["Latitude"].min()), float(X["Latitude"].max()), float(X["Latitude"].mean()))
    Longitude = st.slider("Longitude", float(X["Longitude"].min()), float(X["Longitude"].max()), float(X["Longitude"].mean()))

    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

    # Prepare input data
    input_prepared = num_pipeline.transform(input_data)

    # Predict
    prediction = best_model.predict(input_prepared)[0]

    st.subheader(f"Predicted Median House Value: ${prediction * 100000:.2f}")
    
    display_model_comparison()



if __name__ == "__main__":
    run_streamlit_app()
