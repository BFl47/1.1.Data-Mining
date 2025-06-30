# Flight Price Prediction using Data Mining

This project aims to predict domestic U.S. flight ticket prices using data mining techniques applied to a large real-world dataset. The goal is to identify the most accurate predictive model and understand key factors affecting flight pricing.

## Dataset

- **Source:** [Kaggle – Flight Prices (dilwong, 2021)](https://www.kaggle.com/datasets/dilwong/flightprices)
- **Size:** ~82 million records (reduced to ~8 million from LAX)
- **Period:** April – October 2022
- **Target Variable:** `baseFare` (flight ticket base price)

## Project Overview

- **Focus:** Flights departing from Los Angeles International Airport (LAX)
- **Tools:** Python (scikit-learn, pandas, NumPy), PyTorch (for MLP only)
- **Pipeline:**
  - Data cleaning and preprocessing
  - Feature engineering and transformation
  - Model training and evaluation (MAE, RMSE)

## Models Used

- Dummy Regressor (baseline)
- K-Nearest Neighbors (KNN)
- Linear, Lasso, and Ridge Regression
- Decision Tree & Random Forest Regressors
- Multi-Layer Perceptron (MLP)

## Best Performing Model

- **Random Forest Regressor**
  - **MAE:** ~73.9
  - **RMSE:** ~105.4
  - Best trade-off between performance and interpretability

### Feature Importance 

To understand which variables influenced predictions the most, we analyzed feature importance using impurity-based scores from the Random Forest Regressor. The top three features were:

1. **`isCoach`** – Despite being mostly true (i.e., economy class), premium tickets (false values) had a disproportionately large influence on pricing.
2. **`totalDistance`** – Longer flights tend to be more expensive, as expected.
3. **`totalFlightTime`** – Captures hidden pricing factors like layovers or longer flight duration.

Additional influential features:
- **`flightDate`** and **`searchDate`** (as integers) – Their significance highlights how timing (e.g., seasonality, booking advance) affects pricing.

## Contributors

- B. Fiorillo
- C. Lin
- J. Miranda
- J. Park
- T. Nachev
- Y. Kim


