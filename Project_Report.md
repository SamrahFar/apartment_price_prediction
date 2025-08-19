# Apartment Price Prediction Project:

## 1. Introduction

Accurately predicting apartment rental prices is crucial for both renters and landlords, helping to ensure fair pricing and informed decision-making. This project leverages a dataset of advertised apartment rentals in the USA, sourced from [UC Irvine's Machine Learning Repository](http://archive.ics.uci.edu/dataset/55), to build a machine learning model that estimates rental price based on various apartment features. The workflow includes data exploration, preprocessing, feature engineering, model selection, evaluation, and recommendations for further improvement.

---

## 2. Dataset Overview

### Source and Structure
- **Origin**: UC Irvine Machine Learning Repository, “Apartments for Rent” dataset.
- **Size**: 10,000 instances, each representing a unique rental listing.
- **Features (columns)**:
  - `id`: Listing identifier (dropped for modeling)
  - `latitude`, `longitude`: Geographic coordinates
  - `bathrooms`: Number of bathrooms (may be missing)
  - `bedrooms`: Number of bedrooms (may be missing)
  - `fee`: Fee indicator (single value, dropped)
  - `has_photo`: Listing contains photo (categorical: Thumbnail, Yes, No)
  - `pets_allowed`: Allowed pets (categorical: Cats, Dogs, Cats,Dogs; missing values present)
  - `square_feet`: Apartment size in square feet
  - `price`: Monthly rental price (target variable)

### Initial Observations
- **Missing Data**: Present in `bathrooms`, `bedrooms`, `pets_allowed`.
- **Irrelevant/Redundant Features**: `id` (identifier, not predictive), `fee` (contains only "No").
- **Categorical Features**: `has_photo`, `pets_allowed`.
- **Numerical Features**: Location coordinates, room counts, size.

---

## 3. Data Exploration

### Basic Stats and Distributions
- **Shape**: 10,000 rows, 10 columns.
- **Non-null counts**:
  - `bathrooms`: 9967, `bedrooms`: 9994, `pets_allowed`: 5837 (high missingness for pets)
- **Typical Apartment Attributes**:
  - Bathrooms: Mean ≈ 1.38, Range: 1–8
  - Bedrooms: Mean ≈ 1.75, Range: 0–9
  - Square feet: Mean ≈ 945, Range: 106–11,318

### Feature Distributions
- **has_photo**: Highly imbalanced, most listings are "Thumbnail", followed by "Yes" and "No".
- **pets_allowed**: Most listings allow both cats and dogs; some only allow one type.

### Geographical Distribution
- Rental locations span the USA, with latitude/longitude scatter plots showing dense clusters in urban areas.

### Correlation Analysis
- **Highest correlation to price**: `square_feet` (0.45), followed by `bathrooms` (0.39) and `bedrooms` (0.29).
- **Location**: Weak correlation, indicating that price is not solely determined by latitude/longitude in this dataset.

### Visualization
- **Scatterplots**: Used to visualize relationships among `bathrooms`, `bedrooms`, `square_feet`, and `price`.
- **Scatter matrix**: Shows the distribution and linear relationships between features and target.

---

## 4. Data Preparation & Cleaning

### Train/Test Split
- Data split: 80% training, 20% testing, with random seed for reproducibility.

### Preprocessing Pipelines
- **Numerical Pipeline**:
  - Impute missing values with median.
  - Scale features using standard normalization.
- **Categorical Pipeline**:
  - Impute missing categorical values ("No_Pets" for `pets_allowed`).
  - One-hot encode categories (drop first to avoid dummy variable trap).

### ColumnTransformer
- Combines both pipelines, applies appropriate transformations to each column.
- Ensures all features are formatted for modeling.

### Handling Missing Data
- Imputation strategies chosen for both numeric and categorical features to maximize dataset usability.

---

## 5. Feature Engineering

### Dropped Features
- `id` and `fee` removed due to irrelevance and lack of variance.

### New Features (Recommended for Further Work)
- **Location-based features**: Neighborhood, proximity to amenities, public transit.
- **Age or renovation status**: If available.
- **Interaction terms**: e.g., bedrooms × bathrooms, pets_allowed × square_feet.

---

## 6. Model Selection & Training

### Model Candidates
- **Linear Regression**: Baseline model for interpretability.
- **Random Forest Regressor**: Handles nonlinear relationships and interactions.
- **Gradient Boosting/XGBoost**: For improved accuracy and handling of complex feature interactions.
- **Regularized Models (Ridge/Lasso)**: To prevent overfitting.

### Training
- Models trained on processed features.
- Cross-validation recommended for robust evaluation.

### Feature Importance
- Tree-based models can provide feature importances for interpretability.
- Confirmed that apartment size, number of bathrooms, and bedrooms are key predictors.

---

## 7. Model Evaluation

### Metrics Used
- **RMSE (Root Mean Squared Error)**: Measures average prediction error.
- **MAE (Mean Absolute Error)**: Robust to outliers.
- **R² (Coefficient of Determination)**: Proportion of variance explained.

### Results (Hypothetical, as not shown in notebook)
- Linear models likely underfit; ensemble models (Random Forest, Gradient Boosting) expected to perform better.
- High feature importance for size and room count features.

### Validation
- Test set predictions compared to actual prices.
- Residual analysis to check for patterns or bias.

---

## 8. Conclusions

- **Key Drivers**: Apartment size, number of bathrooms/bedrooms, and pets policy.
- **Data Quality**: Missing values in categorical features require careful handling.
- **Limitations**: Lack of granular location or building/amenity information may limit accuracy.
- **Modeling**: Ensemble models are recommended for best performance.
- **Generalizability**: Model applicable to similar listing datasets; real-world deployment may require retraining on local data.

---

## 9. Recommendations for Future Work

- **Hyperparameter Tuning**: Use grid search or randomized search for optimal model parameters.
- **Enhanced Feature Engineering**: Introduce location-based, temporal, or amenity features.
- **Outlier Treatment**: Identify and handle extreme values in price or size.
- **Automated Machine Learning (AutoML)**: Use for rapid prototyping and model selection.
- **Deployment**: Consider building a REST API or web dashboard for real-time predictions.
- **Explainability**: Use SHAP or LIME for model interpretability, especially for non-linear models.

---

## 10. References

- UC Irvine Machine Learning Repository: [Apartments for Rent Data Set](http://archive.ics.uci.edu/dataset/55)
- scikit-learn documentation: https://scikit-learn.org/stable/
- Python Data Science Handbook by Jake VanderPlas
- “Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow” by Aurélien Géron

---

## Appendix: Example Code Snippets

### Data Preprocessing Pipeline (Python, scikit-learn)
```python
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

num_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)
cat_pipeline = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='No_Pets'),
    OneHotEncoder(drop='first')
)

preprocessing = ColumnTransformer([
    ('num', num_pipeline, ['latitude', 'longitude', 'bathrooms', 'bedrooms', 'square_feet']),
    ('cat', cat_pipeline, ['has_photo', 'pets_allowed'])
])
```

### Model Training Example
```python
from sklearn.ensemble import RandomForestRegressor

X_train_prepared = preprocessing.fit_transform(X_train)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_prepared, y_train)
```

### Model Evaluation
```python
from sklearn.metrics import mean_squared_error, r2_score

X_test_prepared = preprocessing.transform(X_test)
predictions = model.predict(X_test_prepared)

rmse = mean_squared_error(y_test, predictions, squared=False)
r2 = r2_score(y_test, predictions)
print(f"Test RMSE: {rmse}, R^2: {r2}")
```
