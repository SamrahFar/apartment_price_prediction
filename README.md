# Apartment Price Prediction - Machine Learning with Scikit-learn

## 📑 Content Index

- [Overview](#overview)
- [Dataset](#dataset)
- [Steps](#steps)
  - [1. Standard Imports](#1-standard-imports)
  - [2. Data Loading & Exploration](#2-data-loading--exploration)
  - [3. Data Visualization](#3-data-visualization)
  - [4. Correlation Analysis](#4-correlation-analysis)
  - [5. Preprocessing Pipelines](#5-preprocessing-pipelines)
  - [6. Column Transformer](#6-column-transformer)
  - [7. Modeling](#7-modeling)
- [Usage](#usage)
- [Requirements](#requirements)


## Overview

This project demonstrates a basic multiple linear regression workflow in Python using Scikit-learn, focused on predicting apartment rental prices in the USA. It walks through the essential steps of the machine learning process, including data exploration, cleaning, preprocessing with pipelines, model selection, and evaluation.

## Dataset

- **Source**: The dataset contains 10,000 instances of apartment listings, each with the following columns:
  - `id`: Unique identifier (dropped in preprocessing)
  - `latitude`, `longitude`: Geolocation
  - `bathrooms`, `bedrooms`: Apartment specs
  - `fee`: Y/N (dropped in preprocessing)
  - `has_photo`: Y/N/Thumbnail
  - `pets_allowed`: Pets info (may be missing)
  - `square_feet`: Area in sq ft
  - `price`: Target variable

## Steps

### 1. Standard Imports
- Uses pandas, numpy, matplotlib, and scikit-learn.

### 2. Data Loading & Exploration
- Loads CSV data into a DataFrame.
- Drops irrelevant or uniform columns.
- Splits into features and target, then into train/test sets.

### 3. Data Visualization
- Scatterplots for location and selected numeric attributes.
- Value counts for categorical columns.

### 4. Correlation Analysis
- Examines relationships between numeric features and price.

### 5. Preprocessing Pipelines
- Numeric: Impute missing values (median), scale features.
- Categorical: Impute missing as "No_Pets", one-hot encode.

### 6. Column Transformer
- Combines numeric and categorical pipelines for feature engineering.

### 7. Modeling
- Fits a linear regression model to predict prices.
- Prepares test data for final evaluation.

## Usage

1. Place `apartments_for_rent.csv` in the expected directory.
2. Run the notebook from start to finish, ensuring all code blocks execute without error.
3. The main notebook file is named `apartments_assignment.ipynb`.

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- scikit-learn


