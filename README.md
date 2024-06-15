# San Francisco Crime Data Analysis

## Introduction

San Francisco, once known for the notorious Alcatraz prison, has transformed into a hub of technological innovation. Despite its progress, the city faces significant social challenges, including rising wealth inequality, severe housing shortages, and pervasive digital devices contributing to urban crime. This analysis explores a dataset of nearly 12 years of crime reports across San Francisco's diverse neighborhoods to predict crime categories based on temporal and spatial data.

## Problem Statement

### Data Wrangling
- Cleanse the dataset to remove inconsistencies or errors.

### Data Exploration
- Understand variables and develop insights for analysis.

### Feature Engineering
- Create additional variables to enhance predictive power or interpretability.

### Data Transformation
- Standardize or transform the data for machine learning algorithms.

### Training/Testing Data Split
- Partition data into training and testing sets to evaluate model performance.

### Model Selection
- Develop a predictive model to estimate crime types based on location and date.

## Data Overview

### Training Set Columns:
- `Dates`: Timestamp of crime occurrence.
- `Category`: Type of crime (target variable).
- `Descript`: Detailed description of the crime.
- `DayOfWeek`: Day of the week when the crime occurred.
- `PdDistrict`: Police district of the crime.
- `Resolution`: Outcome of the crime.
- `Address`: Location of the crime.
- `X`: Longitude coordinate.
- `Y`: Latitude coordinate.

### Test Set Columns:
- `Id`: Unique identifier.
- `Dates`: Timestamp of crime occurrence.
- `DayOfWeek`: Day of the week.
- `PdDistrict`: Police district.
- `Address`: Location of the crime.
- `X`: Longitude coordinate.
- `Y`: Latitude coordinate.

Both sets contain 878,049 entries each.

## Exploratory Data Analysis (EDA) Report

To facilitate analysis and modeling:
- Convert categorical variables to numerical values.
- Use count encoding for categorical variables with many unique values.
- Apply mapping for ordinal variables.

## Hypotheses

### Hypothesis 1: Weekday vs. Weekend Crimes
Weekdays are hypothesized to have a higher incidence of crimes compared to weekends.

#### Results:
- **Average weekday crimes per day**: 126,906.40
- **Average weekend crimes per day**: 121,758.50

This supports the hypothesis with weekdays showing approximately 4% more crimes.

#### Findings:
- **Daily Crime Distribution**: Higher on weekdays.
- **Crime Categories**:
  - 26 categories show higher frequency on weekdays.
  - 13 categories show higher frequency on weekends.

### Hypothesis 2: Crimes and Time of Day
Late night and early morning hours are hypothesized to have higher crime incidence compared to other times of the day.

#### Findings:
- Contrary to the hypothesis, crime peaks during:
  - **Evening (6 PM - 10 PM)**
  - **Noon (12 PM)**

### Hypothesis 3: Proximity to Crime Category Hotspots Influences Crime Likelihood
Certain crime types are hypothesized to be more concentrated in specific neighborhoods or regions.

#### Findings:
- Specific types of crimes are significantly more concentrated in certain police districts:
  - **Larceny/Theft**: Southern district
  - **Drug/Narcotic incidents**: Tenderloin district
  - **Vehicle Theft**: Ingleside and Bayview districts

## Dataset Preprocessing Steps

### 1. Converting Dates to Timestamps
Convert the `Dates` column into a numeric format (timestamps).

### 2. Feature Engineering
Encode categorical features and add new features based on temporal and spatial data.

### 3. Creating a Preprocessing Pipeline
Use `ColumnTransformer` and `Pipeline` from scikit-learn for consistent transformations.

### 4. Mapping Encoded Features to Original Categories
Retain original category names for the `Category` feature for predictions.

### 5. Splitting Data for Training and Testing
Separate the target variable (`Category`) from the feature set and divide the data into training and testing subsets.

## Model Building

### Random Forest Classifier
- Achieved an accuracy of 26.35% on the test set.

### XGBoost Classifier
- Improved the accuracy to 27.40%.

## Conclusion

### Temporal and Spatial Patterns
- Higher crime rates on weekdays and specific districts.
- Certain crime types cluster in particular areas.

### Model Performance
- Both Random Forest and XGBoost models showed room for improvement.
- Further feature engineering and advanced modeling techniques could enhance accuracy.

### Future Work
- Refine feature sets and explore interactions between features.
- Test other machine learning models and ensemble methods.
- Combine temporal data with precise spatial analysis for better predictions.
- Use insights to guide resource allocation and preventive measures.

The analysis of San Francisco crime data provided valuable insights into crime patterns. Initial models highlighted areas for potential improvement through advanced feature engineering and modeling techniques.

## Improved Code
