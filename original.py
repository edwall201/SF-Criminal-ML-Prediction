import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Reading the data
df_train = pd.read_csv('./train.csv/train.csv')
df_test = pd.read_csv('./test.csv/test.csv')

# Basic info about the dataset
print(f'df_train has {len(df_train.columns)} columns')
print(f'df_train has {len(df_train)} entries.')
print(df_train.columns)
df_train.head()

print(f'df_test has {len(df_test.columns)} columns')
print(f'df_test has {len(df_train)} entries.')
print(df_test.columns)
df_test.head()

# Checking the difference in columns between train and test datasets
diff_columns = df_train.columns.symmetric_difference(df_test.columns)
print(f'They have the following different columns: {diff_columns}')

# Copying the dataframe for EDA purposes
df_eda = df_train.copy()
df_eda.head()

# Function to encode categorical columns using count encoding
def target_encode_proportion(df, column_name):
    df_encoded = df.copy()
    count_encoder = ce.CountEncoder()
    new_column_name = f"{column_name}_encoded"
    df_encoded[new_column_name] = count_encoder.fit_transform(df_encoded[column_name])
    return df_encoded

# Encoding categorical columns
df_eda = target_encode_proportion(df_eda, 'Category')
df_eda = target_encode_proportion(df_eda, 'PdDistrict')
df_eda = target_encode_proportion(df_eda, 'Resolution')
df_eda.head()

# Function to encode day of the week
def encode_day_of_week(df, column_name):
    day_map = {
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6,
        'Sunday': 7
    }
    new_column_name = f"{column_name}_encoded"
    df[new_column_name] = df[column_name].map(day_map)
    return df

df_eda = encode_day_of_week(df_eda, 'DayOfWeek')
df_eda.head()

# Analysis on crimes based on day of the week
weekday_crimes = df_eda[df_eda['DayOfWeek_encoded'].isin(range(1, 6))]['Dates'].count()
weekend_crimes = df_eda[df_eda['DayOfWeek_encoded'].isin([6, 7])]['Dates'].count()
weekday_count = df_eda[df_eda['DayOfWeek_encoded'].isin(range(1, 6))]['DayOfWeek_encoded'].nunique()
weekend_count = df_eda[df_eda['DayOfWeek_encoded'].isin([6, 7])]['DayOfWeek_encoded'].nunique()

weekday_crime_per_day = weekday_crimes / weekday_count
weekend_crime_per_day = weekend_crimes / weekend_count

print(f"Average weekday crimes per day: {weekday_crime_per_day:.2f}")
print(f"Average weekend crimes per day: {weekend_crime_per_day:.2f}")

# Plotting average daily crime incidents
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(['Weekday', 'Weekend'], [weekday_crime_per_day, weekend_crime_per_day])
ax.set_title('Average Daily Crime Incidents')
ax.set_xlabel('Day Type')
ax.set_ylabel('Average Crimes Per Day')
plt.show()

# Plotting crime counts by day
crime_counts_by_day = df_eda.groupby('DayOfWeek_encoded')['Dates'].count()
crime_counts_by_day.plot(kind='bar')
plt.title('Daily Crime Incident Counts')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Crimes')
plt.show()

# Crime distribution by category and day of the week
crime_by_category_weekday = df_eda.groupby(['Category', 'DayOfWeek_encoded']).size().unstack()
crime_by_category_weekday_pct = crime_by_category_weekday.apply(lambda x: x / x.sum(), axis=1)
plt.figure(figsize=(10, 20))
sns.heatmap(crime_by_category_weekday_pct, cmap='YlGnBu', annot=True, fmt='.2%')
plt.xlabel('Day of Week')
plt.ylabel('Crime Category')
plt.title('Crime Distribution by Category and Day of Week')
plt.show()

# Adding weekday/weekend pattern
def add_weekday_weekend_pattern(df):
    category_column = 'Category'
    day_column = 'DayOfWeek'
    category_weekday_counts = df.groupby([category_column, day_column])[category_column].count().unstack(fill_value=0)

    def check_pattern(row):
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        weekends = ['Saturday', 'Sunday']
        weekday_avg = row[weekdays].mean()
        weekend_avg = row[weekends].mean()
        if weekday_avg == 0 and weekend_avg == 0:
            return "?"
        elif weekday_avg > weekend_avg:
            return 1
        elif weekday_avg < weekend_avg:
            return 0
        else:
            return "?"

    pattern_column = 'Weekday>Weekend_Pattern'
    patterns = category_weekday_counts.apply(check_pattern, axis=1)
    df[pattern_column] = df[category_column].map(patterns)
    return df

df_eda = add_weekday_weekend_pattern(df_eda)
df_eda.describe()

# Displaying the pattern of each category
unique_categories = df_eda['Category'].unique()
unique_patterns = df_eda['Weekday>Weekend_Pattern'].unique()
result_dict = {category: df_eda.loc[df_eda['Category'] == category, 'Weekday>Weekend_Pattern'].values[0]
               for category in unique_categories}
category_by_pattern_df = pd.DataFrame.from_dict(result_dict, orient='index').reset_index()
category_by_pattern_df.columns = ['Category', 'Weekday>Weekend_Pattern']
category_by_pattern_df = category_by_pattern_df.sort_values('Weekday>Weekend_Pattern')
category_by_pattern_df

count = category_by_pattern_df['Weekday>Weekend_Pattern'].value_counts()
print(f"Matched cases = {count.get(0,0)} , Unmatched cases = {count.get(1,0)}")

# Extracting time of day from the 'Dates' column
def extract_time_of_day(df):
    df['Dates'] = pd.to_datetime(df['Dates'])
    def get_time_of_day(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 13:
            return 'Noon'
        elif 13 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 21:
            return 'EveningPeak'
        else:
            return 'Night'
    df['time_of_day'] = df['Dates'].dt.hour.apply(get_time_of_day)
    time_of_day_mapping = {
        'Morning': 1,
        'Noon': 2,
        'Afternoon': 3,
        'EveningPeak': 4,
        'Night': 5
    }
    df['time_of_day'] = df['time_of_day'].map(time_of_day_mapping)
    return df

df_eda = extract_time_of_day(df_eda)
df_eda.head()

# Plotting crimes by hour of the day
crime_counts_by_hour = df_eda.groupby(df_eda['Dates'].dt.hour)['Dates'].count()
crime_counts_by_hour.plot(kind='line')
plt.title('Crimes by Hour of the Day')
plt.xlabel('Hour')
plt.ylabel('Number of Crimes')
plt.show()

# Adding temperature level based on month
def add_temp_level(df):
    temp_level_map = {
        1: 1, 2: 1, 3: 1,
        4: 2, 5: 2, 6: 3,
        7: 3, 8: 3, 9: 3,
        10: 2, 11: 1, 12: 1
    }
    df['Temp_Level'] = df['Dates'].dt.month.map(temp_level_map)
    return df

df_eda = add_temp_level(df_eda)
df_eda.describe()

# Plotting crimes by temperature level
crime_counts_by_temp = df_eda.groupby('Temp_Level')['Dates'].count()
crime_counts_by_temp.plot(kind='bar')
plt.title('Crimes by Temperature Level')
plt.xlabel('Temperature Level')
plt.ylabel('Number of Crimes')
plt.show()

# Ratio analysis for districts and categories
district_counts = df_eda['PdDistrict'].value_counts()
total_crimes = len(df_eda)
district_ratios = {district: "{:.5f}".format(count / total_crimes) for district, count in district_counts.items()}
district_ratios_df = pd.DataFrame(list(district_ratios.items()), columns=['District', 'Ratio'])
district_ratios_df

category_counts = df_eda['Category'].value_counts()
category_ratios = {category: "{:.5f}".format(count / total_crimes) for category, count in category_counts.items()}
category_ratios_df = pd.DataFrame(list(category_ratios.items()), columns=['Category', 'Ratio'])
category_ratios_df

category_district_ratios = {}
for category in df_eda['Category'].unique():
    category_data = df_eda[df_eda['Category'] == category]
    category_total = len(category_data)
    all_districts = df_eda['PdDistrict'].unique()
    category_district_ratios[category] = {district: 0 for district in all_districts}
    for district in all_districts:
        district_total = len(category_data[category_data['PdDistrict'] == district])
        ratio = district_total / category_total if category_total > 0 else 0.0
        category_district_ratios[category][district] = ratio

# Calculating weekday/weekend ratios
def calculate_weekday_weekend_ratios(df):
    ratios = {}
    total_weekday = df[df['DayOfWeek_encoded'].isin(range(1, 6))].shape[0]
    total_weekend = df[df['DayOfWeek_encoded'].isin([6, 7])].shape[0]
    for category in df['Category'].unique():
        category_data = df[df['Category'] == category]
        weekday_count = category_data[category_data['DayOfWeek_encoded'].isin(range(1, 6))].shape[0]
        weekend_count = category_data[category_data['DayOfWeek_encoded'].isin([6, 7])].shape[0]
        weekday_ratio = weekday_count / total_weekday if total_weekday > 0 else 0
        weekend_ratio = weekend_count / total_weekend if total_weekend > 0 else 0
        ratios[category] = {'weekday_ratio': weekday_ratio, 'weekend_ratio': weekend_ratio}
    return ratios

weekday_weekend_ratios = calculate_weekday_weekend_ratios(df_eda)
weekday_weekend_ratios

# Plotting crimes per district
plt.figure(figsize=(12, 8))
district_counts.plot(kind='bar')
plt.title('Crimes per District')
plt.xlabel('District')
plt.ylabel('Number of Crimes')
plt.show()

# Data preprocessing for the model
df_eda.drop(['Resolution', 'Address'], axis=1, inplace=True)

X = df_eda.drop('Category', axis=1)
y = df_eda['Category']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extracting features from the 'Dates' column before adding it to the ColumnTransformer
def extract_date_features(df):
    df['Dates'] = pd.to_datetime(df['Dates'])
    df['Year'] = df['Dates'].dt.year
    df['Month'] = df['Dates'].dt.month
    df['Day'] = df['Dates'].dt.day
    df['Hour'] = df['Dates'].dt.hour
    df['Minute'] = df['Dates'].dt.minute
    df['Second'] = df['Dates'].dt.second
    return df.drop('Dates', axis=1)

X_train = extract_date_features(X_train)
X_test = extract_date_features(X_test)

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('dow', OrdinalEncoder(), ['DayOfWeek']),
    ('pddistrict', OrdinalEncoder(), ['PdDistrict']),
    ('year', 'passthrough', ['Year']),
    ('month', 'passthrough', ['Month']),
    ('day', 'passthrough', ['Day']),
    ('hour', 'passthrough', ['Hour']),
    ('minute', 'passthrough', ['Minute']),
    ('second', 'passthrough', ['Second']),
    ('x', 'passthrough', ['X']),
    ('y', 'passthrough', ['Y'])
])

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])

# Fitting the pipeline
pipeline.fit(X_train, y_train)

# Making predictions
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# Calculating accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f'Training Accuracy: {train_accuracy}')
print(f'Test Accuracy: {test_accuracy}')

# ================================== TBD ===========================================
# Additional Random Forest feature importance analysis
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plotting feature importances
plt.figure(figsize=(12, 8))
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.tight_layout()
plt.show()

# Using XGBoost for the final model
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {
    'max_depth': 6,
    'eta': 0.1,
    'objective': 'multi:softmax',
    'num_class': len(np.unique(y_train))
}
epochs = 100

# Training XGBoost model
bst = xgb.train(param, dtrain, epochs)

# Making predictions with XGBoost
preds_train = bst.predict(dtrain)
preds_test = bst.predict(dtest)

# Calculating accuracy
train_accuracy_xgb = accuracy_score(y_train, preds_train)
test_accuracy_xgb = accuracy_score(y_test, preds_test)

print(f'XGBoost Training Accuracy: {train_accuracy_xgb}')
print(f'XGBoost Test Accuracy: {test_accuracy_xgb}')