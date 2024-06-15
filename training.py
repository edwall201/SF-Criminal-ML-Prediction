import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import category_encoders as ce
from category_encoders import CountEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df_train = pd.read_csv('./train.csv/train.csv')
df_test = pd.read_csv('./test.csv/test.csv')

df_train.head()
df_test.head()

df_eda = df_train.copy()
df_eda.head()

def target_encode_proportion(df, column_name):

    # Create a copy of the input DataFrame
    df_encoded = df.copy()
    
    # Initialize the CountEncoder
    count_encoder = ce.CountEncoder()
    
    # Encode the specified column
    new_column_name = f"{column_name}_encoded"
    df_encoded[new_column_name] = count_encoder.fit_transform(df_encoded[column_name])
    
    return df_encoded

df_eda = target_encode_proportion(df_eda, 'Category')
df_eda = target_encode_proportion(df_eda, 'PdDistrict')
df_eda = target_encode_proportion(df_eda, 'Resolution')
df_eda.head()

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
weekday_crimes = df_eda[df_eda['DayOfWeek_encoded'].isin(range(1, 6))]['Dates'].count()
weekend_crimes = df_eda[df_eda['DayOfWeek_encoded'].isin([6, 7])]['Dates'].count()

weekday_count = df_eda[df_eda['DayOfWeek_encoded'].isin(range(1, 6))]['DayOfWeek_encoded'].nunique()
weekend_count = df_eda[df_eda['DayOfWeek_encoded'].isin([6, 7])]['DayOfWeek_encoded'].nunique()

weekday_crime_per_day = weekday_crimes / weekday_count
weekend_crime_per_day = weekend_crimes / weekend_count

weekday_crimes = df_eda[df_eda['DayOfWeek_encoded'].isin(range(1, 6))]['Dates'].count()
weekday_count = df_eda[df_eda['DayOfWeek_encoded'].isin(range(1, 6))]['DayOfWeek_encoded'].nunique()
weekday_crime_per_day = weekday_crimes / weekday_count

weekend_crimes = df_eda[df_eda['DayOfWeek_encoded'].isin([6, 7])]['Dates'].count() 
weekend_count = df_eda[df_eda['DayOfWeek_encoded'].isin([6, 7])]['DayOfWeek_encoded'].nunique()
weekend_crime_per_day = weekend_crimes / weekend_count

crime_counts_by_day = df_eda.groupby('DayOfWeek_encoded')['Dates'].count()
crime_counts_by_day

crime_by_category_weekday = df_eda.groupby(['Category', 'DayOfWeek_encoded']).size().unstack()
crime_by_category_weekday_pct = crime_by_category_weekday.apply(lambda x: x / x.sum(), axis=1)
crime_by_category_weekday_pct


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

# Table shows the pattern of each category
unique_categories = df_eda['Category'].unique()
unique_patterns = df_eda['Weekday>Weekend_Pattern'].unique()

result_dict = {category: df_eda.loc[df_eda['Category'] == category, 'Weekday>Weekend_Pattern'].values[0] 
               for category in unique_categories}

category_by_pattern_df = pd.DataFrame.from_dict(result_dict, orient='index').reset_index()
category_by_pattern_df.columns = ['Category', 'Weekday>Weekend_Pattern']
category_by_pattern_df = category_by_pattern_df.sort_values('Weekday>Weekend_Pattern')

category_by_pattern_df

count = category_by_pattern_df['Weekday>Weekend_Pattern'].value_counts()

def extract_time_of_day(df):
    
    # turn date into datetime
    df['Dates'] = pd.to_datetime(df['Dates'])
    
    # define time segment
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
    
    # encode
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

crime_counts_by_hour = df_eda.groupby(df_eda['Dates'].dt.hour)['Dates'].count()


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
crime_counts_by_temp = df_eda.groupby('Temp_Level')['Dates'].count()

resolutions = df_eda['Resolution'].unique()
resolutions
categories = df_eda['Category'].unique()
categories
district_counts = df_eda['PdDistrict'].value_counts()
total_crimes = len(df_eda)
district_ratios = {district: "{:.5f}".format(count / total_crimes) for district, count in district_counts.items()}

category_district_ratios = {}
all_districts = df_eda['PdDistrict'].unique()

for category in df_eda['Category'].unique():
    category_data = df_eda[df_eda['Category'] == category]
    category_total = len(category_data)
    category_district_ratios[category] = {district: 0.0 for district in all_districts}
    for district in category_data['PdDistrict'].unique():
        district_count = len(category_data[category_data['PdDistrict'] == district])
        ratio = district_count / category_total
        category_district_ratios[category][district] = ratio

all_districts = set()
for district_ratios in category_district_ratios.values():
    all_districts.update(district_ratios.keys())

district_ratio_float = {district: float(category_district_ratios.get(category, {}).get(district, 0.0)) for district in all_districts for category in category_district_ratios}

comparison_results = {}
for category, district_ratios in category_district_ratios.items():
    comparison_results[category] = {}
    for district in all_districts:
        if district in district_ratios:
            difference = district_ratios[district] - district_ratio_float[district]
        else:
            difference = 0.0
        comparison_results[category][district] = difference

import pandas as pd
df_comparison = pd.DataFrame.from_dict(comparison_results, orient='index')
df_comparison

df_train.head()

def convert_dates_to_numeric(df):
    df['Dates_numeric'] = (df['Dates'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    return df

def create_preprocessing_pipeline():
    categorical_features = ['DayOfWeek', 'PdDistrict', 'Resolution', 'Category']
    ordinal_features = ['DayOfWeek']
    numerical_features = ['X', 'Y']
    columns_to_drop = ['Dates', 'Descript', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Category','Address']

    preprocessor = ColumnTransformer(
        transformers=[
            ('count_encode', CountEncoder(), categorical_features),
            ('ord_encode', OrdinalEncoder(), ordinal_features),
            ('date_encode', FunctionTransformer(convert_dates_to_numeric), []),
            ('pattern_recog', FunctionTransformer(add_weekday_weekend_pattern), []),
            ('time_of_day', FunctionTransformer(extract_time_of_day), []),
            ('temp_level', FunctionTransformer(add_temp_level), []),
            ('num_pass', 'passthrough', numerical_features),
            ('drop_cols', 'drop', columns_to_drop)
        ]
    )
    
    preprocessor.set_output(transform='pandas') 
    pipeline = Pipeline([
        ('preprocess', preprocessor)
    ])

    return pipeline, preprocessor

pipeline, preprocessor = create_preprocessing_pipeline()
X = pipeline.fit_transform(df_train)
X.head()

def get_original_feature_names(column_transformer, feature_prefix, feature_name=None):
    if feature_name is None:
        return column_transformer.named_transformers_[feature_prefix].get_feature_names_in()
    else:
        feature_names = column_transformer.named_transformers_[feature_prefix].get_feature_names_in()
        return np.unique(df_train[feature_name].astype(str)).tolist()
    
def create_feature_mapping(original_feature_names):
    return {i: name for i, name in enumerate(original_feature_names)}
category_values = get_original_feature_names(preprocessor, 'count_encode', 'Category')

category_mapping = {i: name for i, name in enumerate(category_values)}
X.head()

y = X['count_encode__Category']
columns_to_drop_encoded = ['count_encode__Resolution', 'count_encode__Category']

X = X.drop(columns_to_drop_encoded, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#=================================================== RandomForestTREE ==================================================================================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


#=================================================== XGB ==================================================================================
y_pred_proba = pd.DataFrame(rf_model.predict_proba(X_test), columns=rf_model.classes_, index=X_test.index)
y_pred_proba.columns = [f"{category_mapping.get(c, f'Unknown_{c}')}" for c in rf_model.classes_]
result_df = y_pred_proba.copy()
result_df.insert(0, 'Id', df_test['Id'])
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, objective='multi:softmax', num_class=39)
xgb_model.fit(X_train, y_train_encoded)
y_pred_encoded = xgb_model.predict(X_test)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")