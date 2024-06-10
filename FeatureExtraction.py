import pandas as pd
import numpy as np
import category_encoders as ce

# Reading the data
df_train = pd.read_csv('./train.csv/train.csv')
df_test = pd.read_csv('./test.csv/test.csv')

# Function to encode categorical columns using count encoding
def target_encode_proportion(df, column_name):
    df_encoded = df.copy()
    count_encoder = ce.CountEncoder()
    new_column_name = f"{column_name}_encoded"
    df_encoded[new_column_name] = count_encoder.fit_transform(df_encoded[column_name])
    return df_encoded

# Encoding categorical columns
df_train = target_encode_proportion(df_train, 'Category')
df_train = target_encode_proportion(df_train, 'PdDistrict')
df_train = target_encode_proportion(df_train, 'Resolution')

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
    df[column_name] = df[column_name].map(day_map)
    return df

df_train = encode_day_of_week(df_train, 'DayOfWeek')

# Extracting time of day from the 'Dates' column
def extract_time_of_day(df):
    df['Dates'] = pd.to_datetime(df['Dates'])
    def get_time_of_day(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 21:
            return 'EveningPeak'
        else:
            return 'Night'
    df['time_of_day'] = df['Dates'].dt.hour.apply(get_time_of_day)
    return df

df_train = extract_time_of_day(df_train)

# Adding temperature level based on month
def add_temp_level(df):
    temp_level_map = {
        1: 'Cold', 2: 'Cold', 3: 'Cold',
        4: 'Mild', 5: 'Mild', 6: 'Warm',
        7: 'Warm', 8: 'Warm', 9: 'Warm',
        10: 'Mild', 11: 'Cold', 12: 'Cold'
    }
    df['Temp_Level'] = df['Dates'].dt.month.map(temp_level_map)
    return df

df_train = add_temp_level(df_train)

# Extracting detailed date features
def extract_date_features(df):
    df['Year'] = df['Dates'].dt.year
    df['Month'] = df['Dates'].dt.month
    df['Day'] = df['Dates'].dt.day
    df['Hour'] = df['Dates'].dt.hour
    df['Minute'] = df['Dates'].dt.minute
    return df.drop('Dates', axis=1)

df_train = extract_date_features(df_train)

# Saving the processed dataframe
df_train.to_csv('/content/drive/MyDrive/ML_project/processed_train.csv', index=False)
