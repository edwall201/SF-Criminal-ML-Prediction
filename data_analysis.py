import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


df_train = pd.read_csv('./train.csv/train.csv')
df_test = pd.read_csv('./test.csv/test.csv')
print(f'df_train has {len(df_train.columns)} columns')
print(f'df_train has {len(df_train)} entries.')
print(df_train.columns)
df_train.head()

print(f'df_test has {len(df_test.columns)} columns')
print(f'df_test has {len(df_train)} entries.')
print(df_test.columns)
df_test.head()

diff_columns = df_train.columns.symmetric_difference(df_test.columns)
print(f'they have following different columns: {diff_columns}')
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

print(f"Average weekday crimes per day: {weekday_crime_per_day:.2f}")
print(f"Average weekend crimes per day: {weekend_crime_per_day:.2f}")

weekday_crimes = df_eda[df_eda['DayOfWeek_encoded'].isin(range(1, 6))]['Dates'].count()
weekday_count = df_eda[df_eda['DayOfWeek_encoded'].isin(range(1, 6))]['DayOfWeek_encoded'].nunique()
weekday_crime_per_day = weekday_crimes / weekday_count

weekend_crimes = df_eda[df_eda['DayOfWeek_encoded'].isin([6, 7])]['Dates'].count() 
weekend_count = df_eda[df_eda['DayOfWeek_encoded'].isin([6, 7])]['DayOfWeek_encoded'].nunique()
weekend_crime_per_day = weekend_crimes / weekend_count

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(['Weekday', 'Weekend'], [weekday_crime_per_day, weekend_crime_per_day])
ax.set_title('Average Daily Crime Incidents')
ax.set_xlabel('Day Type')
ax.set_ylabel('Average Crimes Per Day')
plt.show()

crime_counts_by_day = df_eda.groupby('DayOfWeek_encoded')['Dates'].count()
crime_counts_by_day

import matplotlib.pyplot as plt

crime_counts_by_day.plot(kind='bar')
plt.title('Daily Crime Incident Counts')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Crimes')
plt.show()

crime_by_category_weekday = df_eda.groupby(['Category', 'DayOfWeek_encoded']).size().unstack()
crime_by_category_weekday_pct = crime_by_category_weekday.apply(lambda x: x / x.sum(), axis=1)
crime_by_category_weekday_pct

plt.figure(figsize=(10, 20))
sns.heatmap(crime_by_category_weekday_pct, cmap='YlGnBu', annot=True, fmt='.2%')
plt.xlabel('Day of Week')
plt.ylabel('Crime Category')
plt.title('Crime Distribution by Category and Day of Week')
plt.show()

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

print(f"Matched cases = {count.get(0,0)} , Unmatched cases = {count.get(1,0)}")

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
print(crime_counts_by_hour)
crime_counts_by_hour.plot(kind='line')
plt.title('Crimes by Hour of the Day')
plt.xlabel('Hour')
plt.ylabel('Number of Crimes')
plt.show()

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
print(crime_counts_by_temp)

crime_counts_by_temp.plot(kind='bar')
plt.title('Crimes by Temperature Level')
plt.xlabel('Temperature Level')
plt.ylabel('Number of Crimes')
plt.show()

resolutions = df_eda['Resolution'].unique()
resolutions
categories = df_eda['Category'].unique()
categories
district_counts = df_eda['PdDistrict'].value_counts()
total_crimes = len(df_eda)
district_ratios = {district: "{:.5f}".format(count / total_crimes) for district, count in district_counts.items()}
district_ratios

category_district_ratios = {}

# 取得所有的警區
all_districts = df_eda['PdDistrict'].unique()

# 遍歷每個犯罪類別
for category in df_eda['Category'].unique():
    # 選擇該犯罪類別的資料子集
    category_data = df_eda[df_eda['Category'] == category]
    
    # 計算該犯罪類別的總數
    category_total = len(category_data)
    
    # 初始化該類別的字典
    category_district_ratios[category] = {district: 0.0 for district in all_districts}
    
    # 遍歷每個警區
    for district in category_data['PdDistrict'].unique():
        # 計算該警區的犯罪數量
        district_count = len(category_data[category_data['PdDistrict'] == district])
        
        # 計算該警區犯罪數量佔該類別總數的比例
        ratio = district_count / category_total
        
        # 將該比例寫入字典
        category_district_ratios[category][district] = ratio

# 印出結果
for category, district_ratios in category_district_ratios.items():
    print(f"Category: {category}")
    for district, ratio in district_ratios.items():
        print(f" {district}: {ratio:.5f}")

# 建立一個包含所有警區的列表
all_districts = set()
for district_ratios in category_district_ratios.values():
    all_districts.update(district_ratios.keys())

# 將所有比例轉換為浮點數
district_ratio_float = {district: float(category_district_ratios.get(category, {}).get(district, 0.0)) for district in all_districts for category in category_district_ratios}

# 建立一個空字典用於存儲結果
comparison_results = {}

# 遍歷每個犯罪類別
for category, district_ratios in category_district_ratios.items():
    # 初始化該類別的字典
    comparison_results[category] = {}
    
    # 遍歷每個警區
    for district in all_districts:
        # 計算該類別在該區的比例與該區全體比例的差異
        if district in district_ratios:
            difference = district_ratios[district] - district_ratio_float[district]
        else:
            difference = 0.0
        
        # 將差異值加入字典中
        comparison_results[category][district] = difference

import pandas as pd

df_comparison = pd.DataFrame.from_dict(comparison_results, orient='index')
df_comparison

import seaborn as sns
import matplotlib.pyplot as plt

# 定義以綠色為主的色彩映射
cmap_green = sns.diverging_palette(150, 10, as_cmap=True)

# 繪製熱力圖，調整 figsize 參數使得寬度更寬一些
plt.figure(figsize=(13, 16))
sns.heatmap(df_comparison, cmap=cmap_green, center=0, annot=False)
plt.title('Heatmap of Difference between Category and District Ratio')
plt.xlabel('PdDistrict')
plt.ylabel('Category')
plt.show()

