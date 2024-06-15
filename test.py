import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load and prepare your data
df_train = pd.read_csv('./train.csv/train.csv')
df_test = pd.read_csv('./test.csv/test.csv')

# Feature Engineering
def create_features(df):
    df['Year'] = pd.to_datetime(df['Dates']).dt.year
    df['Month'] = pd.to_datetime(df['Dates']).dt.month
    df['Day'] = pd.to_datetime(df['Dates']).dt.day
    df['Hour'] = pd.to_datetime(df['Dates']).dt.hour
    df['DayOfWeek'] = pd.to_datetime(df['Dates']).dt.dayofweek
    return df.drop(columns=['Dates'])

df_train = create_features(df_train)
df_test = create_features(df_test)

# Select features and target
X = df_train.drop(columns=['Category', 'Resolution', 'Address'])  # Assuming 'Resolution' and 'Address' are dropped
y = df_train['Category']

# Splitting the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Preprocessing
categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Define the model
model = XGBClassifier(n_estimators=100, learning_rate=0.05, n_jobs=4)

# Create and evaluate the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

pipeline.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = pipeline.predict(X_val)
print(f'Validation Accuracy: {accuracy_score(y_val, y_pred)}')

# Optionally, perform a grid search to find better hyperparameters
param_grid = {
    'model__n_estimators': [100, 300, 500],
    'model__learning_rate': [0.01, 0.05, 0.1],
}

search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
search.fit(X_train, y_train)
print(f'Best score: {search.best_score_}')
print(f'Best parameters: {search.best_params_}')

y_pred_test = search.predict(X_val)
print(f'Improved Validation Accuracy: {accuracy_score(y_val, y_pred_test)}')

'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

# Load the datasets
train_df = pd.read_csv('./train.csv/train.csv')
test_df = pd.read_csv('./test.csv/test.csv')

# Feature Engineering function
def create_features(df):
    df['Year'] = pd.to_datetime(df['Dates']).dt.year
    df['Month'] = pd.to_datetime(df['Dates']).dt.month
    df['Day'] = pd.to_datetime(df['Dates']).dt.day
    df['Hour'] = pd.to_datetime(df['Dates']).dt.hour
    df['DayOfWeek'] = pd.to_datetime(df['Dates']).dt.dayofweek
    return df.drop(columns=['Dates'])

train_df = create_features(train_df)
test_df = create_features(test_df)

# Encode the target variable for training
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['Category'])

# Drop unnecessary columns
X_train = train_df.drop(columns=['Category', 'Descript', 'Resolution', 'Address'])
X_test = test_df.drop(columns=['Id', 'Address'])

# Ensure the test set has the same columns as the training set after dropping
X_test = X_test[X_train.columns]

# Data Preprocessing
categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create and evaluate the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Train the model
pipeline.fit(X_train, y_train)

# Predict on training data to calculate training accuracy
y_pred_train = pipeline.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f'Training Accuracy: {train_accuracy}')

# Predict on validation data (split from training data) for validation accuracy
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
pipeline.fit(X_train_split, y_train_split)
y_pred_val = pipeline.predict(X_val_split)
val_accuracy = accuracy_score(y_val_split, y_pred_val)
print(f'Validation Accuracy: {val_accuracy}')

# Predict probabilities on the test set
y_prob_test = pipeline.predict_proba(X_test)

# Convert probabilities to DataFrame with the required format
submission_df = pd.DataFrame(y_prob_test, columns=label_encoder.classes_)
submission_df.insert(0, 'Id', test_df['Id'])

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print('Submission file has been created successfully!')
'''
'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the datasets
train_df = pd.read_csv('./train.csv/train.csv')
test_df = pd.read_csv('./test.csv/test.csv')

# Feature Engineering function
def create_features(df):
    df['Year'] = pd.to_datetime(df['Dates']).dt.year
    df['Month'] = pd.to_datetime(df['Dates']).dt.month
    df['Day'] = pd.to_datetime(df['Dates']).dt.day
    df['Hour'] = pd.to_datetime(df['Dates']).dt.hour
    df['DayOfWeek'] = pd.to_datetime(df['Dates']).dt.dayofweek
    return df.drop(columns=['Dates'])

train_df = create_features(train_df)
test_df = create_features(test_df)

# Encode the target variable for training
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['Category'])

# Drop unnecessary columns
X_train = train_df.drop(columns=['Category', 'Descript', 'Resolution', 'Address'])
X_test = test_df.drop(columns=['Id', 'Address'])

# Ensure the test set has the same columns as the training set after dropping
X_test = X_test[X_train.columns]

# Data Preprocessing
categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Define a simpler model
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)

# Create and evaluate the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Use a smaller sample of data for faster fitting during testing (you can remove this for full fitting)
sample_size = 50000  # Adjust this number to fit a smaller sample
if len(X_train) > sample_size:
    X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, train_size=sample_size, random_state=42)
else:
    X_train_sample, y_train_sample = X_train, y_train

pipeline.fit(X_train_sample, y_train_sample)

# Predict on training sample data to calculate training accuracy
y_pred_train = pipeline.predict(X_train_sample)
train_accuracy = accuracy_score(y_train_sample, y_pred_train)
print(f'Training Accuracy: {train_accuracy}')

# Use a validation split from training data for validation accuracy
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
pipeline.fit(X_train_split, y_train_split)
y_pred_val = pipeline.predict(X_val_split)
val_accuracy = accuracy_score(y_val_split, y_pred_val)
print(f'Validation Accuracy: {val_accuracy}')

# Predict probabilities on the test set
y_prob_test = pipeline.predict_proba(X_test)

# Convert probabilities to DataFrame with the required format
submission_df = pd.DataFrame(y_prob_test, columns=label_encoder.classes_)
submission_df.insert(0, 'Id', test_df['Id'])

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print('Submission file has been created successfully!')
'''
