import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Loading preprocessed data
df_train = pd.read_csv('./test.csv/test.csv')

# Preparing the data
X = df_train.drop('Category', axis=1)
y = df_train['Category']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_test_split(X, y, test_size=0.2, random_state=42)

# Training a Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('Random Forest Test Accuracy:', accuracy_score(y_test, y_pred))

# Using XGBoost for a potentially better performance
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {
    'max_depth': 6,
    'eta': 0.1,
    'objective': 'multi:softmax',
    'num_class': len(np.unique(y_train))
}
epochs = 100

bst = xgb.train(param, dtrain, epochs)
preds = bst.predict(dtest)
print('XGBoost Test Accuracy:', accuracy_head(y_test, preds))
