import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer

# Memuat data
data = pd.read_csv('Student Mental health.csv')

# Langkah-langkah preprocessing
imputer = SimpleImputer(strategy='most_frequent')
data[['Age', 'What is your course?', 'Marital status']] = imputer.fit_transform(data[['Age', 'What is your course?', 'Marital status']])
data.dropna(subset=['Do you have Depression?', 'Choose your gender', 'Do you have Anxiety?', 'Do you have Panic attack?', 'Did you seek any specialist for a treatment?'], inplace=True)
data['Your current year of Study'] = data['Your current year of Study'].str.lower().str.strip()
data['What is your CGPA?'] = data['What is your CGPA?'].str.replace(' ', '').str.replace('-', ' to ')

def parse_cgpa(x):
    try:
        if 'to' in x:
            parts = list(map(float, x.split(' to ')))
            return np.mean(parts)
        return float(x)
    except:
        return np.nan

data['CGPA_numeric'] = data['What is your CGPA?'].apply(parse_cgpa)

Q1 = data['CGPA_numeric'].quantile(0.25)
Q3 = data['CGPA_numeric'].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data['CGPA_numeric'] < (Q1 - 1.5 * IQR)) | (data['CGPA_numeric'] > (Q3 + 1.5 * IQR)))]

mapping_binary = {'Yes': 1, 'No': 0}
data['Do you have Depression?'] = data['Do you have Depression?'].map(mapping_binary).astype(int)
data['Choose your gender'] = data['Choose your gender'].map({'Male': 1, 'Female': 0}).astype(int)

for col in ['Marital status', 'Do you have Anxiety?', 'Do you have Panic attack?', 'Did you seek any specialist for a treatment?']:
    data[col] = data[col].map(mapping_binary).astype(int)

data['Your current year of Study'] = data['Your current year of Study'].map({'year 1':1,'year 2':2,'year 3':3,'year 4':4}).fillna(0).astype(int)
data.drop(columns=['Timestamp', 'What is your course?', 'What is your CGPA?'], inplace=True, errors='ignore')

X = data[['Choose your gender', 'Age', 'Your current year of Study', 'CGPA_numeric', 'Marital status', 'Do you have Anxiety?', 'Do you have Panic attack?', 'Did you seek any specialist for a treatment?']]
y = data['Do you have Depression?']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

rf = RandomForestClassifier(random_state=42, n_estimators=100)
rfe = RFE(rf, n_features_to_select=5)
rfe.fit(X_train, y_train)

X_train_selected = rfe.transform(X_train)
X_test_selected = rfe.transform(X_test)

param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=cv, n_jobs=-1)
grid_search.fit(X_train_selected, y_train)

best_rf = RandomForestClassifier(**grid_search.best_params_, random_state=42)
best_rf.fit(X_train_selected, y_train)

# Simpan model
with open('model_depression.pkl', 'wb') as f:
    pickle.dump({'model': best_rf, 'selector': rfe, 'features': X.columns.tolist()}, f)
print("Model dan selector sudah disimpan sebagai model_depression.pkl")
