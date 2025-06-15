import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from flask import Flask, request, render_template

app = Flask(__name__)

def load_and_train_model():
    data = pd.read_csv('MultipleFiles/Student Mental health.csv')

    # Impute missing values
    imputer = SimpleImputer(strategy='most_frequent')
    data[['Age', 'What is your course?', 'Marital status']] = imputer.fit_transform(
        data[['Age', 'What is your course?', 'Marital status']]
    )

    # Drop rows with missing targets or important features
    data.dropna(subset=[
        'Do you have Depression?',
        'Choose your gender',
        'Do you have Anxiety?',
        'Do you have Panic attack?',
        'Did you seek any specialist for a treatment?'
    ], inplace=True)

    # Preprocessing
    data['Your current year of Study'] = data['Your current year of Study'].str.lower().str.strip()
    data['What is your CGPA?'] = data['What is your CGPA?'].str.replace(' ', '').str.replace('-', ' to ')

    def cgpa_to_num(x):
        try:
            if 'to' in x:
                parts = list(map(float, x.split(' to ')))
                return np.mean(parts)
            else:
                return float(x)
        except:
            return np.nan

    data['CGPA_numeric'] = data['What is your CGPA?'].apply(cgpa_to_num)
    data.dropna(subset=['CGPA_numeric'], inplace=True)

    # Remove outliers from CGPA
    Q1 = data['CGPA_numeric'].quantile(0.25)
    Q3 = data['CGPA_numeric'].quantile(0.75)
    IQR = Q3 - Q1
    data = data[(data['CGPA_numeric'] >= (Q1 - 1.5 * IQR)) & (data['CGPA_numeric'] <= (Q3 + 1.5 * IQR))]

    # Encoding categorical variables
    binary_map = {'Yes': 1, 'No': 0}
    gender_map = {'Male': 1, 'Female': 0}
    year_map = {'year 1': 1, 'year 2': 2, 'year 3': 3, 'year 4': 4}

    data['Do you have Depression?'] = data['Do you have Depression?'].map(binary_map)
    data['Choose your gender'] = data['Choose your gender'].map(gender_map)
    data['Marital status'] = data['Marital status'].map(binary_map)
    data['Do you have Anxiety?'] = data['Do you have Anxiety?'].map(binary_map)
    data['Do you have Panic attack?'] = data['Do you have Panic attack?'].map(binary_map)
    data['Did you seek any specialist for a treatment?'] = data['Did you seek any specialist for a treatment?'].map(binary_map)
    data['Your current year of Study'] = data['Your current year of Study'].map(year_map)

    # Drop unused columns
    data.drop(columns=['Timestamp', 'What is your course?', 'What is your CGPA?'], inplace=True)

    features = [
        'Choose your gender', 'Age', 'Your current year of Study', 'CGPA_numeric',
        'Marital status', 'Do you have Anxiety?', 'Do you have Panic attack?',
        'Did you seek any specialist for a treatment?'
    ]
    X = data[features]
    y = data['Do you have Depression?']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    # Feature selection
    rf = RandomForestClassifier(random_state=42)
    rfe = RFE(rf, n_features_to_select=5)
    rfe.fit(X_train, y_train)

    X_train_selected = rfe.transform(X_train)

    # Grid search
    param_grid = {
        'n_estimators': [100, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                               param_grid=param_grid,
                               cv=cv, n_jobs=-1, verbose=0)
    grid_search.fit(X_train_selected, y_train)

    best_model = RandomForestClassifier(**grid_search.best_params_, random_state=42)
    best_model.fit(X_train_selected, y_train)

    return best_model, rfe

# Load trained model
model, selector = load_and_train_model()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    input_data = {}

    if request.method == "POST":
        try:
            input_data['Choose your gender'] = request.form['Choose your gender']
            input_data['Age'] = int(request.form['Age'])
            input_data['Your current year of Study'] = request.form['Your current year of Study'].lower().strip()
            input_data['What is your CGPA?'] = request.form['What is your CGPA?']
            input_data['Marital status'] = request.form['Marital status']
            input_data['Do you have Anxiety?'] = request.form['Do you have Anxiety?']
            input_data['Do you have Panic attack?'] = request.form['Do you have Panic attack?']
            input_data['Did you seek any specialist for a treatment?'] = request.form['Did you seek any specialist for a treatment?']

            binary_map = {'Yes': 1, 'No': 0}
            gender_map = {'Male': 1, 'Female': 0}
            year_map = {'year 1': 1, 'year 2': 2, 'year 3': 3, 'year 4': 4}

            cgpa_input = input_data['What is your CGPA?']
            if 'to' in cgpa_input:
                parts = list(map(float, cgpa_input.replace('-', ' to ').split(' to ')))
                cgpa_num = np.mean(parts)
            else:
                cgpa_num = float(cgpa_input)

            features_vector = [
                gender_map.get(input_data['Choose your gender'], 0),
                input_data['Age'],
                year_map.get(input_data['Your current year of Study'], 1),
                cgpa_num,
                binary_map.get(input_data['Marital status'], 0),
                binary_map.get(input_data['Do you have Anxiety?'], 0),
                binary_map.get(input_data['Do you have Panic attack?'], 0),
                binary_map.get(input_data['Did you seek any specialist for a treatment?'], 0)
            ]

            features_array = np.array(features_vector).reshape(1, -1)
            features_selected = selector.transform(features_array)

            pred = model.predict(features_selected)[0]
            prediction_result = "Depressed" if pred == 1 else "Not Depressed"

        except Exception as e:
            prediction_result = f"Error: {e}"

    return render_template("index.html", prediction_result=prediction_result, input_data=input_data)

# ✅ FIXED: Prevent error when running in cloud or thread-limited environments
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
