"""
Source code for classification practice
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Read Images.csv
images_df = pd.read_csv('Images.csv', header=None, skiprows=1, delimiter=';')
images_df.columns = ['ID', 'Class']

# Read EdgeHistogram.csv
edge_hist_df = pd.read_csv('EdgeHistogram.csv', header=None, skiprows=1, delimiter=';')
edge_hist_df.columns = ['ID'] + [f'Dim_{i}' for i in range(1, len(edge_hist_df.columns))]

# Join dataframes on ID
merged_df = pd.merge(images_df, edge_hist_df, on='ID')

# Split data into features and target
X = merged_df.drop(['ID', 'Class'], axis=1)
y = merged_df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'RandomForestClassifier': RandomForestClassifier()
}

# Define hyperparameters for grid search
parameters = {
    'RandomForestClassifier': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30]}
}

# Train and evaluate models
results = {}
for model_name, model in models.items():
    print(f'Training {model_name}...')
    clf = GridSearchCV(model, parameters[model_name], cv=5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    results[model_name] = {
        'accuracy': accuracy,
        'confusion_matrix': confusion_mat
    }

    print(f'{model_name} accuracy: {accuracy}')

# Save results to CSV files
for model_name, result_data in results.items():
    # Save confusion matrix
    confusion_df = pd.DataFrame(result_data['confusion_matrix'], columns=sorted(y.unique()), index=sorted(y.unique()))
    confusion_df.to_csv(f'{model_name}_confusion_matrix.csv')

# Save hyperparameters to CSV files
for model_name, clf in models.items():
    params = clf.get_params()
    with open(f'{model_name}_hyperparameters.csv', 'w') as file:
        file.write('name,value\n')
        file.write(f'classifier_name,{model_name}\n')
        file.write('library,scikit-learn\n')
        file.write('test_size,0.2\n')
        for param_name, param_value in params.items():
            file.write(f'{param_name},{param_value}\n')

