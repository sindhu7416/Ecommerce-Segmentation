import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv('marketing_campaign.csv', sep='\t')

# Check column names in the dataset
print("Column names before cleaning:")
print(df.columns)

# Strip any leading or trailing spaces from the column names
df.columns = df.columns.str.strip()

# Check column names again after cleaning
print("\nColumn names after cleaning:")
print(df.columns)

# Define the feature columns and target
features = df.drop(columns=['Response'])  # All columns except the target 'Response'
target = df['Response']

# Define the numerical and categorical features
numeric_features = features.select_dtypes(include=['int64', 'float64']).columns
categorical_features = features.select_dtypes(include=['object']).columns

# Create preprocessing pipeline for numeric data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values by mean imputation
    ('scaler', StandardScaler())  # Standardize the data
])

# Create preprocessing pipeline for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values by most frequent
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One hot encoding for categorical data
])

# Combine both numeric and categorical transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Create a pipeline for Logistic Regression
logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Preprocessing steps
    ('classifier', LogisticRegression(max_iter=1000))  # Logistic Regression classifier
])

# Create a pipeline for Random Forest Classifier
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Preprocessing steps
    ('classifier', RandomForestClassifier(random_state=42))  # Random Forest classifier
])

# Train the Logistic Regression model
logreg_pipeline.fit(X_train, y_train)

# Train the Random Forest model
rf_pipeline.fit(X_train, y_train)

# Evaluate the Logistic Regression model
logreg_predictions = logreg_pipeline.predict(X_test)
logreg_accuracy = classification_report(y_test, logreg_predictions)
print("Logistic Regression Performance:\n", logreg_accuracy)

# Evaluate the Random Forest model
rf_predictions = rf_pipeline.predict(X_test)
rf_accuracy = classification_report(y_test, rf_predictions)
print("Random Forest Performance:\n", rf_accuracy)

# Optionally: Compare accuracy scores
logreg_score = logreg_pipeline.score(X_test, y_test)
rf_score = rf_pipeline.score(X_test, y_test)

print(f'Logistic Regression Accuracy: {logreg_score:.2f}')
print(f'Random Forest Accuracy: {rf_score:.2f}')
