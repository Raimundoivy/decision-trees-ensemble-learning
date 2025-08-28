import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier

# In a real project, you would load a full dataset. For this example, we create a dummy one.
data = [
    {'seniority': 9, 'home': 'rent', 'time': 60, 'age': 30, 'marital': 'married', 'records': 'no', 'job': 'freelance', 'expenses': 73, 'income': 129.0, 'assets': 0.0, 'debt': 0.0, 'amount': 800, 'price': 846, 'default': 0},
    {'seniority': 1, 'home': 'owner', 'time': 36, 'age': 45, 'marital': 'single', 'records': 'no', 'job': 'fixed', 'expenses': 82, 'income': 200.0, 'assets': 5000.0, 'debt': 1000.0, 'amount': 1000, 'price': 1200, 'default': 0},
    {'seniority': 15, 'home': 'owner', 'time': 48, 'age': 55, 'marital': 'married', 'records': 'no', 'job': 'fixed', 'expenses': 90, 'income': 300.0, 'assets': 10000.0, 'debt': 500.0, 'amount': 2000, 'price': 2500, 'default': 0},
    {'seniority': 2, 'home': 'rent', 'time': 24, 'age': 25, 'marital': 'single', 'records': 'yes', 'job': 'partime', 'expenses': 60, 'income': 80.0, 'assets': 100.0, 'debt': 500.0, 'amount': 1200, 'price': 1100, 'default': 1},
    {'seniority': 5, 'home': 'parents', 'time': 36, 'age': 28, 'marital': 'single', 'records': 'no', 'job': 'freelance', 'expenses': 50, 'income': 110.0, 'assets': 0.0, 'debt': 0.0, 'amount': 600, 'price': 900, 'default': 1},
]
df = pd.DataFrame(data)

# Split data into features (X) and target (y)
X_train, _ = train_test_split(df, test_size=0.2, random_state=42)
y_train = X_train['default']
del X_train['default']

# Train the DictVectorizer and transform the data
train_dicts = X_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train_vec = dv.fit_transform(train_dicts)

# Train the XGBoost Model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_vec, y_train)

# Save the vectorizer and model to a single file
output_file = 'model.joblib'
print(f"Saving model and vectorizer to {output_file}...")
joblib.dump((dv, model), output_file)
print("Done.")