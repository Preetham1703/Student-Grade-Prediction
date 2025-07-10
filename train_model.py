# train_model.py
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r"C:\Users\Preet\Downloads\StudentsPerformance.csv")

# Remove 'race/ethnicity'
df = df.drop('race/ethnicity', axis=1)

# Create average score as the target
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

# Drop individual scores from features
X = df.drop(['math score', 'reading score', 'writing score', 'average_score'], axis=1)
X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical features
y = df['average_score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model and columns used
with open('model.pkl', 'wb') as file:
    pickle.dump((model, X.columns), file)

print("Model trained and saved as model.pkl")
