import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# STEP 1: Create dataset
data = {
    'size': [50, 60, 70, 80, 90, 100],
    'rooms': [1, 2, 2, 3, 3, 4],
    'price': [150, 180, 200, 250, 270, 300]
}

df = pd.DataFrame(data)

# STEP 2: Features (input) and target (output)
X = df[['size', 'rooms']]
y = df['price']

# STEP 3: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# STEP 4: Create model
model = LinearRegression()

# STEP 5: Train model
model.fit(X_train, y_train)

# STEP 6: Make prediction
prediction = model.predict([[85, 3]])

print("Predicted house price:", prediction[0])