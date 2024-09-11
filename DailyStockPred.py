import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Step 1: Load historical stock data (ensure your dataset includes 'Close' and 'Volume')
data = pd.read_csv('stock_data.csv')

# Step 2: Feature Engineering
data['PriceChange'] = data['Close'].pct_change()  # percentage change in close price
data['MA10'] = data['Close'].rolling(window=10).mean()  # 10-day moving average
data['VolumeChange'] = data['Volume'].pct_change()  # percentage change in volume

# Label target as 1 for 'up' days and 0 for 'down' days
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Drop rows with NaN values due to rolling windows
data = data.dropna()

# Step 3: Prepare features and target variable
features = ['PriceChange', 'MA10', 'VolumeChange']
X = data[features]
y = data['Target']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and train a Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 6: Make predictions and evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# You can also inspect the model's predictions:
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(predictions)
