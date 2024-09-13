import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv('stock_data.csv')

data['PriceChange'] = data['Close'].pct_change()  
data['MA10'] = data['Close'].rolling(window=10).mean()  
data['VolumeChange'] = data['Volume'].pct_change() 

data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data = data.dropna()

features = ['PriceChange', 'MA10', 'VolumeChange']
X = data[features]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(predictions)
