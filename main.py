import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('sp500_stocks.csv')

# Finding missing value
print(df.isnull().sum())

# Setting up the target column
df['Target'] = df['Close'].shift(-1)

# Handling the missing values
df.fillna(0, inplace= True)

df2 = df.copy()

fifty_symbols = df2['Symbol'].sample(50, random_state= 42).tolist()

df2 = df2[df2['Symbol'].isin(fifty_symbols)]

# One-hot encoding symbol feature
symbol_dummies = pd.get_dummies(df2['Symbol'], drop_first= True, dtype= int)
df2 = pd.concat([df2, symbol_dummies], axis= 1)
df2.drop('Symbol', axis= 1, inplace= True)

# Removing unnecessary columns
df2.drop(['Date','Adj Close'], axis= 1, inplace=True)

# Scaling the values
scaler = StandardScaler()
df2[['Close','High','Low','Open','Volume','Target']] = scaler.fit_transform(df2[['Close','High','Low','Open','Volume','Target']])

# Float64 is taking too much memory and it raises an exception
x = df2.drop('Target', axis= 1).astype('float32')
y = df2['Target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)

xgb_model = XGBRegressor(
    objective = 'reg:squarederror',
    n_estimators = 100,
    learning_rate = 0.1,
    max_depth = 6,
    gamma = 0,
    random_state = 42,
    importance_type = 'gain'
)

xgb_model.fit(x_train, y_train)

y_pred = xgb_model.predict(x_test)

print("MSE: ", mean_squared_error(y_test, y_pred))
print("R2 : ", r2_score(y_test, y_pred))

'''
# Clearly understood
svr_model = SVR(
    kernel= 'rbf',
    gamma= 'scale',
    C= 1,
    epsilon= 0.1
)

svr_model.fit(x_train, y_train)

y_pred = svr_model.predict(x_test)

print("MSE: ", mean_squared_error(y_test, y_pred))
print("R2 : ", r2_score(y_test, y_pred))
'''

## These steps are for the whole dataset from the encoding part for all 500 companies ##
'''
# One-hot encoding symbol feature
symbol_dummies = pd.get_dummies(df['Symbol'], drop_first= True, dtype= int)
df = pd.concat([df, symbol_dummies], axis= 1)
df.drop('Symbol', axis= 1, inplace= True)

# Removing unnecessary columns
df.drop(['Date','Adj Close'], axis= 1, inplace=True)

# Scaling the values
scaler = StandardScaler()
df[['Close','High','Low','Open','Volume','Target']] = scaler.fit_transform(df[['Close','High','Low','Open','Volume','Target']])

# Float64 is taking too much memory and it raises an exception
x = df.drop('Target', axis= 1).astype('float32')
y = df['Target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)

model = LinearRegression(
    fit_intercept= True,
    copy_X= True,
    positive= False
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("MSE: ", mean_squared_error(y_test, y_pred))
print("R2 : ", r2_score(y_test, y_pred))

plt.figure(figsize=(12,7))
plt.plot(y_test.values ,label = 'Actual')
plt.plot(y_pred, label = 'Predicted')
plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Sample Index')
plt.ylabel('Scaled Price')
plt.legend()
plt.show()
'''