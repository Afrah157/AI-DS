import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from google.colab import files

uploaded = files.upload()
df = pd.read_excel("Dataset_AQI4-5.xlsx")

print("Columns:", df.columns.tolist())
print(df.head())

df.dropna(inplace=True)

plt.figure(figsize=(10, 6))
sns.histplot(df['PM2.5'], kde=True)
plt.title('Distribution of PM2.5')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['PM2.5'])
plt.title('Boxplot of PM2.5')
plt.show()

numerical_features = df.select_dtypes(include=np.number).columns.tolist() 

plt.figure(figsize=(10, 6))
sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm')  
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='AQI', y='PM2.5', data=df)
plt.title('AQI vs PM2.5')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(data=df[['PM2.5', 'AQI']])
plt.title('Line Plot of PM2.5 and AQI')
plt.show()

plt.figure(figsize=(10, 6))
sns.pairplot(df["PM2.5 O3 NO2 SO2 CO AQI".split()])
plt.suptitle('Pair Plot of Features', y=1.02)
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='AQI', y='PM2.5', data=df)
plt.title('Violin Plot of PM2.5 vs AQI')
plt.show()

features = ["O3", "NO2", "SO2", "CO", "AQI"]
target = "PM2.5"
X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

def evaluate(name, y_true, y_pred):
    print(f"\n{name} Results")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R² Score:", r2_score(y_true, y_pred))

evaluate("Random Forest", y_test, y_pred_rf)
evaluate("XGBoost", y_test, y_pred_xgb)

plt.figure(figsize=(12,6))
plt.plot(y_test.values[:50], label="Actual", marker='o')
plt.plot(y_pred_rf[:50], label="RF Predicted", linestyle='--')
plt.plot(y_pred_xgb[:50], label="XGB Predicted", linestyle='--')
plt.title("PM2.5 Prediction")
plt.xlabel("Sample")
plt.ylabel("PM2.5")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
