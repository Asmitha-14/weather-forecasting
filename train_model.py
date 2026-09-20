import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

import joblib

data = pd.read_csv("new_weather.csv")

data["Date"] = pd.to_datetime(data["Date"], format="mixed")
data = data.sort_values("Date").reset_index(drop=True)

data = data.drop(["District","Mandal"], axis=1, errors="ignore")
data = data.dropna()

features = [
"Max Temp (°C)","Min Temp (°C)",
"Max Humidity (%)","Min Humidity (%)",
"Max Wind Speed (Kmph)","Min Wind Speed (Kmph)",
"Rain (mm)"
]

targets = features

X = data[features]
y = data[targets].shift(-1)

X = X[:-1]
y = y[:-1]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

time_steps = 14

X_seq = []
y_seq = []

for i in range(len(X_scaled) - time_steps):
    X_seq.append(X_scaled[i:i+time_steps])
    y_seq.append(y_scaled[i+time_steps])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42
)

model = Sequential([
Input(shape=(time_steps,7)),
GRU(128,return_sequences=True),
Dropout(0.2),
GRU(64),
Dense(32,activation="relu"),
Dense(7)
])

model.compile(optimizer="adam",loss="mse")

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test,y_test),
    callbacks=[early_stop],
    verbose=1
)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test,pred)
rmse = np.sqrt(mean_squared_error(y_test,pred))

print("MAE:",mae)
print("RMSE:",rmse)

model.save("weather_model.keras")

joblib.dump(scaler_X,"scalerX.save")
joblib.dump(scaler_y,"scalerY.save")

print("Training complete. Model saved.")