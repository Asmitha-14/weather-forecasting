import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.express as px
from tensorflow.keras.models import load_model

st.set_page_config(page_title="AI Weather Forecast", layout="wide")

st.title("AI Weather Forecast Dashboard")

data = pd.read_csv("new_weather.csv")

data["Date"] = pd.to_datetime(data["Date"], format="mixed")
data = data.sort_values("Date")

features = [
"Max Temp (°C)","Min Temp (°C)",
"Max Humidity (%)","Min Humidity (%)",
"Max Wind Speed (Kmph)","Min Wind Speed (Kmph)",
"Rain (mm)"
]

X = data[features]

scalerX = joblib.load("scalerX.save")
scalerY = joblib.load("scalerY.save")

X_scaled = scalerX.transform(X)

time_steps = 14

model = load_model("weather_model.keras")

st.sidebar.header("Prediction Input")

last_date = data["Date"].iloc[-1]

st.sidebar.write("Last data available:", last_date.date())

user_date = st.sidebar.date_input("Select Future Date")

st.sidebar.subheader("Model Performance")

sample = X_scaled[-time_steps:].reshape(1,time_steps,7)
pred_sample = model.predict(sample, verbose=0)

mae = np.mean(np.abs(pred_sample))
rmse = np.sqrt(np.mean((pred_sample)**2))

st.sidebar.write(f"MAE: {round(mae,3)}")
st.sidebar.write(f"RMSE: {round(rmse,3)}")

predict_btn = st.sidebar.button("Predict Weather")

if predict_btn:

    days = (pd.to_datetime(user_date) - last_date).days

    if days <= 0:
        st.warning("Please select a future date")

    elif days > 14:
        st.warning("Prediction reliable up to 14 days")

    else:

        sequence = X_scaled[-time_steps:]
        sequence = sequence.reshape(1,time_steps,7)

        preds = []

        for i in range(days):

            p = model.predict(sequence, verbose=0)
            preds.append(p[0])

            new_row = sequence[0][-1].copy()

            for j in range(7):
                new_row[j] = p[0][j]

            sequence = np.append(sequence[:,1:,:], [[new_row]], axis=1)

        final = scalerY.inverse_transform([preds[-1]])[0]

        st.header("Prediction Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Max Temperature", f"{round(final[0],2)} °C")
        col2.metric("Min Temperature", f"{round(final[1],2)} °C")
        col3.metric("Rainfall", f"{round(final[6],2)} mm")

        col4, col5 = st.columns(2)

        col4.metric("Humidity", f"{round(final[2],2)} %")
        col5.metric("Wind Speed", f"{round(final[4],2)} Kmph")

        st.subheader("Temperature Forecast")

        past = data["Max Temp (°C)"].tail(30).values

        fig = px.line(
            x=list(range(len(past))),
            y=past,
            labels={"x":"Days","y":"Temperature (°C)"},
            title="Temperature Trend"
        )

        fig.add_scatter(
            x=[len(past)],
            y=[final[0]],
            mode="markers",
            name="Predicted"
        )

        st.plotly_chart(fig, use_container_width=True)