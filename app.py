# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

API_KEY = "c40b181fbf971cb775e6819d4ef87aa4"  # Tarkista, että tämä on oikea ja aktiivinen
LAT, LON = 61.4667, 24.1667  # Kangasala

# Säätiedon haku forecast-rajapinnasta (toimii ilmaisversiossa)
def fetch_weather(lat, lon, api_key):
    url = (
        f"https://api.openweathermap.org/data/2.5/forecast"
        f"?lat={lat}&lon={lon}&units=metric&appid={api_key}"
    )
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Säätiedon haku epäonnistui: {response.status_code} {response.reason}")
        st.stop()
    return response.json()

@st.cache_resource
def train_dummy_model():
    np.random.seed(42)
    df = pd.DataFrame({
        "temp": np.random.normal(18, 5, 100),
        "rain": np.random.exponential(1, 100),
        "weekday": np.random.randint(0, 7, 100),
    })
    df["customers"] = 50 + 3*df["temp"] - 10*df["rain"] + 5*(df["weekday"]>=5) + np.random.normal(0,5,100)
    X = df[["temp", "rain", "weekday"]]
    y = df["customers"]
    model = LinearRegression().fit(X, y)
    joblib.dump(model, "model.joblib")
    return model

model = train_dummy_model()

st.title("Asiakasmääräennuste – Kangasala 🌱")

data = fetch_weather(LAT, LON, API_KEY)

# Muodostetaan päivittäinen keskiarvodata (5 päivää, 3h välein)
df = pd.DataFrame(data["list"])
df["dt"] = pd.to_datetime(df["dt_txt"])
df["date"] = df["dt"].dt.date
df["weekday"] = df["dt"].dt.weekday
df["temp"] = df["main"].apply(lambda x: x["temp"])
df["rain"] = df["rain"].apply(lambda x: x.get("3h", 0) if isinstance(x, dict) else 0)

# Päivittäinen keskiarvo lämpötilasta ja sadekertymä
daily = df.groupby("date").agg({
    "temp": "mean",
    "rain": "sum",
    "weekday": "first"
}).reset_index()

# Näytetään sääennuste
st.subheader("5 päivän sääennuste")
st.line_chart(daily[["temp", "rain"]])

# Ennustetaan asiakasmäärä
Xnew = daily[["temp", "rain", "weekday"]]
pred = model.predict(Xnew)

st.subheader("Arvioidut asiakasmäärät")
st.table(pd.DataFrame({
    "päivä": daily["date"],
    "ennuste": np.round(pred).astype(int)
}))
