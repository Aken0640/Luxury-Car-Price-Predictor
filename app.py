import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib

app = FastAPI()

model         = joblib.load("car_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")
scaler        = joblib.load("scaler.pkl")
scaler_cols   = joblib.load("scaler_columns.pkl")   # ["Year", "Engine_Size", "Mileage"]

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


def preprocess(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    df["Engine_Size_Group"] = pd.cut(
        df["Engine_Size"],
        bins=[0.996, 2.333, 3.667, 5.0],
        labels=["Small", "Medium", "Large"]
    )
    df["Mileage_Group"] = pd.cut(
        df["Mileage"],
        bins=[-274.922, 99999.0, 199973.0, 299947.0],
        labels=["Low", "Medium", "High"]
    )
    df["Year_of_Registration_Group"] = pd.cut(
        df["Year"],
        bins=[1999.977, 2007.667, 2015.333, 2023.0],
        labels=["2000s", "2010s", "2020s"]
    )

    df = pd.get_dummies(df, columns=[
        "Brand", "Model", "Fuel_Type", "Transmission",
        "Engine_Size_Group", "Mileage_Group", "Year_of_Registration_Group",
        "Doors", "Owner_Count"
    ])   

    df = df.reindex(columns=model_columns, fill_value=0)

    # 4. Scale numeric columns — match notebook Cell 72
    cols_to_scale = [c for c in scaler_cols if c in df.columns]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    df = preprocess(data)
    prediction = model.predict(df)
    return {"prediction": round(float(prediction[0]), 2)}