import pandas as pd
import numpy as np
from typing import Optional
import json,pickle,sys
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from regression.features import feature_engineering
from sklearn.metrics import mean_squared_error,r2_score
from regression.preprocessing import encode_dataframe
import os

class Property(BaseModel):
    PostalCode: int
    TypeOfProperty: int
    TypeOfSale: int
    Kitchen: Optional[str]
    StateOfBuilding: Optional[str]
    Bedrooms: Optional[float]
    SurfaceOfGood: Optional[float]
    NumberOfFacades: Optional[float]
    LivingArea: float
    GardenArea: Optional[float]

config=json.load(open("resources/config.json"))
PORT = os.environ.get("PORT", 8000)
app = FastAPI(port=PORT)

model=pickle.load(open(config['model_path'],"rb"))
encoder_struct=pickle.load(open(config["encoder_path"],"rb"))
print("model and encoders loaded!")
    

@app.get("/")
async def root():
    """Route that return 'Alive!' if the server runs."""
    return {"Status": "Alive!"}


@app.post("/predict")
async def predict(data:Property):
    df=pd.DataFrame.from_dict(jsonable_encoder(data),orient="index").transpose()
    df=df.reindex(columns=["PostalCode"]+model.feature_names_in_.tolist())
    print(df.head())
    df=feature_engineering(df)
    df.drop("PostalCode",axis=1,inplace=True)
    df,e=encode_dataframe(df,encoder_struct)
    score=np.abs(model.predict(df))
    return {"Prediction":score[0]}
   