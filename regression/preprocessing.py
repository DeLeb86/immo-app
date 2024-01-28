import json
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder,MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor

columns_to_scale=["LivingArea", "GardenArea","SurfaceOfGood","Price_per_sqm"]

def category_type_of_property(df):
    print(df["TypeOfProperty"].dtype)
    property_type=pd.CategoricalDtype(
        categories=["house","apartment","land","office","garage","industry","business","other"]
    )
    
    if df["TypeOfProperty"].dtype!=int:
        df["TypeOfProperty"]=df["TypeOfProperty"].astype(property_type).cat.codes
    return df
def category_state(df):
    building_state_type = pd.CategoricalDtype(
        categories=[
            "to be done up",
            "to restore",
            "to renovate",
            "good",
            "just renovated",
            "as new",
        ],
        ordered=True,
    )
    df["StateOfBuilding"] = (
        df["StateOfBuilding"].astype(building_state_type).cat.codes
    )
    return df

def category_kitchen(df):
    kitchen_type = pd.CategoricalDtype(
        categories=[
            "not installed",
            "usa not installed",
            "installed",
            "semi equipped",
            "usa installed",
            "usa semi equipped",
            "hyper equipped",
            "usa hyper equipped",
        ],
        ordered=True,
    )
    df["Kitchen"] = (
        df["Kitchen"].astype(kitchen_type).cat.codes
    )
    return df

def encode_dataframe(df,encoders_struct=None):
    df=category_kitchen(df)
    df=category_state(df)
    df=category_type_of_property(df)
    #train the encoders if needed
    if encoders_struct is None :
        encoders_struct={}
        for c in columns_to_scale:
            scaler=MinMaxScaler()
            scaler.fit(df[[c]])
            encoders_struct[c]=scaler
            
    for column in columns_to_scale:
        if column in df.columns:
            df[column]=encoders_struct[column].transform(df[[column]])
    
    df.fillna(-1, inplace=True)
    return df,encoders_struct

