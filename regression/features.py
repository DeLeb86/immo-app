import pandas as pd
import json
PSQM_TABLE=json.load(open("resources/price_psqm_table.json"))

def set_psqm(row):
    if str(row["PostalCode"]) in PSQM_TABLE.keys():
        if str(row["TypeOfSale"]) in PSQM_TABLE[str(row["PostalCode"])].keys():
            return PSQM_TABLE[str(row["PostalCode"])][str(row["TypeOfSale"])]
        return None
    return None

def feature_engineering(df) ->pd.DataFrame:
    df["Price_per_sqm"]=df.apply(lambda row: set_psqm(row),axis=1)
    return df       


