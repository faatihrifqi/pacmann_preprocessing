import pandas as pd
import numpy as np
from sklearn import preprocessing

def feature_engineering(df):
    df.rename(columns={
        "dateCreated": "ad_created",
        "dateCrawled": "date_crawled",
        "fuelType": "fuel_type",
        "lastSeen": "last_seen",
        "monthOfRegistration": "registration_month",
        "notRepairedDamage": "unrepaired_damage",
        "nrOfPictures": "num_of_pictures",
        "offerType": "offer_type",
        "postalCode": "postal_code",
        "powerPS": "power_ps",
        "vehicleType": "vehicle_type",
        "yearOfRegistration": "registration_year"
    }, inplace=True)

    df['ad_created'] = pd.to_datetime(df['ad_created'])
    df['date_crawled'] = pd.to_datetime(df['date_crawled'])
    df['last_seen'] = pd.to_datetime(df['last_seen'])

    df['price'] = [int(''.join(x.split("$")[1].split(",")))
                      for x in df['price']]

    df['odometer'] = [int(''.join(x.split("km")[0].split(",")))
                         for x in df['odometer']]
    
    columns_numeric = [i for i in df if df[i].dtype ==
                   "int64" or df[i].dtype == "float64"]
    columns_str = [i for i in df if df[i].dtype == "O"]

    df.drop(columns=["seller", "offer_type"], inplace=True)

    for i in columns_numeric:
        if len(df[i].value_counts()) == 1:
            df.drop(columns=[i], inplace=True)
    
    df.drop(columns=["name", "postal_code"], inplace=True)

    df = df[(df["price"] >= 500) & (df["price"] <= 40000)]

    columns_numeric = [i for i in df if df[i].dtype ==
                   "int64" or df[i].dtype == "float64"]
    columns_str = [i for i in df if df[i].dtype == "O"]

    data = df.copy()

    for i in columns_numeric:
        data[i].fillna(data[i].median(), inplace=True)

    for i in columns_str:
        data[i].fillna(data[i].mode()[0], inplace=True)
    
    columnsToNormalize = [i for i in columns_numeric if i != "price"]

    scaler = preprocessing.StandardScaler()
    data[columnsToNormalize] = scaler.fit_transform(data[columnsToNormalize])

    data = pd.get_dummies(data, columns=[
        i for i in columns_str if i != "unrepaired_damage"], dummy_na=False)
    
    data["unrepaired_damage"] = np.where(data["unrepaired_damage"] == "nein", 0, 1)

    return data