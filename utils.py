# -*- coding: utf-8 -*-
# pylint: disable=C0114,C0116,C0103
import datetime
import os

import numpy as np
import pandas as pd
import pandas_datareader.data as DataReader
import requests
import yfinance as yf
from sklearn import preprocessing

FX_START_DAY = "2009-12-31"

BTC_URL = "https://api.cryptowat.ch/markets/bitflyer/btcjpy/ohlc"
BTC_FX_URL = "https://api.cryptowat.ch/markets/bitflyer/btcfxjpy/ohlc"
# BTC_START_DAY = "?periods=86400&after=1396278000"
BTC_START_DAY = "?periods=86400&after=1617202800"
BTC_DF_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "QuoteVolume"]


def get_CUR_PAIR_from_yahoo(cur_pair, out_dir):

    # start_dayから今日までのデータをダウンロードし、学習データとする。
    yf.pdr_override()
    df = DataReader.get_data_yahoo(cur_pair + "=X", start=FX_START_DAY)

    today = datetime.date.today()
    lastday = df.tail(1).index[0].date()
    if today == lastday:
        df = df[:-1]
    # ドル円データをcsvにして保存しておく
    make_dir(out_dir)
    df.to_csv(f"{out_dir}/{cur_pair}.csv")
    df = df.drop("Open", axis=1)
    df = df.drop("Adj Close", axis=1)
    df = df.drop("Volume", axis=1)
    return df


def get_BTCJPY_csv(cur_pair, out_dir):
    if cur_pair == "BTCJPY":
        url = BTC_URL
    else:
        url = BTC_FX_URL

    res = requests.get(url + BTC_START_DAY, timeout=30).json()
    df = pd.DataFrame(res["result"]["86400"], columns=BTC_DF_COLUMNS)
    df["Date"] = pd.to_datetime(df["Date"], unit="s")
    df.loc[:, "Date"] = df.loc[:, "Date"] + datetime.timedelta(days=-1)
    df = df[:-1]
    df = df.set_index("Date")
    df.to_csv(f"{out_dir}/{cur_pair}.csv")
    df = df.drop("Open", axis=1)
    df = df.drop("Volume", axis=1)
    df = df.drop("QuoteVolume", axis=1)
    return df


def get_DJIA_price(out_csv_path):
    start = datetime.date(2009, 12, 31)
    end = datetime.date.today()
    df = DataReader.DataReader("^DJI", data_source="stooq", start=start, end=end)
    df.to_csv(out_csv_path + "/DJI.csv")
    df = df.drop("Adj Close", axis=1)
    df = df.drop("Volume", axis=1)
    return df


def read_CUR_PAIR_csv(cur_pair, csv_path):
    df = pd.read_csv(csv_path + "/" + cur_pair + ".csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    return df


def read_HISTORY_csv(cur_pair, csv_path):
    df = pd.read_csv(csv_path + "/" + "predict_history_" + cur_pair + ".csv")
    df = df.set_index("Date")
    return df


def update_HISTORY_csv(df, cur_pair, csv_path):
    df.to_csv(csv_path + "/" + "predict_history_" + cur_pair + ".csv")
    return


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return


def normalize(data):
    tmp_shape = data.shape
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    data_norm = scaler.fit_transform(data.reshape(-1, 1))
    if len(tmp_shape) == 3:
        data_norm = data_norm.reshape(-1, tmp_shape[1], tmp_shape[2])
    return data_norm, scaler


def invert_normalize(scaler, value_norm):
    value_raw = scaler.inverse_transform(value_norm.reshape([-1, 1])).reshape([-1, 1])
    return value_raw


def predict_price(model, input_x):
    predict_y = model.predict(input_x)
    if len(predict_y.shape) == 3:
        predict_y = predict_y[:, 0]
    return predict_y


def concatenate_price(col_list):
    return np.dstack(col_list)
