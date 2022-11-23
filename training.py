# -*- coding: utf-8 -*-
# pylint: disable=C0114,C0115,C0116,C0103,W0612
from __future__ import annotations
from cmath import nan
import math
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import dates as mdates
import utils
from disp_progress import DisplayCallBack
from model import TrainingModel
import send_line

USDJPY_ANALYSIS_DIFF = 0.3
BTCJPY_ANALYSIS_DIFF = 10000


class Training:
    fig_loss, ax_loss = None, None
    fig_test, ax_test = None, None
    fig_loss_fname, fig_test_fname = None, None
    dict_result: dict = {}

    def __init__(self, fx_predict, loss="mse", epochs=400, test=True, validation=False):
        self.loss = loss
        self.epochs = epochs
        self.test = test
        self.validation = validation
        self.fx_predict = fx_predict
        self.neurons = 100
        self.activation = "linear"
        self.fig_predicted_y, self.fig_test_y = None, None
        self.fig_test_date = None
        return

    @classmethod
    def init(cls):
        cls.fig_loss, cls.ax_loss = None, None
        cls.fig_test, cls.ax_test = None, None
        cls.fig_loss_fname, cls.fig_test_fname = None, None
        cls.dict_result = {}
        return

    def do_training(self):
        print(f"==== Start Training for Prediction of {self.fx_predict.cur_pair} ====")

        df = self.fx_predict.df
        for col in self.fx_predict.use_column:
            df.loc[col] = df[col].astype(float)

        # 同じ行に過去 input_len 時点の価格(予想に使う説明変数)と今の価格（目的変数）が並ぶようにする
        for i in range(self.fx_predict.input_len):
            for col in self.fx_predict.use_column:
                df[col + str(i + 1)] = df[col].shift(i + 1)

        df = df.dropna(how="any")

        # dfをnumpy配列に変換 (訓練データ(train_x)は過去 input_len 時点までの価格*行数となっている)
        tmp_train_x = []
        for j in range(len(self.fx_predict.use_column)):
            tmp_train_x.append(
                df[
                    [
                        self.fx_predict.use_column[j] + str(i + 1)
                        for i in range(self.fx_predict.input_len)
                    ]
                ].values
            )

        train_x = utils.concatenate_price(tmp_train_x)
        train_y = df["Close"].values

        train_x, scaler_x = utils.normalize(train_x)
        train_y, scaler_y = utils.normalize(train_y)

        # テストを実行する場合は訓練データの一部をテストデータとして取っておく
        if self.test is True:
            train_x, test_x = train_test_split(
                train_x, train_size=0.9, test_size=0.1, shuffle=False
            )
            train_y, test_y = train_test_split(
                train_y, train_size=0.9, test_size=0.1, shuffle=False
            )

        # モデルの作成 & コンパイル
        model = self.fx_predict.MODEL_FUNC[self.fx_predict.model_type](
            train_x, neurons=self.neurons, loss=self.loss, activation=self.activation
        )
        # print(model.summary())

        # 学習の実行
        if self.validation is True:
            val_split = 0.2
        else:
            val_split = 0.0

        cb_display = DisplayCallBack()
        self.fx_predict.history = model.fit(
            train_x,
            train_y,
            epochs=self.epochs,
            validation_split=val_split,
            verbose=0,
            callbacks=[cb_display],
        )

        # テスト実行時はテスト結果を分析
        if self.test is True:
            # テストデータで予測値を計算
            predicted_y = utils.predict_price(model, test_x)
            # グラフ描画用データの保存
            self.fig_test_y = utils.invert_normalize(scaler_y, test_y)
            self.fig_predicted_y = utils.invert_normalize(scaler_y, predicted_y)
            self.fig_test_date = df.tail(len(self.fig_test_y)).index.values

            # ヒット率分析
            # テストデータ(test_x)は、[Close, High, Low, ...]がinput_len個ずつ並んだ3次元行列
            # Closeだけを切り出す
            test_x = np.dsplit(test_x, len(self.fx_predict.use_column))[0].reshape(
                len(test_x), self.fx_predict.input_len
            )
            inv_test_x = utils.invert_normalize(scaler_x, test_x[:, 0])

            self.analyze_test_result(
                inv_test_x,
                self.fig_test_y,
                self.fig_predicted_y,
                self.fx_predict.MODEL_FUNC[self.fx_predict.model_type].__name__,
            )
        else:
            # 学習結果をファイルに保存(テスト実行時は保存しない)
            TrainingModel.model_save(model, self.fx_predict.param_fname_base)
            TrainingModel.weight_save(model, self.fx_predict.param_fname_base)
        self.fx_predict.model = model
        df = None
        return

    def analyze_test_result(self, test_x, test_y, predicted_y, model_name):
        if self.fx_predict.cur_pair == "USDJPY":
            diff = USDJPY_ANALYSIS_DIFF
        else:
            diff = BTCJPY_ANALYSIS_DIFF

        df = pd.DataFrame(
            index=[self.fig_test_date], columns=["Close", "Predicted", "Real"]
        )
        df["Close"] = test_x
        df["Predicted"] = predicted_y
        df["Real"] = test_y
        df.loc[df["Predicted"] - df["Close"] >= diff, "UpDown(P)"] = "Up"
        df.loc[df["Close"] - df["Predicted"] >= diff, "UpDown(P)"] = "Down"
        df.loc[abs(df["Predicted"] - df["Close"]) < diff, "UpDown(P)"] = nan
        df = df.dropna(how="any")
        # pylint: disable=E1136
        df.loc[df["Real"] - df["Close"] > 0, "UpDown(R)"] = "Up"
        df.loc[df["Real"] - df["Close"] <= 0, "UpDown(R)"] = "Down"
        df.loc[df["UpDown(P)"] == "Up", "Profit"] = df["Real"] - df["Close"]
        df.loc[df["UpDown(P)"] == "Down", "Profit"] = df["Close"] - df["Real"]
        numerator = len(df.loc[df["UpDown(P)"] == df["UpDown(R)"], :])
        denominator = len(df)
        self.dict_result[model_name] = [numerator / denominator, df["Profit"].sum()]
        df = None
        return

    @classmethod
    def make_fig_file_path(cls, kind, out_dir, cur_pair, epochs):
        fig_fname = f"{out_dir}/{kind}_{cur_pair}_e{epochs}"
        return fig_fname

    @classmethod
    def make_figure_loss(cls, out_dir, cur_pair, epochs):
        cls.fig_loss, cls.ax_loss = plt.subplots()
        cls.fig_loss_fname = cls.make_fig_file_path("train", out_dir, cur_pair, epochs)
        return

    @classmethod
    def make_figure_test(cls, out_dir, cur_pair, epochs, num):
        if num == 1:
            cls.fig_test, cls.ax_test = plt.subplots(figsize=(10, 10), dpi=200)
        else:
            row = 3
            cls.fig_test, cls.ax_test = plt.subplots(
                row, math.ceil(num / row), figsize=(10, 10), dpi=200
            )
        cls.fig_test_fname = cls.make_fig_file_path("test", out_dir, cur_pair, epochs)
        return

    def plot_loss(self, model):
        # 学習中の評価値の推移
        self.ax_loss.plot(
            self.fx_predict.history.history[self.loss], label="train(" + model + ")"
        )
        self.ax_loss.plot(
            self.fx_predict.history.history["val_" + self.loss],
            label="val(" + model + ")",
        )
        self.ax_loss.set_xlabel("epoch")
        self.ax_loss.set_ylabel(self.loss)
        self.ax_loss.legend(loc="best")
        bottom, top = self.ax_loss.get_ylim()
        if max(self.fx_predict.history.history[self.loss]) < top:
            self.ax_loss.set_ylim([0, max(self.fx_predict.history.history[self.loss])])
        return

    def plot_test(self, title, ax):
        ax.set_title(title)
        ax.plot(self.fig_test_date, self.fig_test_y, label="real")
        ax.plot(self.fig_test_date, self.fig_predicted_y, label="predicted")
        ax.legend(loc="best")
        ax.set_xticks(self.fig_test_date)
        ax.set_xticklabels(self.fig_test_date, rotation=45)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m/%d"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        return

    @staticmethod
    def plot_savefig(fig, outfile, tight=False):
        dtstr = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        dt.datetime.now()
        if tight is True:
            fig.tight_layout()
        fig.savefig(outfile + "_" + dtstr + ".png")
        return

    @staticmethod
    def print_result(result, outfile, cur_pair, epochs):
        dtstr = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        txt = "\n" + "■" + cur_pair + "\n" + "EPOCHS: " + str(epochs)
        with open(outfile + "_" + dtstr + ".txt", "w", encoding="utf-8") as f:
            print("\r\n=== SUMMARY ======")
            for key in result.keys():
                txt = (
                    txt
                    + "\n"
                    + "MODEL: "
                    + key.ljust(7)
                    + ", Hit: "
                    + str(round(result[key][0] * 100, 1))
                    + "%, Profit: \\"
                    + str(round(result[key][1], 1)).rjust(5)
                )

            print(txt)
            f.write(txt + "\n")
            print("\r\n")

        send_line.send(txt, False)
        return
