# -*- coding: utf-8 -*-
# pylint: disable=C0114,C0115,C0116
from __future__ import annotations
import os
import datetime
from cmath import nan
import numpy as np
import utils
import send_line
from model import TrainingModel


class Predict:
    def __init__(self, fx_predict):
        self.fx_predict = fx_predict
        return

    def __get_latest_prices(self, kind):
        return np.array(
            [
                self.fx_predict.df[kind].iloc[-(i + 1)]
                for i in range(self.fx_predict.input_len)
            ]
        )

    def __make_input_data(self):
        tmp_input_data = []
        for col in self.fx_predict.use_column:
            tmp_input_data.append(self.__get_latest_prices(col))
        return utils.concatenate_price(tmp_input_data)

    def __output_result(self, result, cur_pair, send):
        # 結果出力
        last_date_str = self.fx_predict.df.tail(1).index[0].strftime("%Y-%m-%d")
        last_close_value = self.fx_predict.df["Close"].iloc[-1]
        predict_value = result[0][0]
        message = (
            "\n"
            + f"■{cur_pair}"
            + "\n"
            + f"  前日({last_date_str})終値:\t{last_close_value:.2f}"
            + "\n"
            + f"  次日(AIによる予測)終値:\t{predict_value:.2f}"
        )
        print(f"==== RESULT: {self.fx_predict.cur_pair} ====")
        print(message)

        today = datetime.date.today()
        lastday = self.fx_predict.df.tail(1).index[0].date()
        if (today + datetime.timedelta(days=-1) == lastday) or (cur_pair == "BTCFXJPY"):
            if cur_pair == "BTCFXJPY":
                send_line.send(message, False)
            else:
                send_line.send(message, send)
        else:
            print("Skipped to send message to Line because of market is OFF")

        if send is True:
            tmp_df = utils.read_HISTORY_csv(
                self.fx_predict.cur_pair, self.fx_predict.out_dir
            )
            today_str = today.strftime("%Y-%m-%d")
            tmp_df.at[last_date_str, "当日終値(実績)"] = round(last_close_value, 2)
            tmp_df.loc[today_str] = nan
            tmp_df.at[today_str, "前日終値"] = round(last_close_value, 2)
            tmp_df.at[today_str, "当日終値(予測値)"] = round(predict_value, 2)
            tmp_df.at[today_str, "予測モデル"] = self.fx_predict.MODEL_FUNC[
                self.fx_predict.model_type
            ].__name__
            utils.update_HISTORY_csv(
                tmp_df, self.fx_predict.cur_pair, self.fx_predict.out_dir
            )
            tmp_df = []
        return

    def do_predict(self, msg_send=False):
        print(f"==== Start Prediction of {self.fx_predict.cur_pair} ====")
        weight_fpath = TrainingModel.get_weight_filepath(
            self.fx_predict.param_fname_base
        )
        if os.path.exists(weight_fpath):
            # 入力データ
            input_data = self.__make_input_data()
            input_data, scaler = utils.normalize(input_data)
            print(input_data.shape)
            if self.fx_predict.model is None:
                self.fx_predict.model = TrainingModel.model_load(
                    self.fx_predict.param_fname_base
                )
                # 推論値
                self.fx_predict.model.load_weights(weight_fpath)

            print("Model Loaded: " + self.fx_predict.param_fname_base)
            prediction_norm = utils.predict_price(self.fx_predict.model, input_data)
            prediction = utils.invert_normalize(scaler, prediction_norm)

            # 結果出力
            self.__output_result(prediction, self.fx_predict.cur_pair, msg_send)

        else:
            print(
                f"{self.fx_predict.param_fname_base}.hdf5 isn't exist. Please do training at first."
            )

        return
