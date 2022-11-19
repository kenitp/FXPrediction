# -*- coding: utf-8 -*-
# pylint: disable=C0114,C0115,C0103,C0116,E1136
import os
import argparse
import pandas as pd
import utils
from predict import Predict
from training import Training
from model import TrainingModel


class FXPrediction:
    history = None
    model = None
    MODEL_FUNC = [
        TrainingModel.Simple,
        TrainingModel.Simple2,
        TrainingModel.LSTM,
        TrainingModel.LSTM2,
        TrainingModel.RNN,
        TrainingModel.GRU,
    ]

    def __init__(
        self, model, cur_pair, input_len, out_dir, param_fname, col, getNewData=True
    ):
        FXPrediction.model_type = model
        FXPrediction.cur_pair = cur_pair
        FXPrediction.input_len = input_len
        FXPrediction.out_dir = out_dir
        FXPrediction.param_fname_base = self.__make_param_file_path_base(param_fname)
        FXPrediction.use_column = col
        if getNewData is True:
            FXPrediction.df = self.__get_CUR_PAIR(cur_pair, out_dir)
        else:
            FXPrediction.df = utils.read_CUR_PAIR_csv(cur_pair, out_dir)
        FXPrediction.df = self.__calc_technical_chart(FXPrediction.df)
        return

    def __make_param_file_path_base(self, arg_param_name):
        if arg_param_name == "":
            ret = (
                self.out_dir
                + "/"
                + "param_"
                + self.cur_pair
                + "_"
                + self.MODEL_FUNC[self.model_type].__name__
            )
        else:
            ret = arg_param_name
        return ret

    def __calc_technical_chart(self, df):
        cols = ["ema_12", "ema_26"]
        tmp_df = pd.DataFrame(index=df.index, columns=cols)
        tmp_df["ema_12"] = df["Close"].ewm(span=12).mean()
        tmp_df["ema_26"] = df["Close"].ewm(span=26).mean()
        df["macd"] = tmp_df["ema_12"] - tmp_df["ema_26"]
        df["signal"] = df["macd"].ewm(span=9).mean()
        return df

    def __get_CUR_PAIR(self, cur_pair, out_dir):
        if cur_pair == "BTCFXJPY":
            df = utils.get_BTCJPY_csv(cur_pair, out_dir)
        else:
            df = utils.get_CUR_PAIR_from_yahoo(cur_pair, out_dir)
        return df

    @classmethod
    def set_model_type(cls, model_type):
        cls.model_type = model_type
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learn", action="store_true")
    parser.add_argument("-p", "--predict", action="store_true")
    parser.add_argument("-c", "--curpair", type=str, default="USDJPY")
    parser.add_argument(
        "-o", "--outdir", type=str, default=os.path.dirname(__file__) + "/" + "out"
    )
    parser.add_argument("-n", "--paramname", type=str, default="")
    parser.add_argument("-g", "--getcsv", action="store_true")
    parser.add_argument("-e", "--epochs", type=int, default=400)
    parser.add_argument("-s", "--predictsize", type=int, default=16)
    parser.add_argument("-f", "--modelfunc", type=int, default=0)
    parser.add_argument("-t", "--testmodel", type=int, default=0)
    parser.add_argument("-m", "--messagesend", action="store_true")
    args = parser.parse_args()

    # use_col = ['Close', 'High', 'Low', 'macd', 'signal']
    use_col = ["Close", "High", "Low"]

    if args.testmodel != 0:
        fxPredict = FXPrediction(
            args.modelfunc,
            args.curpair,
            args.predictsize,
            args.outdir,
            args.paramname,
            use_col,
        )
        for i in range(args.testmodel):
            print(f"Time: {i}")
            Training.init()
            Training.make_figure_loss(
                fxPredict.out_dir, fxPredict.cur_pair, args.epochs
            )
            Training.make_figure_test(
                fxPredict.out_dir,
                fxPredict.cur_pair,
                args.epochs,
                len(fxPredict.MODEL_FUNC),
            )
            for index, func in enumerate(fxPredict.MODEL_FUNC):
                train = Training(
                    fxPredict, "mse", epochs=args.epochs, test=True, validation=True
                )
                fxPredict.set_model_type(index)
                train.do_training()
                train.plot_loss(func.__name__)
                if len(fxPredict.MODEL_FUNC) == 1:
                    train.plot_test(func.__name__, train.ax_test)
                else:
                    train.plot_test(
                        func.__name__,
                        train.ax_test[index % 3][index // 3],
                    )
                train = None
            Training.plot_savefig(Training.fig_loss, Training.fig_loss_fname)
            Training.plot_savefig(
                Training.fig_test, Training.fig_test_fname, tight=True
            )
            Training.print_result(
                Training.dict_result, Training.fig_test_fname, args.curpair, args.epochs
            )
        return

    model = None
    if args.learn:
        fxPredict = FXPrediction(
            args.modelfunc,
            args.curpair,
            args.predictsize,
            args.outdir,
            args.paramname,
            use_col,
        )
        train = Training(
            fxPredict, "mse", epochs=args.epochs, test=False, validation=False
        )
        train.do_training()
        model = fxPredict.model

    if args.predict:
        fxPredict = FXPrediction(
            args.modelfunc,
            args.curpair,
            args.predictsize,
            args.outdir,
            args.paramname,
            use_col,
            getNewData=False,
        )
        fxPredict.model = model
        predict = Predict(fxPredict)
        predict.do_predict(msg_send=args.messagesend)

    if args.getcsv:
        utils.get_CUR_PAIR_from_yahoo(args.curpair, args.outdir)
        utils.get_DJIA_price(args.outdir)

    return


if __name__ == "__main__":
    main()
