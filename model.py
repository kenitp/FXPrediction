# -*- coding: utf-8 -*-
# pylint: disable=C0114,C0115,C0103,C0116,W0613
import inspect
from keras.layers import Activation, Dense, Flatten
from keras.layers import LSTM, SimpleRNN, GRU, Dropout
from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import Adam


class TrainingModel:
    @staticmethod
    def Simple(
        inputs,
        neurons=20,
        dropout=0.25,
        loss="mse",
        optimizer="adam",
        activation="linear",
    ):
        print(
            "USE MODEL: "
            + inspect.currentframe().f_code.co_name
            + ", Input Shape = "
            + str(inputs.shape)
        )
        model = Sequential()
        model.add(
            Dense(8, activation="relu", input_shape=(inputs.shape[1], inputs.shape[2]))
        )
        model.add(Dense(8, activation="relu"))
        model.add(Dense(8, activation="relu"))
        model.add(Dense(1))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=[loss])
        return model

    @staticmethod
    def Simple2(
        inputs,
        neurons=20,
        dropout=0.25,
        loss="mse",
        optimizer="adam",
        activation="linear",
    ):
        print(
            "USE MODEL: "
            + inspect.currentframe().f_code.co_name
            + ", Input Shape = "
            + str(inputs.shape)
        )
        model = Sequential()
        model.add(Flatten(input_shape=(inputs.shape[1], inputs.shape[2])))
        model.add(Dense(32))
        model.add(Activation("relu"))
        model.add(Dense(32))
        model.add(Activation("relu"))
        model.add(Dense(1))
        model.add(Activation(activation))
        model.compile(loss=loss, optimizer=Adam(lr=0.001), metrics=[loss])
        return model

    @staticmethod
    def LSTM(
        inputs,
        neurons=20,
        dropout=0.25,
        loss="mse",
        optimizer="adam",
        activation="linear",
    ):
        print("Neurons: " + str(neurons))
        print(
            "USE MODEL: "
            + inspect.currentframe().f_code.co_name
            + ", Input Shape = "
            + str(inputs.shape)
        )
        model = Sequential()
        model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
        model.add(Dropout(dropout))
        model.add(Dense(units=1))
        model.add(Activation(activation))
        model.compile(loss=loss, optimizer=optimizer, metrics=[loss])
        return model

    @staticmethod
    def LSTM2(
        inputs,
        neurons=20,
        dropout=0.25,
        loss="mse",
        optimizer="sgd",
        activation="linear",
    ):
        print(
            "USE MODEL: "
            + inspect.currentframe().f_code.co_name
            + ", Input Shape = "
            + str(inputs.shape)
        )
        model = Sequential()
        model.add(
            LSTM(
                neurons,
                input_shape=(inputs.shape[1], inputs.shape[2]),
                return_sequences=False,
            )
        )
        model.add(Dense(1, activation=activation))
        model.compile(loss=loss, optimizer=optimizer, metrics=[loss])
        return model

    @staticmethod
    def RNN(
        inputs,
        neurons=20,
        dropout=0.25,
        loss="mse",
        optimizer="sgd",
        activation="linear",
    ):
        print(
            "USE MODEL: "
            + inspect.currentframe().f_code.co_name
            + ", Input Shape = "
            + str(inputs.shape)
        )
        model = Sequential()
        model.add(
            SimpleRNN(
                neurons,
                input_shape=(inputs.shape[1], inputs.shape[2]),
                return_sequences=False,
            )
        )
        model.add(Dense(1, activation=activation))
        model.compile(loss=loss, optimizer=optimizer, metrics=[loss])
        return model

    @staticmethod
    def GRU(
        inputs,
        neurons=20,
        dropout=0.25,
        loss="mse",
        optimizer="sgd",
        activation="linear",
    ):
        print(
            "USE MODEL: "
            + inspect.currentframe().f_code.co_name
            + ", Input Shape = "
            + str(inputs.shape)
        )
        model = Sequential()
        model.add(
            GRU(
                neurons,
                input_shape=(inputs.shape[1], inputs.shape[2]),
                return_sequences=False,
            )
        )
        model.add(Dense(1, activation=activation))
        model.compile(loss=loss, optimizer=optimizer, metrics=[loss])
        return model

    @staticmethod
    def model_compile(func, input_data, neurons, loss, activation):
        return func(input_data, neurons, loss, activation)

    @staticmethod
    def model_save(model, fname_base):
        json_string = model.to_json()
        with open(
            TrainingModel.get_model_filepath(fname_base), "w", encoding="utf-8"
        ) as f:
            f.write(json_string)
        return

    @staticmethod
    def model_load(fname_base):
        json_string = open(
            TrainingModel.get_model_filepath(fname_base), encoding="utf-8"
        ).read()
        return model_from_json(json_string)

    @staticmethod
    def weight_save(model, fname_base):
        model.save_weights(TrainingModel.get_weight_filepath(fname_base))
        return

    @staticmethod
    def weight_load(model, fname_base):
        model.load_weights(TrainingModel.get_weight_filepath(fname_base))
        return

    @staticmethod
    def get_weight_filepath(fname_base):
        return fname_base + ".hdf5"

    @staticmethod
    def get_model_filepath(fname_base):
        return fname_base + ".json"
