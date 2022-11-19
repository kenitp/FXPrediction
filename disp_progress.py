# pylint: disable=C0114,C0115,C0116,W0613,C0301
"""
進捗表示用のCallback関数です。
Batch終了時とEpoch終了時にデータを収集して、表示しています。
"""
import datetime
import tensorflow as tf


class DisplayCallBack(tf.keras.callbacks.Callback):
    # コンストラクタ
    def __init__(self):
        self.last_acc, self.last_loss, self.last_val_acc, self.last_val_loss = (
            None,
            None,
            None,
            None,
        )
        self.now_batch, self.now_epoch = None, None

        self.epochs, self.batch_size = None, None
        # self.samples = None

    # カスタム進捗表示 (表示部本体)
    def print_progress(self):
        epoch = self.now_epoch
        epochs = self.epochs

        if self.last_val_acc and self.last_val_loss:
            print(
                f"\rEpoch {epoch+1}/{epochs} -- acc: {self.last_acc} loss: {self.last_loss} - val_acc: {self.last_val_acc} val_loss: {self.last_val_loss}",
                end="",
            )
        else:
            print(
                f"\rEpoch {epoch+1}/{epochs}  -- acc: {self.last_acc} loss: {self.last_loss}",
                end="",
            )

    # fit開始時
    def on_train_begin(self, logs=None):
        print("\n##### Train Start ##### " + str(datetime.datetime.now()))
        self.epochs = self.params["epochs"]
        self.params["verbose"] = 0

    # batch開始時
    def on_batch_begin(self, batch, logs=None):
        self.now_batch = batch

    # batch完了時 (進捗表示)
    def on_batch_end(self, batch, logs=None):
        self.last_acc = logs.get("acc") if logs.get("acc") else 0.0
        self.last_loss = logs.get("loss") if logs.get("loss") else 0.0
        self.print_progress()

    # epoch開始時
    def on_epoch_begin(self, epoch, log=None):
        self.now_epoch = epoch

    # epoch完了時 (進捗表示)
    def on_epoch_end(self, epoch, logs=None):
        self.last_val_acc = logs.get("val_acc") if logs.get("val_acc") else 0.0
        self.last_val_loss = logs.get("val_loss") if logs.get("val_loss") else 0.0
        self.print_progress()

    # fit完了時
    def on_train_end(self, logs=None):
        print("\n##### Train Complete ##### " + str(datetime.datetime.now()))
