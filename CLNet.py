import pickle
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
from keras.layers import *
from keras.models import *
from keras.regularizers import l2

from scipy.interpolate import splev, splrep
from sklearn.metrics import *


save_name = './CLNet.h5'

train_name = './train_data.pkl'
val_name = './val_data.pkl'
test_name = './test_data.pkl'

ir = 3
before = 2
after = 2

scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def load_data(data_name):
    tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))

    with open(data_name, 'rb') as f:
        apnea_ecg = pickle.load(f)

    x_data = []
    o_data, y_data = apnea_ecg["o_data"], apnea_ecg["y_data"]
    for i in range(len(o_data)):
        (rri_tm, rri_signal), (amp_tm, amp_signal) = o_data[i]
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        amp_interp_signal = splev(tm, splrep(amp_tm, scaler(amp_signal), k=3), ext=1)
        x_data.append([rri_interp_signal, amp_interp_signal])
    x_data = np.array(x_data, dtype="float32").transpose((0, 2, 1))
    y_data = np.array(y_data, dtype="float32")

    return x_data, y_data


def create_model(input_shape, weight=1e-3):
    inputs = Input(shape=input_shape)

    x = Conv1D(64, kernel_size=5, strides=2, padding="valid", activation="relu",
               kernel_initializer="he_normal", kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)
    x = Conv1D(64, kernel_size=5, strides=2, padding="valid", activation="relu",
               kernel_initializer="he_normal", kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)
    x = MaxPooling1D(pool_size=3)(x)

    x = Conv1D(64, kernel_size=5, strides=2, padding="valid", activation="relu",
               kernel_initializer="he_normal", kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)
    x = Conv1D(64, kernel_size=5, strides=2, padding="valid", activation="relu",
               kernel_initializer="he_normal", kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)
    x = MaxPooling1D(pool_size=3)(x)

    x = Dropout(0.8)(x)

    x = LSTM(128)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def lr_schedule(epoch, lr):
    if epoch > 70 and (epoch - 1) % 10 == 0:
        lr *= 0.1
    print("Learning rate: ", lr)
    return lr


def plot(history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["loss"], "r-", history["val_loss"], "b-", linewidth=0.5)
    axes[0].set_title("Loss")
    axes[1].plot(history["accuracy"], "r-", history["val_accuracy"], "b-", linewidth=0.5)
    axes[1].set_title("Accuracy")
    fig.tight_layout()
    fig.show()


if __name__ == "__main__":
    x_train, y_train = load_data(train_name)
    x_val, y_val = load_data(val_name)
    y_train = np_utils.to_categorical(y_train, num_classes=2)
    y_val = np_utils.to_categorical(y_val, num_classes=2)

    model = create_model(input_shape=x_train.shape[1:])
    model.summary()

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    lr_scheduler = LearningRateScheduler(lr_schedule)
    history = model.fit(x_train, y_train, batch_size=128, epochs=100,
                        validation_data=(x_val, y_val), callbacks=[lr_scheduler])
    model.save(save_name)

    x_test, y_test = load_data(test_name)
    y_test = np_utils.to_categorical(y_test, num_classes=2)

    loss, accuracy = model.evaluate(x_test, y_test)
    print("Validation loss: %.4f" % loss)
    print("Accuracy: %.4f" % accuracy)

    print()
    y_score = model.predict(x_test)
    y_predict = np.argmax(y_score, axis=1)
    Y_test = np.argmax(y_test, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(Y_test, y_predict))
    print('                Classification Report')
    print(classification_report(Y_test, y_predict))

    print('Total Test Num: %d, Error Num: %d' % (len(Y_test), (Y_test != y_predict).sum()))
    print('Accuracy: %4.1f' % (accuracy_score(Y_test, y_predict) * 100))

    predict_apnea_label = y_score[:, 1]
    try:
        auc = roc_auc_score(Y_test, predict_apnea_label)
        print('AUC: %9.1f' % (auc * 100))
    except ValueError:
        print('AUC error')
        pass
