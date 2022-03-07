from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
import numpy as np
import pickle5 as pickle
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from trenchant_utils import regularization
from trenchant_utils import type_code_graph
from trenchant_utils import prepare_train_test
from trenchant_utils import get_lstm
from trenchant_utils import next_labels_cut

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def run_lstm(G_disturbed, cutted_dict, interval, commodity, time_window, i, path, 
                type_feature='node_type', event_type='event', label_number_feature='type_code', embedding_feature='f', dim=512, labels=4, epochs=100, patience=3       
            ):
    G_disturbed = regularization(G_disturbed, iterations=30, mi=0.75, dim=dim)
    G_disturbed = type_code_graph(G_disturbed)
    X_train, X_test, y_train = prepare_train_test(G_disturbed)
    X_train, X_test = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])), np.reshape(
        X_test, (X_test.shape[0], 1, X_test.shape[1]))
    K.clear_session()
    model = get_lstm(dim, labels)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
    model.fit(X_train, y_train, epochs=epochs, batch_size=time_window, callbacks=[callback])
    y_pred = np.argmax(model.predict(X_test), axis=1)
    pd.Series(y_pred).to_csv('{}/pred_iterative_tuned/lstm_regularization-tuned_{}_{}_{}_{}.csv'.format(path,interval,commodity,time_window,i), index=False)

    

path = "/media/pauloricardo/basement/commodities_usecase/"
fine_tune = 'fine-tuned-twelve-months-soy'
fine_tune_dict = {
  'fine-tuned-twelve-weeks-corn': {'interval': 'week', 'commodity': 'corn', 'time_window': 12},
  'fine-tuned-twenty_four-weeks-corn': {'interval': 'week', 'commodity': 'corn', 'time_window': 24},
  'fine-tuned-fourty_eight-weeks-corn': {'interval': 'week', 'commodity': 'corn', 'time_window': 48},
  'fine-tuned-twelve-weeks-soy': {'interval': 'week', 'commodity': 'soybean', 'time_window': 12},
  'fine-tuned-twenty_four-weeks-soy': {'interval': 'week', 'commodity': 'soybean', 'time_window': 24},
  'fine-tuned-fourty_eight-weeks-soy': {'interval': 'week', 'commodity': 'soybean', 'time_window': 48},
  'fine-tuned-three-months-corn': {'interval': 'month', 'commodity': 'corn', 'time_window': 3},
  'fine-tuned-six-months-corn': {'interval': 'month', 'commodity': 'corn', 'time_window': 6},
  'fine-tuned-twelve-months-corn': {'interval': 'month', 'commodity': 'corn', 'time_window': 12},
  'fine-tuned-three-months-soy': {'interval': 'month', 'commodity': 'soybean', 'time_window': 3},
  'fine-tuned-six-months-soy': {'interval': 'month', 'commodity': 'soybean', 'time_window': 6},
  'fine-tuned-twelve-months-soy': {'interval': 'month', 'commodity': 'soybean', 'time_window': 12},
}

label_codes = {
    'big_down': 0,
    'down': 1,
    'up': 2,
    'big_up': 3,
}

# next event label cut LSTM
with open(f"{path}{fine_tune_dict[fine_tune]['commodity']}_{fine_tune_dict[fine_tune]['interval']}_{fine_tune}.gpickle", "rb") as fh:
    G = pickle.load(fh)
G_cutted, cutted_dict = next_labels_cut(G, time_window=fine_tune_dict[fine_tune]['time_window'], interval=fine_tune_dict[fine_tune]['interval'])
y_true = cutted_dict['event_trend'].neighbor.to_list()
for idx in range(len(y_true)):
    y_true[idx] = label_codes[y_true[idx]]
pd.Series(y_true).to_csv(f'{path}/pred_iterative_tuned/true_{fine_tune_dict[fine_tune]["interval"]}_{fine_tune_dict[fine_tune]["commodity"]}_{fine_tune_dict[fine_tune]["time_window"]}.csv', index=False)
for i in range(10):
    print(f'RUN: {fine_tune}, iterative: {i}')
    start_time = time.time()
    run_lstm(G_cutted, cutted_dict, fine_tune_dict[fine_tune]['interval'], fine_tune_dict[fine_tune]['commodity'], fine_tune_dict[fine_tune]['time_window'], i, path)
    with open(f"{path}pred_iterative_tuned/execution_time.txt", 'a') as f:
        f.write(f'regularization_tuned,{fine_tune_dict[fine_tune]["interval"]},{fine_tune_dict[fine_tune]["time_window"]},{fine_tune_dict[fine_tune]["commodity"]},{i},{(time.time() - start_time)}\n')
