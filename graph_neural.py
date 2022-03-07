from tensorflow.keras import backend as K
from ge import LINE
from ge import Struc2Vec
from ge import Node2Vec
from ge import DeepWalk
import pandas as pd
import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine
import numpy as np
import pickle5 as pickle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from trenchant_utils import metapath2vec
from trenchant_utils import gcn
from trenchant_utils import embedding_graph
from trenchant_utils import regularization
from trenchant_utils import type_code_graph
from trenchant_utils import prepare_train_test
from trenchant_utils import get_lstm
from trenchant_utils import next_labels_cut
import time

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def run_lstm(G_disturbed, cutted_dict, algorithm, interval, commodity, time_window, i, path, 
                type_feature='node_type', event_type='event', label_number_feature='type_code', embedding_feature='f', dim=512, labels=4, epochs=100, patience=5
            ):
    if algorithm == 'regularization':
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
        pd.Series(y_pred).to_csv('{}/pred_iterative_new/lstm_{}_{}_{}_{}_{}.csv'.format(path,algorithm,interval,commodity,time_window,i), index=False)

    elif algorithm == 'deep_walk':
        model_deep_walk = DeepWalk(
            G_disturbed, walk_length=10, num_walks=80, workers=1)
        model_deep_walk.train(window_size=5, iter=3,
                              embed_size=dim)  # train model
        embeddings_deep_walk = model_deep_walk.get_embeddings()  # get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_deep_walk)
        G_disturbed = type_code_graph(G_disturbed)
        X_train, X_test, y_train = prepare_train_test(G_disturbed)
        X_train, X_test = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])), np.reshape(
            X_test, (X_test.shape[0], 1, X_test.shape[1]))
        K.clear_session()
        model = get_lstm(dim, labels)
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
        model.fit(X_train, y_train, epochs=epochs, batch_size=time_window, callbacks=[callback])
        y_pred = np.argmax(model.predict(X_test), axis=1)
        pd.Series(y_pred).to_csv('{}/pred_iterative_new/lstm_{}_{}_{}_{}_{}.csv'.format(path,algorithm,interval,commodity,time_window,i), index=False)

    elif algorithm == 'node2vec':
        model_node2vec = Node2Vec(
            G_disturbed, walk_length=10, num_walks=80, p=0.5, q=1, workers=1)
        model_node2vec.train(window_size=5, iter=3,
                             embed_size=dim)  # train model
        embeddings_node2vec = model_node2vec.get_embeddings()  # get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_node2vec)
        G_disturbed = type_code_graph(G_disturbed)
        X_train, X_test, y_train = prepare_train_test(G_disturbed)
        X_train, X_test = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])), np.reshape(
            X_test, (X_test.shape[0], 1, X_test.shape[1]))
        K.clear_session()
        model = get_lstm(dim, labels)
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
        model.fit(X_train, y_train, epochs=epochs, batch_size=time_window, callbacks=[callback])
        y_pred = np.argmax(model.predict(X_test), axis=1)
        pd.Series(y_pred).to_csv('{}/pred_iterative_new/lstm_{}_{}_{}_{}_{}.csv'.format(path,algorithm,interval,commodity,time_window,i), index=False)

    elif algorithm == 'struc2vec':
        model_struc2vec = Struc2Vec(
            G_disturbed, 10, 80, workers=2, verbose=40)  # init model
        model_struc2vec.train(window_size=5, iter=3,
                              embed_size=dim)  # train model
        embeddings_struc2vec = model_struc2vec.get_embeddings()  # get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_struc2vec)
        G_disturbed = type_code_graph(G_disturbed)
        X_train, X_test, y_train = prepare_train_test(G_disturbed)
        X_train, X_test = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])), np.reshape(
            X_test, (X_test.shape[0], 1, X_test.shape[1]))
        K.clear_session()
        model = get_lstm(dim, labels)
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
        model.fit(X_train, y_train, epochs=epochs, batch_size=time_window, callbacks=[callback])
        y_pred = np.argmax(model.predict(X_test), axis=1)
        pd.Series(y_pred).to_csv('{}/pred_iterative_new/lstm_{}_{}_{}_{}_{}.csv'.format(path,algorithm,interval,commodity,time_window,i), index=False)

    elif algorithm == 'metapath2vec':
        embeddings_metapath2vec = metapath2vec(G_disturbed, dimensions=dim)
        G_disturbed = embedding_graph(G_disturbed, embeddings_metapath2vec)
        G_disturbed = type_code_graph(G_disturbed)
        X_train, X_test, y_train = prepare_train_test(G_disturbed)
        X_train, X_test = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])), np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        K.clear_session()
        model = get_lstm(dim, labels)
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
        model.fit(X_train, y_train, epochs=epochs, batch_size=time_window, callbacks=[callback])
        y_pred = np.argmax(model.predict(X_test), axis=1)
        pd.Series(y_pred).to_csv('{}/pred_iterative_new/lstm_{}_{}_{}_{}_{}.csv'.format(path,algorithm,interval,commodity,time_window,i), index=False)

    elif algorithm == 'line':
        # init model,order can be ['first','second','all']
        model_line = LINE(G_disturbed, embedding_size=dim, order='second')
        model_line.train(batch_size=8, epochs=20, verbose=0)  # train model
        embeddings_line = model_line.get_embeddings()  # get embedding vectors
        G_disturbed = embedding_graph(G_disturbed, embeddings_line)
        G_disturbed = type_code_graph(G_disturbed)
        X_train, X_test, y_train = prepare_train_test(G_disturbed)
        X_train, X_test = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1])), np.reshape(
            X_test, (X_test.shape[0], 1, X_test.shape[1]))
        K.clear_session()
        model = get_lstm(dim, labels)
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
        model.fit(X_train, y_train, epochs=epochs, batch_size=time_window, callbacks=[callback])
        y_pred = np.argmax(model.predict(X_test), axis=1)
        pd.Series(y_pred).to_csv('{}/pred_iterative_new/lstm_{}_{}_{}_{}_{}.csv'.format(path,algorithm,interval,commodity,time_window,i), index=False)

    elif algorithm == 'gcn':
        y_pred = gcn(G_disturbed, interval, i, dimensions=dim)
        pd.Series(y_pred).to_csv('{}/pred_iterative_new/lstm_{}_{}_{}_{}_{}.csv'.format(path,algorithm,interval,commodity,time_window,i), index=False)


algorithms = ['regularization', 'deep_walk', 'node2vec', 'struc2vec', 'metapath2vec', 'line', 'gcn']
path = "/media/pauloricardo/basement/commodities_usecase/"
#intervals = ['week', 'month']
intervals = ['month']
commodities = ['corn', 'soybean']
time_windows = {'month': [3, 6, 12], 'week': [12, 24, 48]}

label_codes = {
    'big_down': 0,
    'down': 1,
    'up': 2,
    'big_up': 3,
}

# next event label cut LSTM
for interval in intervals:
    for commodity in commodities:
        with open(f"{path}{commodity}_{interval}.gpickle", "rb") as fh:
            G = pickle.load(fh)
        for time_window in time_windows[interval]:
            G_cutted, cutted_dict = next_labels_cut(G, time_window=time_window, interval=interval)
            y_true = cutted_dict['event_trend'].neighbor.to_list()
            for idx in range(len(y_true)):
                y_true[idx] = label_codes[y_true[idx]]
            pd.Series(y_true).to_csv('{}/pred_iterative_new/true_{}_{}_{}.csv'.format(path, interval, commodity, time_window), index=False)
            for i in range(10):
                for algorithm in algorithms:
                    print('TEST: {}, {}, {}, {}, {}'.format(algorithm, interval, commodity, time_window, i))
                    start_time = time.time()
                    run_lstm(G_cutted, cutted_dict, algorithm, interval, commodity, time_window, i, path)
                    with open(f"{path}pred_iterative_new/execution_time.txt", 'a') as f:
                        f.write(f'{algorithm},{interval},{time_window},{commodity},{i},{(time.time() - start_time)}\n')