"""
Created on 2018/10/9 by Chun-hui Yin(yinchunhui.ahu@gmail.com).
Description: Script file for running our experiments on response-time QoS data.
"""
import argparse
import multiprocessing
import os
import sys
from time import time

import numpy as np
from keras import initializers
from keras import regularizers
from keras.layers import Embedding, Input, Dense, Flatten, GRU, Reshape, Concatenate, K, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

from DataSet import DataSet
from Evaluator import evaluate, saveResult

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    parser = argparse.ArgumentParser(description="Parameter Settings")
    parser.add_argument('--dataType', default='rt', type=str, help='Type of data:rt|tp.')
    parser.add_argument('--shape', default=(142, 4500, 64), type=tuple, help='(UserNum,ItemNum,TimeNum).')
    parser.add_argument('--parallel', default=False, type=bool, help='Whether to use multi-process.')
    parser.add_argument('--density', default=list(np.arange(0.05, 21, 0.05)), type=list, help='Density of matrix.')
    parser.add_argument('--epochNum', default=50, type=int, help='Numbers of epochs per run.')
    parser.add_argument('--batchSize', default=2048, type=int, help='Size of a batch.')
    parser.add_argument('--gruLayers', default=[2048, 1, 1], type=list, help='Layers of MLP.')
    parser.add_argument('--regLayers', default=[0., 0., 0.], type=list, help='Regularization.')
    parser.add_argument('--dropLayers', default=[5e-1, 5e-1, 5e-1], type=list, help='Dropout.')
    parser.add_argument('--optimizer', default=Adam, type=str, help='The optimizer:Adam|Adamax|Nadam|Adagrad.')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate of the model.')
    parser.add_argument('--decay', default=0.0, type=float, help='Decay ratio for lr.')
    parser.add_argument('--verbose', default=1, type=int, help='Iterations per evaluation.')
    parser.add_argument('--store', default=True, type=bool, help='Whether to store the model and result.')
    parser.add_argument('--dataPath', default='./Data/dataset#2/', type=str, help='Path to load data.')
    parser.add_argument('--modelPath', default='./Model/', type=str, help='Path to save the model.')
    parser.add_argument('--imagePath', default='./Image/', type=str, help='Path to save the image.')
    parser.add_argument('--resultPath', default='./Result/', type=str, help='Path to save the result.')
    args = parser.parse_args()

    if args.parallel:
        pool = multiprocessing.Pool()
        for density in args.density:
            pool.apply_async(PGRU, (args, density))
        pool.close()
        pool.join()
    else:
        for density in args.density:
            model = PGRU(args, density)
            del model


class PGRU:

    def __init__(self, args, density):

        self.dataSet = DataSet(args, density)
        self.dataType = self.dataSet.dataType
        self.density = self.dataSet.density
        self.shape = self.dataSet.shape

        self.train = self.dataSet.train
        self.test = self.dataSet.test

        self.epochNum = args.epochNum
        self.batchSize = args.batchSize
        self.gruLayers = args.gruLayers
        self.regLayers = args.regLayers
        self.dropLayers = args.dropLayers
        self.lr = args.lr
        self.decay = args.decay
        self.optimizer = args.optimizer
        self.verbose = args.verbose

        self.store = args.store
        self.modelPath = args.modelPath
        self.imagePath = args.imagePath
        self.resultPath = args.resultPath

        self.model = self.load_model()

        self.run()

    def run(self):
        # Initialization
        x_test, y_test = self.dataSet.getTestInstance(self.test)
        print('Initializing...')
        mae, rmse = evaluate(self.model, x_test, y_test, self.batchSize)
        sys.stdout.write('\rInitializing done.MAE = %.4f|RMSE = %.4f.\n' % (mae, rmse))
        best_mae, best_rmse, best_epoch = mae, rmse, -1
        metrics = ['MAE', 'RMSE']
        evalResults = np.zeros((self.epochNum, 2))
        # Training model
        print('=' * 14 + 'Start Training' + '=' * 22)
        for epoch in range(self.epochNum):
            sys.stdout.write('\rEpoch %d starts...' % epoch)
            start = time()
            x_train, y_train = self.dataSet.getTrainInstance(self.train)
            # Training
            history = self.model.fit(x_train, y_train, batch_size=self.batchSize, epochs=1, verbose=0, shuffle=True)
            # , callbacks=[TensorBoard(log_dir='./Log')])
            end = time()
            sys.stdout.write('\rEpoch %d ends.[%.1fs]' % (epoch, end - start))
            # Evaluation
            if epoch % self.verbose == 0:
                sys.stdout.write('\rEvaluating Epoch %d...' % epoch)
                mae, rmse = evaluate(self.model, x_test, y_test, self.batchSize)
                loss = history.history['loss'][0]
                sys.stdout.write('\rEvaluating completes.[%.1fs] ' % (time() - end))
                if mae < best_mae:
                    best_mae, best_rmse, best_epoch = mae, rmse, epoch
                evalResults[epoch, :] = [mae, rmse]
                saveResult('pgru', self.resultPath, self.dataType, self.density, evalResults, metrics)
                sys.stdout.write('\rEpoch %d : MAE = %.4f|RMSE = %.4f|Loss = %.4f\n' % (epoch, mae, rmse, loss))
        print('=' * 14 + 'Training Complete!' + '=' * 18)
        print('The best is at epoch %d : MAE = %.4f|RMSE = %.4f.' % (best_epoch, best_mae, best_rmse))
        if self.store:
            self.save_model(self.model)
            print('The model is stored in %s.' % self.modelPath)
            print('The result is stored in %s.' % self.resultPath)

    def load_model(self):

        _model = self.build_model(self.shape, self.gruLayers, self.regLayers,
                                  self.dropLayers)
        _model.compile(optimizer=self.optimizer(lr=self.lr, decay=self.decay), loss=self.hybrid_loss)
        plot_model(_model, to_file=self.imagePath + 'PGRU.jpg', show_shapes=True)
        return _model

    @staticmethod
    def build_model(shape, gru_layers, reg_layers, drop_layers):

        # Embedding Layer
        user_input = Input(shape=(1,), dtype='int32', name='user_input')

        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        time_input = Input(shape=(1,), dtype='int32', name='time_input')

        user_embedding = Flatten()(Embedding(input_dim=shape[0], output_dim=gru_layers[0],
                                             embeddings_initializer=initializers.random_normal(),
                                             embeddings_regularizer=regularizers.l2(reg_layers[0]),
                                             input_length=1, name='gru_user_embedding')(user_input))
        item_embedding = Flatten()(Embedding(input_dim=shape[1], output_dim=gru_layers[0],
                                             embeddings_initializer=initializers.random_normal(),
                                             embeddings_regularizer=regularizers.l2(reg_layers[0]),
                                             input_length=1, name='gru_item_embedding')(item_input))
        time_embedding = Flatten()(Embedding(input_dim=shape[2], output_dim=gru_layers[1],
                                             embeddings_initializer=initializers.random_normal(),
                                             embeddings_regularizer=regularizers.l2(reg_layers[0]),
                                             input_length=1, name='gru_time_embedding')(time_input))

        user_embedding = Dropout(drop_layers[0])(user_embedding)
        item_embedding = Dropout(drop_layers[0])(item_embedding)
        time_embedding = Dropout(drop_layers[0])(time_embedding)

        gru_vector = Concatenate(axis=1)([user_embedding, item_embedding])
        gru_vector = Reshape(target_shape=(int(gru_layers[1]), -1))(gru_vector)

        for index in range(1, len(gru_layers) - 1):
            layers = GRU(units=gru_layers[index], kernel_initializer=initializers.he_normal(),
                         kernel_regularizer=regularizers.l2(reg_layers[index]),
                         activation='tanh', recurrent_activation='hard_sigmoid', dropout=drop_layers[index],
                         return_sequences=(index != (len(gru_layers) - 2)), name='gru_layer_%d' % index)
            gru_vector = layers([gru_vector, time_embedding])

        gru_vector = Dropout(drop_layers[-1])(gru_vector)

        prediction = Dense(units=gru_layers[-1], activation='relu', kernel_initializer=initializers.lecun_normal(),
                           kernel_regularizer=regularizers.l2(reg_layers[-1]), name='gru_prediction')(gru_vector)
        _model = Model(inputs=[user_input, item_input, time_input], outputs=prediction)
        return _model

    def hybrid_loss(self, y_true, y_pred, delta=0.5):
        l1 = K.abs(y_true - y_pred)
        l2 = K.square(y_true - y_pred)
        hybrid_loss = delta * l1 + (1 - delta) * l2
        return hybrid_loss

    def save_model(self, _model):
        _model.save_weights(self.modelPath + 'pgru_%s_%.2f_%s.h5'
                            % (self.dataType, self.density, self.gruLayers), overwrite=True)


if __name__ == '__main__':
    main()
