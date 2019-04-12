"""
Created on 2018/10/9 by Chun-hui Yin(yinchunhui.ahu@gmail.com).
Description: Script file for running our experiments on response-time QoS data.
"""
import argparse
import multiprocessing
import os
import sys

import numpy as np
from keras import initializers
from keras import regularizers
from keras.layers import Embedding, Input, Dense, Flatten, Add, Dot, K, Dropout
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
    parser.add_argument('--density', default=list(np.arange(0.05, 0.21, 0.05)), type=list, help='Density of matrix.')
    parser.add_argument('--gtfLayers', default=[1024, 1], type=list, help='Layers of GTF.')
    parser.add_argument('--regLayers', default=[0., 0.], type=list, help='Regularization.')
    parser.add_argument('--dropLayers', default=[5e-1, 5e-1], type=list, help='Dropout.')
    parser.add_argument('--epochNum', default=50, type=int, help='Numbers of epochs per run.')
    parser.add_argument('--batchSize', default=2048, type=int, help='Size of a batch.')
    parser.add_argument('--optimizer', default=Adam, type=str, help='The optimizer:Adam|Adamax|Nadam|Adagrad.')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate of the model.')
    parser.add_argument('--decay', default=0., type=float, help='Decay ratio for lr.')
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
            pool.apply_async(GTF, (args, density))
        pool.close()
        pool.join()
    else:
        for density in args.density:
            model = GTF(args, density)
            del model


class GTF:

    def __init__(self, args, density):

        self.dataSet = DataSet(args, density)
        self.dataType = self.dataSet.dataType
        self.density = self.dataSet.density
        self.shape = self.dataSet.shape

        self.train = self.dataSet.train
        self.test = self.dataSet.test

        self.epochNum = args.epochNum
        self.batchSize = args.batchSize
        self.gtfLayers = args.gtfLayers
        self.regLayers = args.regLayers
        self.dropLayers = args.dropLayers
        self.lr = args.lr
        self.decay = args.decay
        self.optimizer = args.optimizer
        self.verbose = args.verbose

        self.store = args.store
        self.imagePath = args.imagePath
        self.modelPath = args.modelPath
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

            x_train, y_train = self.dataSet.getTrainInstance(self.train)
            # Training
            history = self.model.fit(x_train, y_train, batch_size=self.batchSize, epochs=1, verbose=0, shuffle=True)
            sys.stdout.write('\rEpoch %d ends.')
            # Evaluation
            if epoch % self.verbose == 0:
                sys.stdout.write('\rEvaluating Epoch %d...' % epoch)
                mae, rmse = evaluate(self.model, x_test, y_test, self.batchSize)
                loss = history.history['loss'][0]
                sys.stdout.write('\rEvaluating done.')
                if mae < best_mae:
                    best_mae, best_rmse, best_epoch = mae, rmse, epoch
                evalResults[epoch, :] = [mae, rmse]
                saveResult('gtf', self.resultPath, self.dataType, self.density, evalResults, metrics)
                sys.stdout.write('\rEpoch %d : MAE = %.4f|RMSE = %.4f|Loss = %.4f\n' % (epoch, mae, rmse, loss))
        print('=' * 14 + 'Training Complete!' + '=' * 18)
        print('The best is at epoch %d : MAE = %.4f|RMSE = %.4f.' % (best_epoch, best_mae, best_rmse))
        if self.store:
            self.save_model(self.model)
            print('The model is stored in %s.' % self.modelPath)
            print('The result is stored in %s.' % self.resultPath)

    def load_model(self):

        _model = self.build_model(self.shape, self.gtfLayers, self.regLayers, self.dropLayers)
        _model.compile(optimizer=self.optimizer(lr=self.lr, decay=self.decay), loss=self.hybrid_loss)
        plot_model(_model, to_file=self.imagePath + 'GTF.jpg', show_shapes=True)
        return _model

    @staticmethod
    def build_model(shape, gtf_layers, reg_layers, drop_layers):

        # Embedding Layer
        user_input = Input(shape=(1,), dtype='int32', name='user_input')

        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        time_input = Input(shape=(1,), dtype='int32', name='time_input')

        user_embedding = Flatten()(Embedding(input_dim=shape[0], output_dim=gtf_layers[0], input_length=1,
                                             embeddings_initializer=initializers.random_normal(),
                                             embeddings_regularizer=regularizers.l2(reg_layers[0]),
                                             name='gtf_user_embedding')(user_input))
        item_embedding = Flatten()(Embedding(input_dim=shape[1], output_dim=gtf_layers[0], input_length=1,
                                             embeddings_initializer=initializers.random_normal(),
                                             embeddings_regularizer=regularizers.l2(reg_layers[0]),
                                             name='gtf_item_embedding')(item_input))
        time_embedding = Flatten()(Embedding(input_dim=shape[2], output_dim=gtf_layers[0], input_length=1,
                                             embeddings_initializer=initializers.random_normal(),
                                             embeddings_regularizer=regularizers.l2(reg_layers[0]),
                                             name='gtf_time_embedding')(time_input))

        user_embedding = Dropout(drop_layers[0])(user_embedding)
        item_embedding = Dropout(drop_layers[0])(item_embedding)
        time_embedding = Dropout(drop_layers[0])(time_embedding)

        us = Dot(axes=-1)([user_embedding, item_embedding])
        ut = Dot(axes=-1)([user_embedding, time_embedding])
        st = Dot(axes=-1)([item_embedding, time_embedding])

        mf_vector = Add()([us, ut, st])

        mf_vector = Dropout(drop_layers[-1])(mf_vector)

        prediction = Dense(units=gtf_layers[-1], activation='relu', use_bias=True,
                           kernel_initializer=initializers.lecun_normal(),
                           kernel_regularizer=regularizers.l2(reg_layers[-1]), name='gtf_prediction')(mf_vector)

        _model = Model(inputs=[user_input, item_input, time_input], outputs=prediction)
        return _model

    def hybrid_loss(self, y_true, y_pred, delta=0.5):
        l1 = K.abs(y_true - y_pred)
        l2 = K.square(y_true - y_pred)
        hybrid_loss = delta*l1+(1-delta)*l2
        return hybrid_loss

    def save_model(self, _model):
        _model.save_weights(self.modelPath + 'gtf_%s_%.2f_%s.h5'
                            % (self.dataType, self.density, self.gtfLayers), overwrite=True)


if __name__ == '__main__':
    main()
