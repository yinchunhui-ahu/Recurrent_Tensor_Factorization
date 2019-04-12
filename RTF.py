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
from keras.layers import Embedding, Input, Dense, Flatten, GRU, Reshape, Concatenate, Add, Dot, K, Maximum, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers
from keras.utils import plot_model
from DataSet import DataSet
from Evaluator import evaluate, saveResult
from GTF import GTF
from PGRU import PGRU

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    parser = argparse.ArgumentParser(description="Parameter Settings")
    parser.add_argument('--dataType', default='rt', type=str, help='Type of data:rt|tp.')
    parser.add_argument('--shape', default=(142, 4500, 64), type=tuple, help='(UserNum,ItemNum,TimeNum).')
    parser.add_argument('--parallel', default=False, type=bool, help='Whether to use multi-process.')
    parser.add_argument('--density', default=list(np.arange(0.05, 0.21, 0.05)), type=list, help='Density of matrix.')
    parser.add_argument('--epochNum', default=50, type=int, help='Numbers of epochs per run.')
    parser.add_argument('--batchSize', default=2048, type=int, help='Size of a batch.')
    parser.add_argument('--gtfLayers', default=[1024, 1], type=list, help='Layers of GTF.')
    parser.add_argument('--gruLayers', default=[2048, 1, 1], type=list, help='Layers of GRU.')
    parser.add_argument('--regLayers', default=[0., 0., 0.], type=list, help='Regularization.')
    parser.add_argument('--dropLayers', default=[5e-1, 5e-1, 5e-1], type=list, help='Dropout.')
    parser.add_argument('--optimizer', default=Adam, type=str, help='The optimizer:Adam|Adamax|Nadam|Adagrad.')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate of the model.')
    parser.add_argument('--decay', default=0.00, type=float, help='Decay ratio for lr.')
    parser.add_argument('--verbose', default=1, type=int, help='Iterations per evaluation.')
    parser.add_argument('--preTraining', default=False, type=bool, help='Whether to load the pre-training model.')
    parser.add_argument('--store', default=True, type=bool, help='Whether to store the model and result.')
    parser.add_argument('--dataPath', default='./Data/dataset#2/', type=str, help='Path to load data.')
    parser.add_argument('--modelPath', default='./Model/', type=str, help='Path to save the model.')
    parser.add_argument('--imagePath', default='./Image/', type=str, help='Path to save the image.')
    parser.add_argument('--resultPath', default='./Result/', type=str, help='Path to save the result.')
    args = parser.parse_args()

    if args.parallel:
        pool = multiprocessing.Pool()
        for density in args.density:
            pool.apply_async(RTF, (args, density))
        pool.close()
        pool.join()
    else:
        for density in args.density:
            model = RTF(args, density)
            del model


class RTF:

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
        self.gtfLayers = args.gtfLayers
        self.regLayers = args.regLayers
        self.dropLayers = args.dropLayers
        self.lr = args.lr
        self.decay = args.decay
        self.optimizer = args.optimizer
        self.verbose = args.verbose

        self.preTraining = args.preTraining
        self.store = args.store
        self.modelPath = args.modelPath
        self.imagePath = args.imagePath
        self.resultPath = args.resultPath

        self.model = self.load_model()

        self.run()

    def run(self):

        # Initializing model
        x_test, y_test = self.dataSet.getTestInstance(self.test)
        print('Initializing...')
        mae, rmse = evaluate(self.model, x_test, y_test, self.batchSize)
        best_mae, best_rmse, best_epoch = mae, rmse, -1
        metrics = ['MAE', 'RMSE', 'Loss']
        evalResults = np.zeros((self.epochNum, len(metrics)))
        print('Initializing done.MAE = %.4f|RMSE = %.4f.' % (mae, rmse))
        # Training model
        print('=' * 25 + 'Start Training' + '=' * 30)
        for epoch in range(self.epochNum):
            sys.stdout.write('\rEpoch %d starts...' % epoch)
            x_train, y_train = self.dataSet.getTrainInstance(self.train)
            # Training
            history = self.model.fit(x_train, y_train, batch_size=self.batchSize, epochs=1, verbose=0, shuffle=True)
            # , callbacks=[TensorBoard(log_dir='./Log')])
            sys.stdout.write('\rEpoch %d ends.' % epoch)
            # Evaluation
            if epoch % self.verbose == 0:
                sys.stdout.write('\rEvaluating Epoch %d...' % epoch)
                mae, rmse = evaluate(self.model, x_test, y_test, self.batchSize)
                loss = history.history['loss'][0]
                evalResults[epoch, :] = [mae, rmse, loss]
                sys.stdout.write('\rEvaluating done.')
                if mae < best_mae:
                    best_mae, best_rmse, best_epoch = mae, rmse, epoch
                saveResult('rtf', self.resultPath, self.dataType, self.density, evalResults, metrics)
                sys.stdout.write('\rEpoch %d : MAE = %.4f|RMSE = %.4f|Loss = %.4f \n' % (epoch, mae, rmse, loss))
        print('=' * 23 + 'Training Complete!' + '=' * 28)
        print('The best is at epoch %d : MAE = %.4f RMSE = %.4f.' % (best_epoch, best_mae, best_rmse))
        if self.store:
            self.save_model(self.model)
            print('The model is stored in %s.' % self.modelPath)
            print('The result is stored in %s.' % self.resultPath)

    def load_model(self):

        rtf_model = self.build_model(self.shape, self.gruLayers, self.gtfLayers, self.regLayers, self.dropLayers)

        if self.preTraining:
            gtf_model = GTF.build_model(self.shape, self.gtfLayers, self.regLayers, self.dropLayers)
            gtf_model.load_weights(
                self.modelPath + 'gtf_%s_%.2f_%s.h5' % (self.dataType, self.density, self.gtfLayers))

            gru_model = PGRU.build_model(self.shape, self.gruLayers, self.regLayers, self.dropLayers)
            gru_model.load_weights(
                self.modelPath + 'pgru_%s_%.2f_%s.h5' % (self.dataType, self.density, self.gruLayers))

            rtf_model = self.load_pretrain_model(rtf_model, gtf_model, gru_model, self.gruLayers)

        rtf_model.compile(optimizer=self.optimizer(lr=self.lr, decay=self.decay), loss=self.hybrid_loss)
        plot_model(rtf_model, to_file=self.imagePath + 'RTF.jpg', show_shapes=True)
        return rtf_model

    def build_model(self, shape, gru_layers, gtf_layers, reg_layers, drop_layers):

        # Granulation
        user_input = Input(shape=(1,), dtype='int32', name='user_input')

        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        time_input = Input(shape=(1,), dtype='int32', name='time_input')

        # GTF

        tf_user_embedding = Flatten()(Embedding(input_dim=shape[0], output_dim=gtf_layers[0],
                                                embeddings_initializer=initializers.random_normal(),
                                                embeddings_regularizer=regularizers.l2(reg_layers[0]),
                                                input_length=1, name='gtf_user_embedding')(user_input))
        tf_item_embedding = Flatten()(Embedding(input_dim=shape[1], output_dim=gtf_layers[0],
                                                embeddings_initializer=initializers.random_normal(),
                                                embeddings_regularizer=regularizers.l2(reg_layers[0]),
                                                input_length=1, name='gtf_item_embedding')(item_input))
        tf_time_embedding = Flatten()(Embedding(input_dim=shape[2], output_dim=gtf_layers[0],
                                                embeddings_initializer=initializers.random_normal(),
                                                embeddings_regularizer=regularizers.l2(reg_layers[0]),
                                                input_length=1, name='gtf_time_embedding')(time_input))

        tf_user_embedding = Dropout(drop_layers[0])(tf_user_embedding,)
        tf_item_embedding = Dropout(drop_layers[0])(tf_item_embedding)
        tf_time_embedding = Dropout(drop_layers[0])(tf_time_embedding)

        us = Dot(axes=-1)([tf_user_embedding, tf_item_embedding])

        ut = Dot(axes=-1)([tf_user_embedding, tf_time_embedding])

        st = Dot(axes=-1)([tf_item_embedding, tf_time_embedding])

        gtf_vector = Add()([us, ut, st])

        gtf_prediction = Dense(units=gtf_layers[-1], activation='relu', kernel_initializer=initializers.lecun_normal(),
                               kernel_regularizer=regularizers.l2(reg_layers[-1]),
                               name='gtf_prediction')(gtf_vector)

        # PGRU

        gru_user_embedding = Flatten()(Embedding(input_dim=shape[0], output_dim=gru_layers[0],
                                                 embeddings_initializer=initializers.random_normal(),
                                                 embeddings_regularizer=regularizers.l2(reg_layers[0]),
                                                 input_length=1, name='gru_user_embedding')(user_input))
        gru_item_embedding = Flatten()(Embedding(input_dim=shape[1], output_dim=gru_layers[0],
                                                 embeddings_initializer=initializers.random_normal(),
                                                 embeddings_regularizer=regularizers.l2(reg_layers[0]),
                                                 input_length=1, name='gru_item_embedding')(item_input))
        gru_time_embedding = Flatten()(Embedding(input_dim=shape[2], output_dim=gru_layers[1],
                                                 embeddings_initializer=initializers.random_normal(),
                                                 embeddings_regularizer=regularizers.l2(reg_layers[0]),
                                                 input_length=1, name='gru_time_embedding')(time_input))

        gru_user_embedding = Dropout(drop_layers[0])(gru_user_embedding)
        gru_item_embedding = Dropout(drop_layers[0])(gru_item_embedding)
        gru_time_embedding = Dropout(drop_layers[0])(gru_time_embedding)

        gru_vector = Concatenate(axis=-1)([gru_user_embedding, gru_item_embedding])

        gru_vector = Reshape(target_shape=(int(gru_layers[1]), -1))(gru_vector)

        for index in range(1, len(gru_layers) - 1):

            layers = GRU(units=gru_layers[index], kernel_initializer=initializers.he_uniform(),
                         kernel_regularizer =regularizers.l2(reg_layers[index]),
                         activation='tanh', recurrent_activation='hard_sigmoid', dropout=drop_layers[index],
                         return_sequences=(index != (len(gru_layers) - 2)), name='gru_layer_%d' % index)
            gru_vector = layers([gru_vector, gru_time_embedding])

        gru_vector = Dropout(drop_layers[-1])(gru_vector)

        gru_prediction = Dense(units=gru_layers[-1], activation='relu', kernel_initializer=initializers.lecun_normal(),
                               kernel_regularizer=regularizers.l2(reg_layers[-1]), name='gru_prediction')(gru_vector)

        rtf_prediction = Maximum(name='rtf_prediction')([gru_prediction, gtf_prediction])

        _model = Model(inputs=[user_input, item_input, time_input], outputs=rtf_prediction)
        return _model

    def load_pretrain_model(self, rtf_model, gtf_model, gru_model, gru_layers):

        print("Loading pre-training models...")

        # GTF
        gtf_user_embedding_weight = gtf_model.get_layer('gtf_user_embedding').get_weights()
        gtf_item_embedding_weight = gtf_model.get_layer('gtf_item_embedding').get_weights()
        gtf_time_embedding_weight = gtf_model.get_layer('gtf_time_embedding').get_weights()
        gtf_prediction_weight = gtf_model.get_layer('gtf_prediction').get_weights()

        rtf_model.get_layer('gtf_user_embedding').set_weights(gtf_user_embedding_weight)
        rtf_model.get_layer('gtf_item_embedding').set_weights(gtf_item_embedding_weight)
        rtf_model.get_layer('gtf_time_embedding').set_weights(gtf_time_embedding_weight)
        rtf_model.get_layer('gtf_prediction').set_weights(gtf_prediction_weight)

        # GRU
        gru_user_embedding_weight = gru_model.get_layer('gru_user_embedding').get_weights()
        gru_item_embedding_weight = gru_model.get_layer('gru_item_embedding').get_weights()
        gru_time_embedding_weight = gru_model.get_layer('gru_time_embedding').get_weights()
        rtf_model.get_layer('gru_user_embedding').set_weights(gru_user_embedding_weight)
        rtf_model.get_layer('gru_item_embedding').set_weights(gru_item_embedding_weight)
        rtf_model.get_layer('gru_time_embedding').set_weights(gru_time_embedding_weight)

        for index in range(1, len(gru_layers) - 1):

            gru_layer_weights = gru_model.get_layer('gru_layer_%d' % index).get_weights()
            rtf_model.get_layer('gru_layer_%d' % index).set_weights(gru_layer_weights)

        gru_prediction_weight = gru_model.get_layer('gru_prediction').get_weights()
        rtf_model.get_layer('gru_prediction').set_weights(gru_prediction_weight)

        print("Loading pre-training models done.")
        return rtf_model

    def hybrid_loss(self, y_true, y_pred, delta=0.5):
        l1 = K.abs(y_true - y_pred)
        l2 = K.square(y_true - y_pred)
        hybrid_loss = delta * l1 + (1 - delta) * l2
        return hybrid_loss

    def save_model(self, _model):

        _model.save_weights(self.modelPath + 'rtf_%s_%.2f.h5' % (self.dataType, self.density), overwrite=True)


if __name__ == '__main__':
    main()
