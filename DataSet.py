"""
Created on 2018/10/21 by Chunhui Yin(yinchunhui.ahu@gmail.com).
Description:Loading data.

"""
import numpy as np
import pandas as pd


class DataSet(object):

    def __init__(self, args, density):

        self.dataPath, self.dataType = args.dataPath, args.dataType
        self.density, self.shape = density, args.shape
        self.train, self.test = self.getTrainTest()

    def getTrainTest(self):
        print('Loading data...')
        if self.dataType == 'rt' or self.dataType == 'tp':
            train = pd.read_csv(self.dataPath + '%s_train_%.2f.txt' % (self.dataType, self.density), sep='\t')
            test = pd.read_csv(self.dataPath + '%s_test_%.2f.txt' % (self.dataType, self.density), sep='\t')
            print("Loading data done. user=%d | item=%d | time=%d | density=%.2f | dataType=%s"
                  % (self.shape[0], self.shape[1], self.shape[2], self.density, self.dataType))
            return train, test
        else:
            raise Exception('Data type error.')

    def getTrainInstance(self, data):
        user = np.array(data.iloc[:, 0])
        item = np.array(data.iloc[:, 1])
        time = np.array(data.iloc[:, 2])
        qos = np.array(data.iloc[:, 3])
        return [user, item, time], qos

    def getTestInstance(self, data):
        user = np.array(data.iloc[:, 0])
        item = np.array(data.iloc[:, 1])
        time = np.array(data.iloc[:, 2])
        qos = np.array(data.iloc[:, 3])
        return [user, item, time], qos

