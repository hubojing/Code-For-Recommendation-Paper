import numpy as np
import random

# 数据集切分
def loadData(fileName, ratio):    # ratio，训练集与测试集比列
    trainingData = []
    testData = []
    with open(fileName) as txtData:
        lines = txtData.readlines()
        for line in lines:
            lineData = line.strip().split(',')
            if random.random() < ratio:           #数据集分割比例
                trainingData.append(lineData)     #训练数据集列表
            else:
                testData.append(lineData)         #测试数据集列表
            np.savetxt('diabetes_train.txt', trainingData, delimiter=',',fmt='%s')
            np.savetxt('diabetes_test.txt', testData, delimiter=',',fmt='%s')
    return trainingData, testData

diabetes_file = 'diabetes.csv'
trainingData, testData = loadData(diabetes_file, 0.8)

print(trainingData)
print(testData)

# FM二分类
# diabetes皮马人糖尿病数据集
# coding:UTF-8

from numpy import *
from random import normalvariate  # 正态分布
from datetime import datetime
import pandas as pd
import numpy as np


# 处理数据
def preprocessData(data):
    feature = np.array(data.iloc[:, :-1])  # 取特征(8个特征)
    label = data.iloc[:, -1].map(lambda x: 1 if x == 1 else -1)  # 取标签并转化为 +1，-1

    # 将数组按行进行归一化
    zmax, zmin = feature.max(axis=0), feature.min(axis=0)  # 特征的最大值，特征的最小值
    feature = (feature - zmin) / (zmax - zmin)
    label = np.array(label)

    return feature, label


def sigmoid(inx):
    return 1.0 / (1 + np.exp(-inx))


# 训练FM模型
def FM(dataMatrix, classLabels, k, iter, alpha):
    '''
    :param dataMatrix:  特征矩阵
    :param classLabels: 标签矩阵
    :param k:           v的维数
    :param iter:        迭代次数
    :return:            常数项w_0, 一阶特征系数w, 二阶交叉特征系数v
    '''
    # dataMatrix用的是matrix, classLabels是列表
    m, n = shape(dataMatrix)  # 矩阵的行列数，即样本数m和特征数n

    # 初始化参数
    w = zeros((n, 1))  # 一阶特征的系数
    w_0 = 0  # 常数项
    v = normalvariate(0, 0.2) * ones((n, k))  # 即生成辅助向量(n*k)，用来训练二阶交叉特征的系数

    for it in range(iter):
        for x in range(m):  # 随机优化，每次只使用一个样本
            # 二阶项的计算
            inter_1 = dataMatrix[x] * v  # 每个样本(1*n)x(n*k),得到k维向量（FM化简公式大括号内的第一项）
            inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)  # 二阶交叉项计算，得到k维向量（FM化简公式大括号内的第二项）
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.  # 二阶交叉项计算完成（FM化简公式的大括号外累加）

            p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出，即FM的全部项之和
            tmp = 1 - sigmoid(classLabels[x] * p[0, 0])  # tmp迭代公式的中间变量，便于计算
            w_0 = w_0 + alpha * tmp * classLabels[x]

            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] + alpha * tmp * classLabels[x] * dataMatrix[x, i]
                    for j in range(k):
                        v[i, j] = v[i, j] + alpha * tmp * classLabels[x] * (
                                dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])

        # 计算损失函数的值
        if it % 10 == 0:
            loss = getLoss(getPrediction(mat(dataMatrix), w_0, w, v), classLabels)
            print("第{}次迭代后的损失为{}".format(it, loss))

    return w_0, w, v


# 损失函数
def getLoss(predict, classLabels):
    m = len(predict)
    loss = 0.0
    for i in range(m):
        loss -= log(sigmoid(predict[i] * classLabels[i]))
    return loss


# 预测
def getPrediction(dataMatrix, w_0, w, v):
    m = np.shape(dataMatrix)[0]
    result = []
    for x in range(m):
        inter_1 = dataMatrix[x] * v
        inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)  # multiply对应元素相乘
        # 完成交叉项
        interaction = np.sum(multiply(inter_1, inter_1) - inter_2) / 2.
        p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出
        pre = sigmoid(p[0, 0])
        result.append(pre)
    return result


# 评估预测的准确性
def getAccuracy(predict, classLabels):
    m = len(predict)
    allItem = 0
    error = 0
    for i in range(m):  # 计算每一个样本的误差
        allItem += 1
        if float(predict[i]) < 0.5 and classLabels[i] == 1.0:
            error += 1
        elif float(predict[i]) >= 0.5 and classLabels[i] == -1.0:
            error += 1
        else:
            continue

    return float(error) / allItem


if __name__ == '__main__':
    trainData = 'diabetes_train.txt'
    testData = 'diabetes_test.txt'
    train = pd.read_csv(trainData)
    test = pd.read_csv(testData)
    dataTrain, labelTrain = preprocessData(train)
    dataTest, labelTest = preprocessData(test)
    date_startTrain = datetime.now()

    print("开始训练")
    w_0, w, v = FM(mat(dataTrain), labelTrain, 4, 1000, 0.03)
    print("w_0:", w_0)
    print("w:", w)
    print("v:", v)
    predict_train_result = getPrediction(mat(dataTrain), w_0, w, v)  # 得到训练的准确性
    print("训练准确性为：%f" % (1 - getAccuracy(predict_train_result, labelTrain)))
    date_endTrain = datetime.now()
    print("训练用时为：%s" % (date_endTrain - date_startTrain))

    print("开始测试")
    predict_test_result = getPrediction(mat(dataTest), w_0, w, v)  # 得到训练的准确性
    print("测试准确性为：%f" % (1 - getAccuracy(predict_test_result, labelTest)))