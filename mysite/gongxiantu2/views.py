import random
import time
from urllib import request
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import datetime
import threading
# ________________________
import numpy as np
import pandas as pd
import random
# from datapreprocessing import z_score
# from SFA import dongtaikuozhan
from scipy import linalg
from scipy.stats import chi2, f, norm, gaussian_kde
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
import time
import os
path=os.getcwd()
print(path)
from lib import mysql_df
# Create your views here.
sns.set()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False

global num  # 现在时刻计算T2值
global prenum  # 上一次计算的T2值
num = 0  # 初始化
prenum = 0  # 初始化
global now_time  # 当前时间
global a  # 变量编号
global major_variable  # 锡槽过程监控的过程变量
global ini_timeList1  # 前端T2统计量时间序列初始化
global ini_timeList2  # 前端动态贡献图时间序列初始化
global ini_num  # 前端T2统计量初始化
global ini_t2gx  # 前端T2贡献值初始化

ini_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ini_t2gx = [
    [0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0], [0, 4, 0], [0, 5, 0], [0, 6, 0], [0, 7, 0], [0, 8, 0], [0, 9, 0],
    [0, 10, 0], [0, 11, 0], [0, 12, 0], [0, 13, 0], [0, 14, 0], [0, 15, 0], [0, 16, 0], [0, 17, 0], [0, 18, 0],
    [0, 19, 0], [0, 20, 0], [0, 21, 0], [0, 22, 0], [0, 23, 0], [0, 24, 0], [0, 25, 0], [0, 26, 0], [0, 27, 0],
    [0, 28, 0], [0, 29, 0], [0, 30, 0], [0, 31, 0], [0, 32, 0],
    [1, 0, 0], [1, 1, 0], [1, 2, 0], [1, 3, 0], [1, 4, 0], [1, 5, 0], [1, 6, 0], [1, 7, 0], [1, 8, 0], [1, 9, 0],
    [1, 10, 0], [1, 11, 0], [1, 12, 0], [1, 13, 0], [1, 14, 0], [1, 15, 0], [1, 16, 0], [1, 17, 0], [1, 18, 0],
    [1, 19, 0], [1, 20, 0], [1, 21, 0], [1, 22, 0], [1, 23, 0], [1, 24, 0], [1, 25, 0], [1, 26, 0], [1, 27, 0],
    [1, 28, 0], [1, 29, 0], [1, 30, 0], [1, 31, 0], [1, 32, 0],
    [2, 0, 0], [2, 1, 0], [2, 2, 0], [2, 3, 0], [2, 4, 0], [2, 5, 0], [2, 6, 0], [2, 7, 0], [2, 8, 0], [2, 9, 0],
    [2, 10, 0], [2, 11, 0], [2, 12, 0], [2, 13, 0], [2, 14, 0], [2, 15, 0], [2, 16, 0], [2, 17, 0], [2, 18, 0],
    [2, 19, 0], [2, 20, 0], [2, 21, 0], [2, 22, 0], [2, 23, 0], [2, 24, 0], [2, 25, 0], [2, 26, 0], [2, 27, 0],
    [2, 28, 0], [2, 29, 0], [2, 30, 0], [2, 31, 0], [2, 32, 0],
    [3, 0, 0], [3, 1, 0], [3, 2, 0], [3, 3, 0], [3, 4, 0], [3, 5, 0], [3, 6, 0], [3, 7, 0], [3, 8, 0], [3, 9, 0],
    [3, 10, 0], [3, 11, 0], [3, 12, 0], [3, 13, 0], [3, 14, 0], [3, 15, 0], [3, 16, 0], [3, 17, 0], [3, 18, 0],
    [3, 19, 0], [3, 20, 0], [3, 21, 0], [3, 22, 0], [3, 23, 0], [3, 24, 0], [3, 25, 0], [3, 26, 0], [3, 27, 0],
    [3, 28, 0], [3, 29, 0], [3, 30, 0], [3, 31, 0], [3, 32, 0],
    [4, 0, 0], [4, 1, 0], [4, 2, 0], [4, 3, 0], [4, 4, 0], [4, 5, 0], [4, 6, 0], [4, 7, 0], [4, 8, 0], [4, 9, 0],
    [4, 10, 0], [4, 11, 0], [4, 12, 0], [4, 13, 0], [4, 14, 0], [4, 15, 0], [4, 16, 0], [4, 17, 0], [4, 18, 0],
    [4, 19, 0], [4, 20, 0], [4, 21, 0], [4, 22, 0], [4, 23, 0], [4, 24, 0], [4, 25, 0], [4, 26, 0], [4, 27, 0],
    [4, 28, 0], [4, 29, 0], [4, 30, 0], [4, 31, 0], [4, 32, 0],
    [5, 0, 0], [5, 1, 0], [5, 2, 0], [5, 3, 0], [5, 4, 0], [5, 5, 0], [5, 6, 0], [5, 7, 0], [5, 8, 0], [5, 9, 0],
    [5, 10, 0], [5, 11, 0], [5, 12, 0], [5, 13, 0], [5, 14, 0], [5, 15, 0], [5, 16, 0], [5, 17, 0], [5, 18, 0],
    [5, 19, 0], [5, 20, 0], [5, 21, 0], [5, 22, 0], [5, 23, 0], [5, 24, 0], [5, 25, 0], [5, 26, 0], [5, 27, 0],
    [5, 28, 0], [5, 29, 0], [5, 30, 0], [5, 31, 0], [5, 32, 0],
    [6, 0, 0], [6, 1, 0], [6, 2, 0], [6, 3, 0], [6, 4, 0], [6, 5, 0], [6, 6, 0], [6, 7, 0], [6, 8, 0], [6, 9, 0],
    [6, 10, 0], [6, 11, 0], [6, 12, 0], [6, 13, 0], [6, 14, 0], [6, 15, 0], [6, 16, 0], [6, 17, 0], [6, 18, 0],
    [6, 19, 0], [6, 20, 0], [6, 21, 0], [6, 22, 0], [6, 23, 0], [6, 24, 0], [6, 25, 0], [6, 26, 0], [6, 27, 0],
    [6, 28, 0], [6, 29, 0], [6, 30, 0], [6, 31, 0], [6, 32, 0],
    [7, 0, 0], [7, 1, 0], [7, 2, 0], [7, 3, 0], [7, 4, 0], [7, 5, 0], [7, 6, 0], [7, 7, 0], [7, 8, 0], [7, 9, 0],
    [7, 10, 0], [7, 11, 0], [7, 12, 0], [7, 13, 0], [7, 14, 0], [7, 15, 0], [7, 16, 0], [7, 17, 0], [7, 18, 0],
    [7, 19, 0], [7, 20, 0], [7, 21, 0], [7, 22, 0], [7, 23, 0], [7, 24, 0], [7, 25, 0], [7, 26, 0], [7, 27, 0],
    [7, 28, 0], [7, 29, 0], [7, 30, 0], [7, 31, 0], [7, 32, 0],
    [8, 0, 0], [8, 1, 0], [8, 2, 0], [8, 3, 0], [8, 4, 0], [8, 5, 0], [8, 6, 0], [8, 7, 0], [8, 8, 0], [8, 9, 0],
    [8, 10, 0], [8, 11, 0], [8, 12, 0], [8, 13, 0], [8, 14, 0], [8, 15, 0], [8, 16, 0], [8, 17, 0], [8, 18, 0],
    [8, 19, 0], [8, 20, 0], [8, 21, 0], [8, 22, 0], [8, 23, 0], [8, 24, 0], [8, 25, 0], [8, 26, 0], [8, 27, 0],
    [8, 28, 0], [8, 29, 0], [8, 30, 0], [8, 31, 0], [8, 32, 0],
    [9, 0, 0], [9, 1, 0], [9, 2, 0], [9, 3, 0], [9, 4, 0], [9, 5, 0], [9, 6, 0], [9, 7, 0], [9, 8, 0], [9, 9, 0],
    [9, 10, 0], [9, 11, 0], [9, 12, 0], [9, 13, 0], [9, 14, 0], [9, 15, 0], [9, 16, 0], [9, 17, 0], [9, 18, 0],
    [9, 19, 0], [9, 20, 0], [9, 21, 0], [9, 22, 0], [9, 23, 0], [9, 24, 0], [9, 25, 0], [9, 26, 0], [9, 27, 0],
    [9, 28, 0], [9, 29, 0], [9, 30, 0], [9, 31, 0], [9, 32, 0],
]
ini_timeList1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ini_timeList2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def z_score(xtrain, xtest):
    xr, xl = np.shape(xtrain)
    tr, tl = np.shape(xtest)
    xtrain_mean = np.mean(xtrain, axis=0)
    xtrain_std = np.std(xtrain, axis=0)
    # xtrain_mean = xtrain_mean.reshape(1, -1)
    xtrain = (xtrain - np.tile(xtrain_mean, (xr, 1))) / np.tile(xtrain_std, (xr, 1))
    xtest = (xtest - np.tile(xtrain_mean, (tr, 1))) / np.tile(xtrain_std, (tr, 1))
    return xtrain, xtest


class PCA:
    def __init__(self, xtrain, xtest, FN):
        self._xtrain = xtrain
        self._xtest = xtest
        self._PCs = []
        self._eigenVector = []
        self.n_tezheng = 0
        self.T2 = []
        self.Q = []
        self.Qucl = 0
        self.T2ucl = 0
        self.FN = FN

    def pcafit_transform(self, kdelim=False, contrfig=False, rconfig=False):
        global a
        global ini_t2gx
        global ini_num
        xtrain = self._xtrain
        xtest = self._xtest
        # 求协方差矩阵
        sigmaxtrain = np.cov(xtrain, rowvar=False)
        # 对协方差矩阵进行特征分解，self._PCs为特征值构成的对角阵，T的列为单位特征向量，且与self._PCs中的特征值一一对应：
        self._PCs, self._eigenVector = linalg.eig(sigmaxtrain)

        # 取对角元素(结果为一列向量)，即self._PCs值，并上下反转使其从大到小排列，主元个数初值为1，若累计贡献率小于85%则增加主元个数
        self._eigenVector = self._eigenVector[:, self._PCs.argsort()]  # 将特征向量的列按照主特征升序排列
        lamda = self._PCs
        lamda.sort()  # 将self._PCs按升序排列
        lamda = -np.sort(-np.real(lamda))  # 按降序排列
        self.n_tezheng = 1
        while sum(lamda[0:self.n_tezheng]) / sum(lamda) < 0.90:
            self.n_tezheng = self.n_tezheng + 1

        # 取与self._PCs相对应的特征向量
        tr, tl = xtest.shape
        P = np.mat(self._eigenVector[:, tl - self.n_tezheng:tl])
        # 求置信度为99%时的T2统计控制限
        self.T2ucl = self.n_tezheng * (tr - 1) * (tr + 1) * f.ppf(0.99, self.n_tezheng, tr - self.n_tezheng) / (
                tr * (tr - self.n_tezheng))

        # 置信度为99%的Q统计控制限
        theta = []
        for i in range(1, self.n_tezheng + 1):
            theta.append(sum((lamda[self.n_tezheng:]) ** i))
        h0 = 1 - 2 * theta[0] * theta[2] / (3 * theta[1] ** 2)
        ca = norm.ppf(0.99, 0, 1)
        self.Qucl = theta[0] * (
                h0 * ca * np.sqrt(2 * theta[1]) / theta[0] + 1 + theta[1] * h0 * (h0 - 1) / theta[0] ** 2) ** (
                            1 / h0)

        # 求T2统计量，Q统计量在线值
        r, y = (P.dot(P.T)).shape
        I_mat = np.mat(np.eye(r, y))
        xtest = np.array(xtest, type(float))
        xtest = np.mat(xtest)
        # for i in xtest:
        # tr, tl = np.shape(xtest)
        # i = 0
        # if i < tr:

        # 按顺序读取第a个样本，计算a个样本的T2
        # 一般样本集

        tr, tl = np.shape(xtest)
        xx = xtest[a, :]  # 第a组数据
        self.T2 = (xx * P * np.mat(np.diag(self._PCs.real[tl - self.n_tezheng:])).I * P.T * xx.T)
        T2_value = round(self.T2[0, 0], 2)  # 保留2位小数
        del ini_num[0]  # 删除最旧的T2统计量
        ini_num.append(T2_value)  # 加入最新的T2统计量
        # if self.T2[0,0] < 13.4:
        #     return self.T2[0,0]

        if (self.T2[0, 0] < self.T2ucl):  # 如果T2统计量小于控制线
            contrfig = False  # 不用计算贡献图判断逻辑
            print("没发生故障")
            t2gx = []  # 贡献值数组，可以先看贡献值计算部分

            for i in range(0, 33):
                del ini_t2gx[0]  # t2贡献图初始化数组添加0为目前贡献值
                # 可以先看贡献值计算部分，下面
            ini_t2gx = list(map(lambda x: [x[0] - 1, x[1], x[2]], ini_t2gx))
            # 因为没发生故障，所以不用计算贡献值，直接添加0和T2统计量返回前端就好
            for i in range(0, 33):
                t2gx.append([9, i, 0])
                ini_t2gx.append([9, i, 0])

            return T2_value, t2gx

        ####动态贡献图

        self.T2 = np.diag(xtest * P * np.mat(np.diag(self._PCs.real[tl - self.n_tezheng:])).I * P.T * xtest.T)
        # for i in self.T2:
        #     print(i)
        self.Q = np.diag(xtest * (I_mat - P * P.T) * (I_mat - P * P.T).T * xtest.T)
        if kdelim:
            T2train = np.diag(xtrain * P * np.mat(np.diag(self._PCs.real[tl - self.n_tezheng:])).I * P.T * xtrain.T)
            Qtrain = np.diag(xtrain * (I_mat - P * P.T) * (I_mat - P * P.T).T * xtrain.T)
            T2ucl = gaussian_kde(T2train)
            x_grid = np.linspace(0, max(T2train), 20000)
            # plt.plot(x_grid, T2ucl.evaluate(x_grid))
            xx = T2ucl.evaluate(x_grid)
            lim = 0
            for i in range(len(xx)):
                lim = lim + xx[i] / sum(xx)
                if lim > 0.99:
                    k = i
                    break
            self.T2ucl = x_grid[k]
            Qucl = gaussian_kde(Qtrain)
            x_grid = np.linspace(0, max(Qtrain), 20000)
            # plt.plot(x_grid, T2ucl.evaluate(x_grid))
            xx = Qucl.evaluate(x_grid)
            lim = 0
            for i in range(len(xx)):
                lim = lim + xx[i] / sum(xx)
                if lim > 0.99:
                    k = i
                    break
            self.Qucl = x_grid[k]

        if contrfig:
            n, m = np.shape(xtest)
            # Q
            I = np.matrix(np.eye(m))
            P = np.matrix(P)
            # contq = np.zeros(m)
            contqi = np.array((xtest[self.FN + 1, :] * (I - P * P.T)).T) * np.array(
                (xtest[self.FN + 1, :] * (I - P * P.T)).T)
            # print(contqi)

            contq = np.diag((xtest[self.FN + 1:, :] * (I - P * P.T)).T * (xtest[self.FN + 1:, :] * (I - P * P.T)))
            # Econtq = np.diag((xtrain*(I- P*P.T)).T*(xtrain*(I- P*P.T)))
            Econtq = np.diag((I - P * P.T).T * np.matrix(np.cov(xtrain, rowvar=False)) * (I - P * P.T))
            rcontq = contq / Econtq

            # T2
            # lamd = lamda[:self.n_tezheng]  # self._PCs
            # lamd = np.diag(lamd)
            P1 = self._eigenVector
            In = np.matrix(np.eye(n))
            D = P1.T * np.matrix(np.linalg.inv(np.diag(lamda))) * P1
            v, Q = np.linalg.eig(D)
            V = np.diag(v ** 0.5)
            D = Q * V * Q ** -1
            contt2i_part = np.array((xtest[self.FN + 1, :] * D).T) * np.array((xtest[self.FN + 1, :] * D).T)
            contt2i = np.array((xtest[a, :] * D).T) * np.array((xtest[a, :] * D).T)
            contt2 = np.diag((xtest[a, :] * D).T * xtest[a, :] * D)
            Econtt2 = np.diag((D.T * np.matrix(np.cov(xtrain, rowvar=False)) * D))
            rcontt2 = contt2 / Econtt2
            # print(f'第1个变量第{a}个样本对故障的T2贡献是{contt2i[0,a]}')#(33,335)
            t2gx = []  # 贡献值数组容器
            # 贡献值数组容器
            # 以锡槽动态贡献图只展示计算10组数据为例（33个变量）,前端（33*10）
            # 贡献值ini_t2gx为[[0，0，var],[0,0,var]...[9,31,var][9,32,var]],ini_t2gx.shape=(10,33)
            # 以[9,32,var]为例,[9,...]表是计算的是最新一组数据（0最旧，9最新），[...,32,...]表示变量号，[...,var]表示该变量的贡献值
            # 因此[9,32,var]表示，第九组（最新）的数据的第33个变量（编号32，因为从0开始编号）贡献值为var（eg:7652.3)
            # [9,32,var]在图中的坐标为(32,9)
            for i in range(0, 33):
                del ini_t2gx[0]  # 删除十组数据中最旧的数据
            ini_t2gx = list(map(lambda x: [x[0] - 1, x[1], x[2]], ini_t2gx))  # 所有数据的组数索引减1，0最旧，8最新
            # 一批数据33个变量，添加33次
            for i in range(0, 33):
                t2gx.append([9, i, round(contt2i[i, 0])])  # 添加最新的一组值
                ini_t2gx.append([9, i, round(contt2i[i, 0])])  # 下一次供页面刷新初始化用
            # print(ini_t2gx[0])
            # print(t2gx)
            print('故障发生')
            return T2_value, t2gx

            # print('形状',contt2i_part.shape,type(contt2i_part),contt2i_part[0,0])
            # print('形状',contt2i.shape,type(contt2i))


@csrf_exempt
def index(request):
    global prenum
    global now_time
    global t2gx
    global major_variable
    global ini_timeList1
    global ini_timeList2
    global ini_t2gx
    global ini_num
    # print('前端来了')
    if prenum != num:  # 计算结果有变化才刷新数据，否则pass
        if request.method == 'POST':
            prenum = num  # 如果
            return JsonResponse(
                {"datas": {"local_time": now_time, "numgxt": t2gx, "num": num, "major_variable": major_variable,
                           "ini_time1": ini_timeList1,
                           "ini_time2": ini_timeList2,
                           "ini_num": ini_num,
                           "ini_t2gx": ini_t2gx
                           }})
    else:
        pass
    return render(request, 'dtT2_sec.html')


def run2():
    global num
    global prenum
    global now_time
    global t2gx
    global a
    global major_variable
    global ini_timeList1
    global ini_timeList2
    global ini_t2gx
    global ini_num
    time_start = time.time()  # 函数计算时间用

    xtrain = np.array(pd.read_excel(path+'\exl\锡槽0.7平衡2.xlsx',sheet_name='train'))  # 3 6 Sheet5  锡槽0.7平衡
    xtest = np.array(pd.read_excel(path+'\exl\锡槽0.7平衡2.xlsx', sheet_name='test'))  # 4 7  smote1.1
    # xtrain = np.array(mysql_df.read_df('train'),dtype='float64')  # 3 6 Sheet5  锡槽0.7平衡
    # xtest = np.array(mysql_df.read_df('test'),dtype='float64')  # 4 7  smote1.1

    n, m = np.shape(xtrain)
    xtrain = xtrain[:, :m - 2]
    xtest = xtest[:, :m - 2]
    f_out = open(path+'/exl/num_xicao.txt', 'r+')
    a = f_out.read()
    a = int(a) + 1
    print("现在的变量：", a)
    f_out.seek(0)
    f_out.truncate()
    f_out.write(str(a))
    f_out.close()
    tr, ts = np.shape(xtest)
    if a == tr:  # 如果数据读取完了，再令txt中a为0
        f_out = open(path+'/exl/num_xicao.txt', 'r+')
        a = f_out.read()
        a = 0
        print("现在的变量：", a)
        f_out.seek(0)
        f_out.truncate()
        f_out.write(str(a))
        f_out.close()

    major_variable = [round(xtest[a, 0], 2), round(xtest[a, 3], 2), round(xtest[a, 4], 2), round(xtest[a, 23], 2)]
    # major_variable = print('主传动速度是：',xtest[a,0],'m/min','冷端速度是：', xtest[a,3], 'm/min','流道热偶温度是：',xtest[a,4], '摄氏度','一号挡帘高度是：',xtest[a,23],'mm')
    xtrain, xtest = z_score(xtrain, xtest)
    pca = PCA(xtrain, xtest, 0)
    num, t2gx = pca.pcafit_transform(kdelim=False, contrfig=True)  # 计算结果
    # 横坐标
    del ini_timeList1[0]  # 横坐标更新删去最早的时间
    del ini_timeList2[0]  # 横坐标更新
    now_time = time.strftime("%H:%M:%S")
    ini_timeList1.append(now_time)  # 横坐标更新加入最新的时间
    ini_timeList2.append(now_time)
    if num != prenum:  # 如果现在的T2统计量和上一次的不一样，即num,t2gx = pca.pcafit_transform(kdelim=False, contrfig=True)计算出新的结果
        threading.Timer(1, run2).start()  # 再次调用run2函数，进行下一次计算
    else:
        print("!!!!!!!!!!!!!还未计算完成", prenum, num)  # 否则提示，再次调用run2
        run2
    time_end = time.time()
    print('time cost', time_end - time_start, 's')


timer = threading.Timer(1, run2)  # 延迟1s运行timer
timer.start()  # 执行方法

# def run():
#     global num
#     global now_time
#     now_time = time.strftime("%H:%M:%S")
#     num = random.randint(1, 100)
#     threading.Timer(1,run).start()
#     print("后端自动运行：",num,'----',now_time,type(num))
