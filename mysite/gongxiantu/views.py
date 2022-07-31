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
# # from SFA import dongtaikuozhan
from scipy import linalg
from scipy.stats import chi2, f, norm, gaussian_kde
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator
import seaborn as sns
import time
from lib import mysql_df
# Create your views here.

sns.set()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False

global num
global prenum
num = 0
prenum = 0
global now_time
global a
global major_variable


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
        xx = xtest[a, :]
        self.T2 = (xx * P * np.mat(np.diag(self._PCs.real[tl - self.n_tezheng:])).I * P.T * xx.T)
        T2_value = round(self.T2[0, 0], 2)
        # if self.T2[0,0] < 13.4:
        #     return self.T2[0,0]
        if (self.T2[0, 0] < self.T2ucl):
            contrfig = False
            print("没发生故障")
            t2gx = []
            for i in range(0, 33):
                t2gx.append([9, i, 0])
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

        # # 计算监测指标
        # FART2 = 0
        # FARQ = 0
        # FDRT2 = 0
        # FDRQ = 0
        # for i in range(self.FN):
        #     if self.T2[i] > self.T2ucl:
        #         FART2 = FART2 + 1
        #     if self.Q[i] > self.Qucl:
        #         FARQ = FARQ + 1
        # self.FART2 = FART2 / self.FN
        # self.FARQ = FARQ / self.FN

        # for i in range(self.FN, tr):
        #     if self.T2[i] > self.T2ucl:
        #         FDRT2 = FDRT2 + 1
        #     if self.Q[i] > self.Qucl:
        #         FDRQ = FDRQ + 1
        # self.FDRT2 = FDRT2 / (tr - self.FN)
        # self.FDRQ = FDRQ / (tr - self.FN)

        # print(f'FART2={self.FART2},FAR_SPE={self.FARQ}\nFDRT2={self.FDRT2},FDRQ={self.FDRQ}')
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
            t2gx = []
            for i in range(0, 19):
                t2gx.append([9, i, round(contt2i[i, 0])])
            # print(t2gx)
            print('故障发生')
            return T2_value, t2gx

            # print('形状',contt2i_part.shape,type(contt2i_part),contt2i_part[0,0])
            # print('形状',contt2i.shape,type(contt2i))


def run2():
    global num
    global prenum
    global now_time
    global t2gx
    global a
    global major_variable
    time_start = time.time()
    # xtrain = np.array(pd.read_excel(r'/Users/lizhexi/Desktop/玻璃厂项目/PCA/PCA程序/锡槽0.7平衡2.xlsx', sheet_name='train'))  # 3 6 Sheet5  锡槽0.7平衡
    # xtest = np.array(pd.read_excel(r'/Users/lizhexi/Desktop/玻璃厂项目/PCA/PCA程序/锡槽0.7平衡2.xlsx', sheet_name='test'))  # 4 7  smote1.1
    xtrain = np.array(
        pd.read_excel(r'/project/mysite/static/退火变量筛选后数据.xlsx', sheet_name='train2'))  # 3 6 Sheet5  锡槽0.7平衡
    xtest = np.array(pd.read_excel(r'/project/mysite/static/退火变量筛选后数据.xlsx', sheet_name='test2'))  # 4 7  smote1.1
    n, m = np.shape(xtrain)
    xtrain = xtrain[:, :m - 2]
    xtest = xtest[:, :m - 2]
    f_out = open('/project/mysite/static/num.txt', 'r+')
    a = f_out.read()
    a = int(a) + 1
    print("现在的变量：", a)
    f_out.seek(0)
    f_out.truncate()
    f_out.write(str(a))
    f_out.close()
    tr, ts = np.shape(xtest)
    if a == tr:
        f_out = open('/project/mysite/static/num.txt', 'r+')
        a = f_out.read()
        a = 0
        print("现在的变量：", a)
        f_out.seek(0)
        f_out.truncate()
        f_out.write(str(a))
        f_out.close()
    major_variable = [round(xtest[a, 0], 2), round(xtest[a, 3], 2), round(xtest[a, 4], 2), round(xtest[a, 19], 2)]
    # major_variable = print('主传动速度是：',xtest[a,0],'m/min','冷端速度是：', xtest[a,3], 'm/min','流道热偶温度是：',xtest[a,4], '摄氏度','一号挡帘高度是：',xtest[a,23],'mm')
    xtrain, xtest = z_score(xtrain, xtest)
    pca = PCA(xtrain, xtest, 0)
    num, t2gx = pca.pcafit_transform(kdelim=False, contrfig=True)
    now_time = time.strftime("%H:%M:%S")
    if num != prenum:
        threading.Timer(1, run2).start()
    else:
        print("!!!!!!!!!!!!!还未计算完成", prenum, num)
        run2

    time_end = time.time()
    print('time cost', time_end - time_start, 's')


#
def index(request):
    global prenum
    global now_time
    global t2gx
    global major_variable
    # print('前端来了')
    if prenum != num:
        if request.method == 'POST':
            prenum = num
            return JsonResponse(
                {"datas": {"local_time": now_time, "numgxt": t2gx, "num": num, "major_variable": major_variable}})
    else:
        pass
    return render(request, 'dtT2.html')


timer = threading.Timer(1, run2)  # 每秒运行
timer.start()  # 执行方法

# def run():
#     global num
#     global now_time
#     now_time = time.strftime("%H:%M:%S")
#     num = random.randint(1, 100)
#     threading.Timer(1,run).start()
#     print("后端自动运行：",num,'----',now_time,type(num))
