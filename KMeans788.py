# coding=utf-8
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import math
import time

def distance(vecA, vecB):
    '''计算vecA与vecB之间的欧式距离的平方
    input:  vecA(mat)A点坐标
            vecB(mat)B点坐标
    output: dist[0, 0](float)A点与B点距离的平方
    '''
    dist = (vecA - vecB) * (vecA - vecB).T
    return dist[0, 0]
def loadDataSet(fileName):  # 解析文件按tab分割字段得到一个浮点数字类型的矩阵
     dataMat = []              # 文件的最后一个字段是类别标签
     fr = open(fileName)
     for line in fr.readlines():
         curLine = line.strip().split('\t')
         fltLine =list(map(float, curLine) )  # 将每个元素转成float类型
         dataMat.append(fltLine)
     return dataMat
 
 # 计算欧几里得距离
def distEclud(vecA, vecB):
     return sqrt(sum(power(vecA - vecB, 2))) # 求两个向量之间的距离
 
 # 构建聚簇中心，取k个(此例中为4)随机质心
def randCent(dataSet, k):
     n = shape(dataSet)[1]
     centroids = mat(zeros((k,n)))   # 每个质心有n个坐标值，总共要k个质心
     for j in range(n):
         minJ = min(dataSet[:,j])
         maxJ = max(dataSet[:,j])
         rangeJ = float(maxJ-minJ)
   
         centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
     return centroids

 # k-means 聚类算法
def kMeans_2(dataSet, k, distMeans =distEclud, createCent = randCent):
     m = shape(dataSet)[0]
     clusterAssment = mat(zeros((m,2)))    # 用于存放该样本属于哪类及质心距离
     # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
     centroids = createCent(dataSet, k)
     clusterChanged = True   # 用来判断聚类是否已经收敛
     while clusterChanged:
         clusterChanged = False
         for i in range(m):  # 把每一个数据点划分到离它最近的中心点
             minDist = inf;minIndex = -1;
             for j in range(k):
                 distJI = distMeans(centroids[j,:], dataSet[i,:])
                 if distJI < minDist:
                     minDist = distJI
                     minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
             if clusterAssment[i,0] != minIndex: 
                 clusterChanged = True  # 如果分配发生变化，则需要继续迭代
             clusterAssment[i,:] = minIndex,minDist**2   # 并将第i个数据点的分配情况存入字典
         for cent in range(k):   # 重新计算中心点
             ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]   # 去第一列等于cent的所有列
             centroids[cent,:] = mean(ptsInClust, axis = 0)  # 算出这些数据的中心点
     return centroids, clusterAssment
 # --------------------测试----------------------------------------------------
 # 用测试数据及测试kmeans算法
def show(dataSet, k, centroids, clusterAssment):
   
     numSamples, dim = dataSet.shape  
     mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
     for i in range(numSamples):  
         markIndex = int(clusterAssment[i, 0])  
         plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  
     mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
     for i in range(k):  
         plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)  
     plt.show()
def save_result(file_name, source):
    '''保存source中的结果到file_name文件中
    input:  file_name(string):文件名
            source(mat):需要保存的数据
    output:
    '''
    m, n = np.shape(source)
    f = open(file_name, "w")
    for i in xrange(m):
        tmp = []
        for j in xrange(n):
            tmp.append(str(source[i, j]))
        f.write("\t".join(tmp) + "\n")
    f.close()
def save_result_add(file_name, source):
    '''以添加数据的方法保存source中的结果到file_name文件中
    input:  file_name(string):文件名
            source(mat):需要保存的数据
    output:
    '''
    m, n = shape(source)
    f = open(file_name, "a")
    for i in xrange(m):
        tmp = []
        for j in xrange(n):
            tmp.append(str(source[i, j]))
        f.write("\t".join(tmp) + "\n")
    f.close()
def kmeans(data, k, centroids):
    '''根据KMeans算法求解聚类中心
    input:  data(mat):训练数据
            k(int):类别个数
            centroids(mat):随机初始化的聚类中心
    output: centroids(mat):训练完成的聚类中心
            subCenter(mat):每一个样本所属的类别
    '''
    m, n = shape(data) # m：样本的个数，n：特征的维度
    subCenter = mat(zeros((m, 2)))  # 初始化每一个样本所属的类别
    change = True  # 判断是否需要重新计算聚类中心
    while change == True:
        change = False  # 重置
        for i in xrange(m):
            minDist = inf  # 设置样本与聚类中心之间的最小的距离，初始值为正无穷
            minIndex = 0  # 所属的类别
            for j in xrange(k):
                # 计算i和每个聚类中心之间的距离
                dist = distance(data[i, ], centroids[j, ])
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            # 判断是否需要改变
            if subCenter[i, 0] <> minIndex:  # 需要改变
                change = True
                subCenter[i, ] = mat([minIndex, minDist])
        # 重新计算聚类中心
        for j in xrange(k):
            sum_all = mat(zeros((1, n)))
            r = 0  # 每个类别中的样本的个数
            for i in xrange(m):
                if subCenter[i, 0] == j:  # 计算第j个类别
                    sum_all += data[i, ]
                    r += 1
            for z in xrange(n):
                try:
                    centroids[j, z] = sum_all[0, z] / r
                except:
                    print " r is zero"
    return subCenter
def main1():
     dataMat =mat(loadDataSet('788points1.txt'))
     myCentroids, clustAssing= kMeans_2(dataMat,7)
     print (myCentroids)
     show(dataMat, 7, myCentroids, clustAssing)

def main2():
    h = 8  # 做h次小的Kmeans算法
    k = 7  # 聚类中心的个数
    dataMat = mat(loadDataSet('788points1.txt'))
    for i in xrange(0,h):
        ran100_data = randCent(dataMat, 100) #随机抽取100个数据作为新的数据集
        myCentroids, clustAssing = kMeans_2(ran100_data, 7) # 对这100个数据进行一次聚类算法
        save_result_add("ran100_center.txt", myCentroids) #记录聚类中心

    ran100_data = mat(loadDataSet('ran100_center.txt'))
    myCentroids2, clustAssing2 = kMeans_2(ran100_data, 7)  # 对之前得到的聚类中心进行聚类算法

    subcenter = kmeans(dataMat,7,myCentroids2) #用得到的聚类中心对原始数据进行聚类
    show(dataMat, 7, myCentroids2, subcenter)
if __name__ == "__main__":
    main2()
     
