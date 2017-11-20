# coding=utf-8
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import math
import time


def loadDataSet(fileName):  # 解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
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
def kMeans(dataSet, k, distMeans =distEclud, createCent = randCent):
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
         print (centroids)
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
       
def main():     
     dataMat =mat(loadDataSet('788points1.txt'))
     myCentroids, clustAssing= kMeans(dataMat,7)
     print (clustAssing)
     show(dataMat, 7, myCentroids, clustAssing)  
     
if __name__ == "__main__":
    main()
