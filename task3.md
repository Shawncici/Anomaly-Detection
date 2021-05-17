# 线性回归模型检测异常
核心思想：
真实值与模型预测值之间的差值可以用来衡量这个数据点的异常可能性。
## 检测步骤
1. 将数据集分割为两部分：训练集（全部为正常点）和测试集（包含异常点）；
2. 利用训练集得到线性回归模型的参数和选择是否判断为异常的阀值；
3. 利用第二步得到的模型和阀值检测测试集


  ![image](https://github.com/Shawncici/Anomaly-Detection/blob/main/BB261DAD-E7F9-4DA7-9590-9B060B8C102B.png)
左侧是基于正常数据的拟合结果，右侧是基于含有异常数据的拟合结果，蓝色为拟合线，绿色是基于正常数据的拟合线。通过上图可以发现，因为异常点的存在，使得拟合线发生了偏移，使得“正常值显得有些不正常，异常值显得有点正常”。

# 异常检测中的异常类型
异常检测中主要包含三种异常：点异常（point anomalies）、上下文异常（contextual anomalies）以及集合异常（collective anomalies）。
## 点异常
其中，O1和O2就属于点异常
![image](https://github.com/Shawncici/Anomaly-Detection/blob/main/905780D9-5B8E-4F78-ADB0-377B87A9EFEC.png)
## 上下文异常
![image](https://github.com/Shawncici/Anomaly-Detection/blob/main/A11B2641-0B72-4964-A651-B962AABD7DE4.png)
其中t2处的异常属于上下文异常
## 集合异常

![image](https://github.com/Shawncici/Anomaly-Detection/blob/main/147E88AD-D390-43D8-A0ED-2A879C9ED58A.png)
图中箭头所指的红框区域为集合异常。

# 数据降维
## 问题
* 想建立一个AI模型，筛选金融股票，潜在数据指标：价格、交易量、换手率、股东人数、最近N日涨跌幅、RSI指标、市值、营业额、净利润、负债率、利润增长率…多达几百\上千个因子
* 两大问题：求解困难、模型过拟合
## 定义
在一定的限定条件下，按照一定的规则，尽可能保留原始数据重要信息的同时，降低数据维度。

[image:91A615DE-0926-4244-A4DB-448FBDE0BA7F-1613-000004AABE814051/7B68215F-D397-4B02-B203-C4E2151A1F30.png]
## 为什么要降低维度
维数灾难：随着特征数量越来越多，为了避免过拟合，对样本数量的需求会以指数速度增长
[image:D2B28E87-4E2A-425D-BCF9-5C32F0728804-1613-000004B72291F8F7/711515F8-594F-43D3-B540-295F71DA29E7.png]
数据可视化：高维数据不能可视化，只有降低到二维或三维才可以可视化
## 数据降维的最常用的方法——主成分分析（PCA）
也成为主分量分析，按照一定规则把数据变换到一个新的坐标系统中，使得任何数据投影后尽可能可以分开（新数据尽可能不相关，分布方差最大化）
核心：投影后的数据尽可能分得开（即不相关）
### PCA的实现
* 使投影后数据得方差最大，因为方差越大数据也越分散
* 数据预处理（数据分布标准化：μ=0，α=1）
* 计算协方差矩阵特征向量、及数据在各特征向量投影后的方差
* 根据需求（任务指定或方差比例）确定降维维度K
* 选取K维特征向量，计算数据在其形成空间得投影
[image:676EE2E2-A1BE-478C-9A1B-1790625E9C3A-1447-0000025A2AC09DEC/A56B274D-9E14-4B7D-B904-FEEA77BDBECB.png]
[image:571732B3-FABF-4CEF-ABC8-FA93732BC97C-1447-00000260774F474B/088E0578-3341-4945-B2F9-66AD7B55F787.png]
``` 
#数据标准化处理,均值变为0，标准差变为1
from sklearn.preprocessing import StandardScaler
X_norm=StandardScaler().fit_transform(X)
print(X_norm)
```
``` python
##数据降维到2维
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_norm)
print(X_pca.shape,X_norm.shape)
#计算方差比例
var_ratio2=pca.explained_variance_ratio_
print(var_ratio2)
```



