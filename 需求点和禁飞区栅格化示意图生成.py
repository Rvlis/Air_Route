"""
需求点和禁飞区栅格化示意图
"""

import matplotlib
from turtle import circle, color
import pandas as pd  #导入pandas库
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
# import folium
import pandas as pd
import webbrowser as wb
import numpy as np
# from folium.plugins import HeatMap
import matplotlib.colors as mcolors   # 取色器
import pylab   # 画图
df = pd.read_excel("./data/xuqiudian.xlsx") # 读取excle
X = df.values 
X=X[:,1:]
# print(X)
ss=StandardScaler()
ss.fit(X)
X_scale=ss.transform(X)
# print('原始数据直接标准化:',X_scale)
# data_df = pd.DataFrame(X_scale)   #关键1，将ndarray格式转换为DataFrame

# # # 更改表的索引
# data_df.columns = ['经度','纬度','需求量']  #将第一行的0,1,2,...,9变成A,B,C,...,J
# # data_df.index = ['a','b','c','d','e','f','g','h','i','j']

# # 将文件写入excel表格中
# writer = pd.ExcelWriter('2.xlsx')  #关键2，创建名称为hhh的excel表格
# data_df.to_excel(writer,'page_1',float_format='%.5f')  #关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
# writer.save()  #关键4
# X, Y = np.meshgrid(X[:5,0], X[:5,1])

X, Y = X[:,0]-120, X[:,1]-30
barrier = pd.read_excel("./data/jinfeiqu.xlsx") # 读取excle
bar = barrier.values
bar = bar[:,1:]
# print(bar)
barY , barX = bar[:,0]-30,bar[:,1]-120
# figure = plt.figure()
# plt=figure.add_subplot()
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
# plt.rcParams.update({"font.size":10})#此处必须添加此句代码方可改变标题字体大小
plt.scatter(X, Y,marker='.',label='需求点')
plt.scatter(barX,barY,marker='.',color='red',label='禁飞区', linewidth=1)

plt.plot(barX,barY,linestyle="solid", color="red", linewidth=2)  #绘制禁飞区轮廓
plt.plot([barX[0],barX[-1]],[barY[0],barY[-1]],linestyle="solid", color="red", linewidth=2) #禁飞点首尾相连

# circle = plt.Circle((barX[0], barY[0]), 0.14, color="y", fill=False)
# plt.gcf().gca().add_artist(circle)

plt.xticks(np.linspace(1,2,21),fontsize=8)
plt.yticks(np.linspace(0.7,1.5,17),fontsize=8)
plt.title('上海市需求点和禁飞区栅格化示意图')
plt.xlabel('经度')
plt.ylabel('纬度')
plt.grid(True)
plt.legend()
plt.show()
## 18.2
# from math import sin, asin, cos, radians, fabs, sqrt
 
# EARTH_RADIUS=6371           # 地球平均半径，6371km
 
# def hav(theta):
#     s = sin(theta / 2)
#     return s * s
 
# def get_distance_hav(lat0, lng0, lat1, lng1):
#     "用haversine公式计算球面两点间的距离。"
#     # 经纬度转换成弧度
#     lat0 = radians(lat0)
#     lat1 = radians(lat1)
#     lng0 = radians(lng0)
#     lng1 = radians(lng1)
 
#     dlng = fabs(lng0 - lng1)
#     dlat = fabs(lat0 - lat1)
#     h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
#     distance = 2 * EARTH_RADIUS * asin(sqrt(h))
 
#     return distance
 
# lon1,lat1 = (22.599578, 113.973129) #深圳野生动物园(起点）
# lon2,lat2 = (22.6986848, 114.3311032) #深圳坪山站 (百度地图测距：38.3km)
# d2 = get_distance_hav(lon1,lat1,lon2,lat2)
# print(d2)
 
# lon2,lat2 = (39.9087202, 116.3974799) #北京天安门(1938.4KM)
# d2 = get_distance_hav(lon1,lat1,lon2,lat2)
# print(d2)
 
# lon2,lat2 =(34.0522342, -118.2436849) #洛杉矶(11625.7KM)
# d2 = get_distance_hav(lon1,lat1,lon2,lat2)
# print(d2)


# theta = np.linspace(0, 2 * np.pi, 200)
# x = np.cos(theta)
# y = np.sin(theta)
# fig, plt = plt.subplots(figsize=(4, 4))
# plt.plot(x, y, color="darkred", linewidth=2)
# plt.xpltis.set_major_locator(plt.NullLocator()) # 删除x轴刻度，如需显示x轴刻度注释该行代码即可。
# plt.ypltis.set_major_locator(plt.NullLocator()) # 删除y轴刻度，如需显示y轴刻度注释该行代码即可。
# plt.pltis("equal")


# from scipy.spatial.distance import pdist, squareform

# points = X

# dist = squareform(pdist(points))

# for i in range(len(dist)):
#     sortindex = np.argsort(dist[i])
#     for j in range(1,k+1):
#         d = dist[i,sortindex[j]]

