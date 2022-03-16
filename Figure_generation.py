"""
生成图示
参数说明:
        dir：指定中间生成文件（Knn航线信息和Prime航线信息）的保存文件夹，默认为data文件夹
        cp: 是否生成圆心图，1是，0否，默认生成
        knn: 是否生成KNN航线图，1是，0否，默认生成
        prime: 是否生成Prime航线图，1是，0否，默认生成
        grid: 图片是否带栅格，1是，0否，默认否
        title: 指定图片标题
        jf: 禁飞点的文件路径，默认为./data/jinfeiqu.xlsx
        k/th: 指定knn算法的k值和阈值，默认k=5，th=30.0
        yxform: 圆心形式，星形（0）或者圆形（1），默认星形（0）
        yuan: 是否需要绘制圆，一般和yxform配合使用，1绘制，0不绘制， 默认绘制

具体用法(以精度2为例):
(精度2算法1): 
python .\Figure_generation.py --dir ./data/2/2_1 --pre 2 --fun 1 --knn 0 --prime 0 --title 网络节点示意图
python .\Figure_generation.py --dir ./data/2/2_1 --pre 2 --fun 1 --knn 1 --prime 0 --title 基于K-NN算法的候选网络示意图
python .\Figure_generation.py --dir ./data/2/2_1 --pre 2 --fun 1 --knn 1 --prime 1 --title 基于Prims算法的最小生成树示意图
python .\Figure_generation.py --dir ./data/2/2_1 --pre 2 --fun 1 --knn 0 --prime 1 --yxform 1 --yuan 0 --title 上海市低空无人机干线航路网络示意图
(支线图带圆) python .\Figure_generation.py --dir ./data/2/2_1 --pre 2 --fun 1 --knn 0 --prime 1 --feeder 1 --yxform 1 --title 上海市低空无人机支线航路网络示意图
(支线图不带圆) python .\Figure_generation.py --dir ./data/2/2_1 --pre 2 --fun 1 --knn 0 --prime 1 --feeder 1 --yxform 1 --yuan 0 --title 上海市低空无人机支线航路网络示意图

(精度2算法1): 
python .\Figure_generation.py --dir ./data/2/2_2 --pre 2 --fun 2 --knn 0 --prime 0 --title 网络节点示意图
python .\Figure_generation.py --dir ./data/2/2_2 --pre 2 --fun 2 --knn 1 --prime 0 --title 基于K-NN算法的候选网络示意图
python .\Figure_generation.py --dir ./data/2/2_2 --pre 2 --fun 2 --knn 1 --prime 1 --title 基于Prims算法的最小生成树示意图
python .\Figure_generation.py --dir ./data/2/2_2 --pre 2 --fun 2 --knn 0 --prime 1 --yxform 1 --yuan 0 --title 上海市低空无人机干线航路网络示意图
(支线图带圆) python .\Figure_generation.py --dir ./data/2/2_2 --pre 2 --fun 2 --knn 0 --prime 1 --feeder 1 --yxform 1 --title 上海市低空无人机支线航路网络示意图
(支线图不带圆) python .\Figure_generation.py --dir ./data/2/2_2 --pre 2 --fun 2 --knn 0 --prime 1 --feeder 1 --yxform 1 --yuan 0 --title 上海市低空无人机支线航路网络示意图
"""

from fileinput import filename
from tokenize import Double
from geopy.distance import geodesic
import pandas as pd
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from isinpolygon2D import isin_multipolygon
import argparse
import os
import cv2 as cv


def cross_no_fly(long, lat, no_fly, pre):
    """判断航线是否跨越禁飞区
    """
    pre_dict = {"1":0.1, "2":0.01, "3":0.001, "4":0.0001, "5":0.00001}
    if pre in pre_dict.keys():
        step = pre_dict[pre]
    else:
        step = 0.01
    if long[0] != long[1]:
        k =  (lat[0]-lat[1])/(long[0]-long[1])
        b = lat[0] - long[0] * k
        n_long = min(long[0], long[1])
        end_long = max(long[0], long[1])
        while n_long < end_long:
            n_lat = k * n_long + b
            if isin_multipolygon([n_long, n_lat], no_fly, contain_boundary=True):
                return True
            n_long += step
    else:
        n_lat = min(lat[0], lat[1])
        end_lat = max(lat[0], lat[1])
        while n_lat < end_lat:
            if isin_multipolygon([long[0], n_lat], no_fly, contain_boundary=True):
                return True
            n_lat += step

    return False


def Figure_generation(args, pre, fun):
    """生成图片
    """
    files = os.listdir(args.dir)
    yx_path = ""
    xq_path = ""
    for file_name in files:
        if "圆心_" in file_name:
            yx_path = os.path.join(args.dir, file_name)
        elif "筛选后需求点_" in file_name:
            xq_path = os.path.join(args.dir, file_name)
    if yx_path == "" or xq_path == "":
        print("请检查输入路径是否正确且该路径下存在圆心和筛选后需求点文件")
        return None

    # 读取圆心坐标
    # X = pd.read_excel(args.yx)
    X = pd.read_excel(yx_path) 
    X = X.values
    X, Y = X[:,0]-120.0, X[:,1]-30.0

    # 读取禁飞点坐标
    barrier = pd.read_excel(args.jf)
    bar = barrier.values
    bar = bar[:,1:]
    barY , barX = bar[:,0]-30.0,bar[:,1]-120.0

    # 读取需求点坐标
    # require = pd.read_excel(args.xq)
    require = pd.read_excel(xq_path)
    require = require.values
    require = require[:,1:3]
    reX, reY = require[:,0]-120.0, require[:,1]-30.0
    
    
    matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
    # plt.rcParams.update({"font.size":10})             #此处必须添加此句代码方可改变标题字体大小

    
    # 绘制禁飞点和需求点
    # 可在此处修改禁飞点和需求点的形状及颜色（RGB格式要转成16进制，如(255，255，255) -> #EEEEEE）
    plt.scatter(barX,barY,marker='',color='#ededed')
    plt.scatter(reX, reY, marker='.', color="blue")

    # 绘制禁飞区轮廓
    # 可在此处修改禁飞区轮廓的形状、颜色和粗细
    plt.plot(barX,barY,linestyle="solid", color="red", linewidth=1.5)
    plt.plot([barX[0],barX[-1]],[barY[0],barY[-1]],linestyle="solid", color="red", linewidth=1.5) #禁飞点首尾相连


    # 绘制圆及圆心
    # 手动调整圆心半径
    r = args.r
    if args.cp == 1:
        # 可在此处修改圆心的类型和颜色
        if args.yxform == 0:
            plt.scatter(X, Y, marker='*', color="black")
            if args.yuan == 1:
                for x, y in zip(X, Y):
                    # 可在此处修改圆的颜色、半径（0.13）、透明度（alpha）和粗细
                    circle = plt.Circle((x, y), r, color="black", alpha=0.5, linewidth=1, fill=False)
                    plt.gcf().gca().add_artist(circle)

        elif args.yxform == 1:
            plt.scatter(X, Y, marker='', color="black")
            for x, y in zip(X, Y):
                circle = plt.Circle((x, y), 0.013, color="black", alpha=0.5, linewidth=1, fill=False)
                plt.gcf().gca().add_artist(circle)
                if args.yuan == 1:
                    circle = plt.Circle((x, y), r, color="black", alpha=0.5, linewidth=1, fill=False)
                    plt.gcf().gca().add_artist(circle)

    # 禁飞区点集合
    no_fly = list()
    for x,y in zip(barX, barY):
        no_fly.append([x+120.0, y+30.0])
    no_fly.append(no_fly[0])
    #邻接矩阵（不包括跨越禁飞区的航线）
    # edges = dict()
    nodes = set()
    # print("(121.3, 31.34)是否在禁飞区内：", isin_multipolygon([121.3, 31.34], no_fly, contain_boundary=True))

    # 存储节点和边的信息，用于性能评估
    edge_set = set()        #不包含跨越禁飞区的航线
    all_edge_set = set()    #包含跨越禁飞区的航线
    node_id = 0
    node_id_set = set()
    node_id_dict = dict()
    id2coor = dict()    #id对应坐标

    # 绘制KNN航线图
    k = args.k
    all_neighbors = KNN(yx_path, k, args.th)
    index = 0
    for x,y in zip(X, Y):
        neighbors = all_neighbors[index]
        for num, neighbor in enumerate(neighbors):
            if num >= k:
                break
            if cross_no_fly([x+120.0, neighbor[0]], [y+30.0,neighbor[1]], no_fly, pre):
                if args.knn == 1:
                    # 可在此处设置knn中跨越禁飞区航线的类型，颜色和粗细 
                    plt.plot([x, neighbor[0]-120.0], [y, neighbor[1]-30.0], linestyle="--", color="forestgreen", linewidth=1)

                if (x, y) not in node_id_set:
                    node_id_set.add((x,y))
                    node_id_dict[(x, y)] = node_id
                    id2coor[node_id] = (x+120.0, y+30.0)
                    s_id = node_id
                    node_id += 1
                else:
                    s_id = node_id_dict[(x, y)]

                if (neighbor[0]-120.0, neighbor[1]-30.0) not in node_id_set:
                    node_id_set.add((neighbor[0]-120.0, neighbor[1]-30.0))
                    node_id_dict[(neighbor[0]-120.0, neighbor[1]-30.0)] = node_id
                    id2coor[node_id] = (neighbor[0], neighbor[1])
                    t_id = node_id
                    node_id += 1
                else:
                    t_id = node_id_dict[(neighbor[0]-120.0, neighbor[1]-30.0)]
                
                all_edge_set.add((s_id, t_id, neighbor[2], 1))
                all_edge_set.add((t_id, s_id, neighbor[2], 1))
            else:
                if ((x+120.0, y+30.0)) not in nodes:
                    # edges[(x+120.0, y+30.0)] = list()
                    # edges[(x+120.0, y+30.0)].append([neighbor[0], neighbor[1], neighbor[2]])
                    nodes.add((x+120.0, y+30.0))
                else:
                    # edges[(x+120.0, y+30.0)].append([neighbor[0], neighbor[1], neighbor[2]])
                    if args.knn == 1:
                        # 设置knn中普通航线的类型、颜色（RGB转成16进制）和粗细
                        plt.plot([x, neighbor[0]-120.0], [y, neighbor[1]-30.0], linestyle="-", color="darkgreen", linewidth=1.5)

                    if (x, y) not in node_id_set:
                        node_id_set.add((x,y))
                        node_id_dict[(x, y)] = node_id
                        id2coor[node_id] = (x+120.0, y+30.0)
                        s_id = node_id
                        node_id += 1
                    else:
                        s_id = node_id_dict[(x, y)]
                    if (neighbor[0]-120.0, neighbor[1]-30.0) not in node_id_set:
                        node_id_set.add((neighbor[0]-120.0, neighbor[1]-30.0))
                        node_id_dict[(neighbor[0]-120.0, neighbor[1]-30.0)] = node_id
                        id2coor[node_id] = (neighbor[0], neighbor[1])
                        t_id = node_id
                        node_id += 1
                    else:
                        t_id = node_id_dict[(neighbor[0]-120.0, neighbor[1]-30.0)]
                    
                    edge_set.add((s_id, t_id, neighbor[2]))
                    edge_set.add((t_id, s_id, neighbor[2]))
                    all_edge_set.add((s_id, t_id, neighbor[2], 0))
                    all_edge_set.add((t_id, s_id, neighbor[2], 0))
        index += 1

    # print("node num is {}.".format(node_id))
    # print(len(all_edge_set))
    edges = list()
    for knn_edge in all_edge_set:
        # print("{} -> {}: {}".format(edge[0], edge[1], edge[2]))
        s_long = id2coor[knn_edge[0]][0]
        s_lat = id2coor[knn_edge[0]][1]
        t_long = id2coor[knn_edge[1]][0]
        t_lat = id2coor[knn_edge[1]][1]
        edges.append([knn_edge[0], knn_edge[1], s_long, s_lat, t_long, t_lat, knn_edge[2], knn_edge[3]])
    df = pd.DataFrame(edges, columns=["id1","id2","起点经度","起点纬度","终点经度","终点纬度","距离", "跨越禁飞区"])
    # df.to_excel("./data/knn_edges.xlsx", index=False)
    # df.to_excel(args.ke, index=False)
    df.to_excel(os.path.join(args.dir, "Knn航线信息_{}_{}.xlsx".format(pre, fun)), index=False)

    # 绘制Prime航线图
    if args.prime == 1:
        # Prime_E = Prime(nodes, edges)
        Prime_E = Prime(node_id, edge_set)

        prime_edges = list()
        # print(len(Prime_E))
        for pe in Prime_E:
            s_id = pe[0]
            t_id = pe[1]
            s_long = id2coor[s_id][0] - 120.0
            s_lat = id2coor[s_id][1] - 30.0
            t_long = id2coor[t_id][0] - 120.0
            t_lat = id2coor[t_id][1] - 30.0

            prime_edges.append([s_id, t_id, s_long+120.0, s_lat+30.0, t_long+120.0, t_lat+30.0, pe[2]])
            # s_long = pe[0][0] - 120.0
            # s_lat = pe[0][1] - 30.0
            # t_long = pe[1][0] - 120.0
            # t_lat = pe[1][1] - 30.0
            # 可在此处设置Prime航线的类型、颜色和粗细
            plt.plot([s_long, t_long], [s_lat, t_lat], linestyle="-", color="black", linewidth=3)

            # 设置圆心id
            # plt.text(s_long, s_lat, str(s_id), fontsize=15, color="red")
            # plt.text(t_long, t_lat, str(t_id), fontsize=15, color="red")


        df = pd.DataFrame(prime_edges, columns=["id1","id2","起点经度","起点纬度","终点经度","终点纬度","距离"])
        # df.to_excel("./data/prime_edges.xlsx", index=False)
        # df.to_excel(args.pe, index=False)
        df.to_excel(os.path.join(args.dir, "Prime航线信息_{}_{}.xlsx".format(pre, fun)), index=False)

    # 绘制支线航路图
    if args.feeder == 1:
        selected_cp = Feeder_route(os.path.join(args.dir, "圆内需求点_{}_{}.xlsx".format(pre, fun)))
        # print("selected_cp:",selected_cp)
        for cp in selected_cp:
            # print(cp)
            cp_long = cp[0][0]
            cp_lat = cp[0][1]
            for xq in cp[2]:
                xq_long = xq[0]
                xq_lat = xq[1]
                # print("({}, {}) -> ({}, {})".format(cp_long, cp_lat, xq_long, xq_lat))
                plt.plot([cp_long-120.0, xq_long-120.0],[cp_lat-30.0, xq_lat-30.0],linestyle="-", color="#48c4ee", linewidth=1)    


    ax = plt.axes()

    # 可在此处设置图片背景色
    ax.set(facecolor = "#d0d0d0")
    
    # 可在此处设置禁飞区背景色、透明度（alpha）
    plt.fill(barX, barY, facecolor='#ededed', alpha=1)

    plt.xticks(np.linspace(1,2,21),fontsize=8)
    plt.yticks(np.linspace(0.7,1.5,17),fontsize=8)
    plt.title(args.title)
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.grid((args.grid == 1))
    plt.show()


def KNN(data_path, k=5, threshold=30.0):
    """
    K近邻
    """
    cps = list(pd.read_excel(data_path).values)
    all_neighbors = list()  #
    # adj_mat = dict()    #邻接矩阵（不包括跨越禁飞区的航线）

    for i in range(len(cps)):
        cp1 = list(cps[i])
        neighbors = list()
        # adj_mat[tuple(cps[i])] = list()
        for j in range(len(cps)):
            if i == j:
                continue
            cp2 = list(cps[j])
            dis = geodesic((cp1[1], cp1[0]), (cp2[1], cp2[0])).km
            if dis > threshold:
                continue
            else:
                cp2.append(dis)
                neighbors.append(list(cp2))
        neighbors = sorted(neighbors, key=lambda x:x[2])
        # print(neighbors)
        all_neighbors.append(neighbors)
    
    return all_neighbors

def Prime(node_num, edges):
    """
    Prime
    """
    # print("node num:", node_num)
    Prime_N = set()
    Prime_E = list()
    root = -1
    # print(root)
    # Prime_N.add(root)
    # count = 0
    node_num -= 1
    while node_num > 0:
        # count += 1
        shortest_dis = geodesic((31.50, 122.00), (30.70, 121.00)).km
        # p_id = -1
        s_id = -1
        for edge in edges:
            if root == -1:
                root = edge[0]
                Prime_N.add(root)
            if edge[0] in Prime_N and edge[1] not in Prime_N and edge[2] < shortest_dis:
                s_id = edge[0]
                t_id = edge[1]
                dis = edge[2]
                shortest_dis = dis
        if s_id != -1:
            if s_id in Prime_N and t_id not in Prime_N:
                Prime_N.add(t_id)
                Prime_E.append([s_id, t_id, shortest_dis])

        node_num -= 1

    return Prime_E

def Feeder_route(cp_with_xq_path):
    """
    支线航路
    """
    cp_with_xq = pd.read_excel(cp_with_xq_path).values
    cp_with_xq = list(cp_with_xq)
    # print("cp_with_xq:", len(cp_with_xq))
    cp_with_xq_list = list()
    cp_set = set()
    cp_dict = dict()
    for  item in cp_with_xq[::-1]:
        yx_long = item[0]
        yx_lat = item[1]
        xq_num = int(item[2])
        xq_long = item[3]
        xq_lat = item[4]
        xq_re = item[5]
        if xq_num >= 2 and xq_num <= 15:
            if (yx_long, yx_lat) not in cp_set:
                cp_set.add((yx_long, yx_lat))
                cp_dict[(yx_long, yx_lat)] = list()
                cp_dict[(yx_long, yx_lat)].append([xq_long, xq_lat, xq_re])
            else:
                cp_dict[(yx_long, yx_lat)].append([xq_long, xq_lat, xq_re])
        
    for key, value in cp_dict.items():
        # print("{}: {}".format(key, value))
        cp_with_xq_list.append([key, len(value), value])

    selected_cp = list()
    shortest_dis = geodesic((31.50, 122.00), (30.70, 121.00)).km
    for i in range(len(cp_with_xq_list)):
        for j in range(i+1, len(cp_with_xq_list)):
            i_long = cp_with_xq_list[i][0][0]
            i_lat = cp_with_xq_list[i][0][1]
            j_long = cp_with_xq_list[j][0][0]
            j_lat = cp_with_xq_list[j][0][1]
            dis = geodesic((i_lat, i_long), (j_lat, j_long)).km
            
            if dis < shortest_dis and dis <= 30.0:
                shortest_dis = dis
                selected_cp.clear()
                selected_cp.append(cp_with_xq_list[i])
                selected_cp.append(cp_with_xq_list[j])
                selected_cp_index = {i,j}

    selected_cp_2 = list()
    shortest_dis = geodesic((31.50, 122.00), (30.70, 121.00)).km
    for i in range(len(cp_with_xq_list)):
        if i in selected_cp_index:
            continue
        for j in range(i+1, len(cp_with_xq_list)):
            if j in selected_cp_index:
                continue
            i_long = cp_with_xq_list[i][0][0]
            i_lat = cp_with_xq_list[i][0][1]
            j_long = cp_with_xq_list[j][0][0]
            j_lat = cp_with_xq_list[j][0][1]
            dis = geodesic((i_lat, i_long), (j_lat, j_long)).km
            if dis < shortest_dis and dis <= 50.0:
                shortest_dis = dis
                # selected_cp.clear()
                selected_cp_2.clear()
                selected_cp_2.append(cp_with_xq_list[i])
                selected_cp_2.append(cp_with_xq_list[j])
    
    selected_cp.extend(selected_cp_2)

    return selected_cp
    
# def enlarged_drawing():
#     """
#     局部放大图
#     """
    # 读取图像并判断是否读取成功
    # img = cv.imread("./data/2/2_2/Figure_1.png")
    # #需要放大的部分
    # print(img.shape)
    
    # x1 = 173
    # y1 = 286
    # x2 = 338
    # y2 = 404
    # xlim = 575
    # ylim = 60
    # scaleX = int(1.2*(x2-x1))
    # scaleY = int(1.2*(y2-y1))

    # part = img[y1:y2, x1:x2]
    # # part = img[x1:x2, y1:y2]
    # #双线性插值法
    
    # # mask = cv.resize(part, (150, 150), fx=0, fy=0, interpolation=cv.INTER_LINEAR)
    # mask = cv.resize(part, (scaleX, scaleY), fx=0, fy=0, interpolation=cv.INTER_LINEAR)
    # if img is None is None:
    #     print('Failed to read picture')
    #     sys.exit()
        
    # #放大后局部图的位置img[210:410,670:870]
    # # img[0:int(1.5*(338-173)), 0: int(1.5*(404-286))]=mask

    # # img[58:208, 426:576]=mask
    # img[ylim:ylim+scaleY, xlim-scaleX:xlim]=mask
    

    # #画框并连线
    # # cv.rectangle(img,(173,286),(338,404),(0,255,0),2)
    # # cv.rectangle(img,(426,58),(576,208),(0,255,0),2)
    # # img = cv.line(img,(338,286),(426,208),(0,255,0),2)
    # # img = cv.line(img,(338,404),(576,208),(0,255,0),2)

    # cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    # cv.rectangle(img,(xlim-scaleX,ylim),(xlim,ylim+scaleY),(0,255,0),2)
    # img = cv.line(img,(x2,y1),(xlim-scaleX,ylim+scaleY),(0,255,0),2)
    # img = cv.line(img,(x2,y2),(xlim,ylim+scaleY),(0,255,0),2)
    # #展示结果
    # cv.imshow('img',img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # 图片路径
    # img = cv.imread('./data/2/2_2/Figure_1.png')
    # a = []
    # b = []
    
    
    # def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    #     if event == cv.EVENT_LBUTTONDOWN:
    #         xy = "%d,%d" % (x, y)
    #         a.append(x)
    #         b.append(y)
    #         cv.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
    #         cv.putText(img, xy, (x, y), cv.FONT_HERSHEY_PLAIN,
    #                     1.0, (0, 0, 0), thickness=1)
    #         cv.imshow("image", img)
    #         print(x,y)
    
    
    # cv.namedWindow("image")
    # cv.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    # cv.imshow("image", img)
    # cv.waitKey(0)
    # print(a[0], b[0])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="生成图")
    parser.add_argument("--dir", default="./data", help="指定生成图片和中间文件的保存文件夹，默认为data文件夹")
    parser.add_argument("--pre", choices=["1","2","3","4","5"], default="", help="给定精度")
    parser.add_argument("--fun", choices=["1","2"], default="", help="给定算法")
    parser.add_argument("--cp", choices=[0,1], type=int, default=1, help="是否生成圆心图，0：否；1：是，默认是")
    parser.add_argument("--knn", choices=[0,1], type=int, default=1, help="是否生成KNN图，0：否；1：是，默认是")
    parser.add_argument("--prime", choices=[0,1], type=int, default=1, help="是否生成KNN图，0：否；1：是，默认是")
    parser.add_argument("--feeder", choices=[0,1], type=int, default=0, help="是否绘制支线航路图，0：否；1：是，默认否")
    parser.add_argument("--grid", choices=[0,1], type=int, default=0, help="图片是否添加栅格，0：否；1：是，默认否")
    parser.add_argument("--title", type=str, default="图片标题", help="图片标题")
    parser.add_argument("--jf", default="./data/jinfeiqu.xlsx", help="读取禁飞点文件的路径，默认为'./data/jinfeiqu.xlsx'")
    # parser.add_argument("--xq", default="./data/xuqiudian.xlsx", help="读取需求点文件的路径，默认为'./data/xuqiudian.xlsx'")
    # parser.add_argument("--yx", default="./data/yuanxin.xlsx", help="读取圆心文件的路径，默认为'./data/yuanxin.xlsx'")
    parser.add_argument("--k", type=int, default=5, help="KNN算法选择参数k，默认为5")
    parser.add_argument("--th", type=float, default=35.0, help="KNN算法阈值（threshold）取值，默认为35.0")
    parser.add_argument("--yxform", choices=[0,1], type=int, default=0, help="圆心形式，星形（0）或者圆形（1），默认星形（0） ")
    parser.add_argument("--yuan", choices=[0,1], type=int, default=1, help="是否需要绘制圆，一般和yxform配合使用")
    parser.add_argument("--r", type=float, default=0.138, help="设置绘制圆的半径，默认为0.138（即13.8km）")
    

    args = parser.parse_args()

    # print("{},{}".format(pre, fun))
    
    Figure_generation(args, args.pre, args.fun)

    
 