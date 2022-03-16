"""
性能评估
参数说明:
        dir：指定使用文件（Knn航线信息和Prime航线信息）的文件夹，默认为data文件夹

具体用法(以精度2为例):
(精度2算法1): 
python .\Evaluation.py --dir .\data\2\2_1\

(精度2算法1): 
python .\Evaluation.py --dir .\data\2\2_2\
"""

import pandas as pd
from geopy.distance import geodesic
from isinpolygon2D import isin_multipolygon
import argparse
import os

def degree(edges):
    """
    度
    """
    # node_set = set()
    k = dict()
    node_set = set()
    id2coor = dict()
    for edge in edges:
        s_id = int(edge[0])
        t_id = int(edge[1])
        s_long = edge[2]
        s_lat = edge[3]
        t_long = edge[4]
        t_lat = edge[5]

        if s_id not in node_set:
            k[s_id] = 1
            node_set.add(s_id)
            id2coor[s_id] = (s_long, s_lat)
        else:
            k[s_id] += 1
        
        if t_id not in node_set:
            k[t_id] = 1
            node_set.add(t_id)
            id2coor[t_id] = (t_long, t_lat)
        else:
            k[t_id] += 1

    N = len(node_set)
    return k, id2coor, N
        
def clustering_coefficient(knn_edges,edges):
    """
    聚类系数
    """
    node_set = set()
    adj_node = dict()

    for edge in edges:
        s_id = int(edge[0])
        t_id = int(edge[1])
        if s_id not in node_set:
            node_set.add(s_id)
            adj_node[s_id] = set()
            adj_node[s_id].add(t_id)
        else:
            adj_node[s_id].add(t_id)
        
        if t_id not in node_set:
            node_set.add(t_id)
            adj_node[t_id] = set()
            adj_node[t_id].add(s_id)
        else:
            adj_node[t_id].add(s_id)

    knn_node_set = set()
    knn_adj_node = dict()
    knn_k, _, _ = degree(knn_edges)
    for key, value in knn_k.items():
        knn_k[key] = int(value/2)
    
    for knn_edge in knn_edges:
        s_id = int(knn_edge[0])
        t_id = int(knn_edge[1])
        if s_id not in knn_node_set:
            knn_node_set.add(s_id)
            knn_adj_node[s_id] = set()
            knn_adj_node[s_id].add(t_id)
        else:
            knn_adj_node[s_id].add(t_id)
        
        if t_id not in knn_node_set:
            knn_node_set.add(t_id)
            knn_adj_node[t_id] = set()
            knn_adj_node[t_id].add(s_id)
        else:
            knn_adj_node[t_id].add(s_id)
        
    C_list = list()
    C = 0.0
    
    for s_id, t_ids in knn_adj_node.items():
        ki = knn_k[s_id]
        # print("{}: {}".format(s_id, ki))
        Ei = 0
        for t_id1 in t_ids:
            for t_id2 in t_ids:
                if t_id2 in knn_adj_node[t_id1]:
                    Ei += 1
        if ki*(ki-1) != 0:
            C_list.append([s_id, list(t_ids), 1.0*Ei/(ki*(ki-1))])
            C += 1.0*Ei/(ki*(ki-1))
    
    C = C/len(knn_node_set)


    return C, adj_node, C_list

def DFS(id1, id2, adj_node, dfs_node_set, step, path):
    length = -1
    if id1 == id2:
        return step, path
    for id in adj_node[id1]:
        if id not in dfs_node_set:
            dfs_node_set.add(id)
            path.append(id)
            length, path = DFS(id, id2, adj_node, dfs_node_set, step+1, path)
            if length != -1:
                return length, path
            path.pop()

    return length, path

def aver_path_length(edges, adj_node):
    """
    平均路径长度
    """
    L = 0.0
    sum_L = 0
    node_set = set()
    for edge in edges:
        s_id = int(edge[0])
        t_id = int(edge[1])
        if s_id not in node_set:
            node_set.add(s_id)

        if t_id not in node_set:
            node_set.add(t_id)
    # node_num = len(node_set)
    node_list = list(node_set)
    sorted(node_list)
    # print(node_list)
    path_len_list = list()
    paths = list()
    for id1 in node_list:
        for id2 in node_list:
            if id2 <= id1:
                continue
            else:
                
                dfs_node_set = set()
                dfs_node_set.add(id1)
                path = list()
                path.append(id1)
                length, path = DFS(id1, id2, adj_node, dfs_node_set, 0, path)
                # print("{}->{}: {}".format(id1, id2, length))
                # print(path)
                paths.append(path)
                path_len_list.append([id1, id2, length, path])
                sum_L += length
                dfs_node_set.clear()
    N = len(node_set)
    L = sum_L/(N*(N+1)/2.0)
    # print("sum is {}, N is {}".format(sum_L, N))
    return L, path_len_list, paths

def network_efficiency(path_len_list):
    """
    网络效率
    """
    D = 0.0
    node_set = set()
    network_efficiency_list = list()
    for path in path_len_list:
        if path[0] not in node_set:
            node_set.add(path[0])
        if path[1] not in node_set:
            node_set.add(path[1])
        D += 2.0/path[2]
        network_efficiency_list.append([path[0], path[1], path[2], 1.0/path[2]])
        network_efficiency_list.append([path[1], path[0], path[2], 1.0/path[2]])
    N = len(node_set)
    E = D/(N*(N-1))
    
    # print(N)
    return E, network_efficiency_list

def connectivity_factor(L):
    """
    连通系数
    """
    C = 1.0/L
    return C

def network_connectivity(N, M):
    """
    网络连接度
    """
    return 2.0*M/N

def network_length(edges):
    """
    航路网长度
    """
    NL = 0.0
    for edge in edges:
        NL += edge[-1]
    return NL

def nonlinear_coefficient(paths, id2coor):
    """
    非直线系数
    """
    NLC = 0.0
    NLC_list = list()
    for path in paths:
        s_id = path[0]
        t_id = path[-1]
        s_long = id2coor[s_id][0]
        s_lat = id2coor[s_id][1]
        t_long = id2coor[t_id][0]
        t_lat = id2coor[t_id][1]
        linear_dis = geodesic((s_lat, s_long), (t_lat, t_long)).km
        nonlinear_dis = 0.0
        for i in range(len(path)-1):
            id1 = path[i]
            id2 = path[i+1]
            long1 = id2coor[id1][0]
            lat1 = id2coor[id1][1]
            long2 = id2coor[id2][0]
            lat2 = id2coor[id2][1]
            nonlinear_dis += geodesic((lat1, long1), (lat2, long2)).km
        # print(s_id, t_id, "linear dis:", linear_dis, "non dis:", nonlinear_dis)
        NLC += nonlinear_dis/linear_dis
        NLC_list.append([s_id, t_id, path, nonlinear_dis/linear_dis])
    
    return NLC/len(paths), NLC_list


def evaluation(args):
    
    files = os.listdir(args.dir)
    knn_path = ""
    prime_path = ""
    for file_name in files:
        if "Knn航线信息" in file_name:
            knn_path = file_name
        elif "Prime航线信息" in file_name:
            prime_path = file_name

    if knn_path == "" or prime_path == "":
        print("请检查路径是否正确并确保该路径已保存所需文件：Knn航线信息和Prime航线信息")
        return None
    
    metrics = dict()
    # ["起点id","终点id","起点经度","起点纬度","终点经度","终点纬度","距离"]
    # edges = pd.read_excel("./data/prime_edges.xlsx")
    edges = pd.read_excel(os.path.join(args.dir, prime_path))
    edges = list(edges.values)
    # ["起点id","终点id","起点经度","起点纬度","终点经度","终点纬度","距离", "是否跨越禁飞区"]
    # knn_edges = pd.read_excel("./data/knn_edges.xlsx")

    knn_edges = pd.read_excel(os.path.join(args.dir, knn_path))
    knn_edges = list(knn_edges.values)
    
    # 度 
    k, id2coor, N = degree(edges)
    k_list = list()
    D = 0
    for key, value in k.items():
        k_list.append([key, value, id2coor[key][0], id2coor[key][1]])
        D += value
    df = pd.DataFrame(k_list, columns=["id", "度", "经度", "纬度"])
    # df.to_excel("./data/度.xlsx", index=False)
    df.to_excel(os.path.join(args.dir, "度.xlsx"), index=False)
    D = D/len(k_list)
    metrics["度"] = D

    # 聚类系数
    C, adj_node, C_list = clustering_coefficient(knn_edges, edges)
    # print("C is:", C)
    df = pd.DataFrame(C_list, columns=["id", "邻接节点", "聚类系数"])
    # df.to_excel("./data/聚类系数.xlsx", index=False)
    df.to_excel(os.path.join(args.dir, "聚类系数.xlsx"), index=False)
    metrics["聚类系数"] = C
    # for key, value in cc.items():
    #     print("{}: {}".format(key, value))

    # 平均路径长度
    L, path_len_list, paths = aver_path_length(edges, adj_node)
    df = pd.DataFrame(path_len_list, columns=["id1", "id2", "最短距离", "路径"])
    # df.to_excel("./data/平均路径长度.xlsx", index=False)
    df.to_excel(os.path.join(args.dir, "平均路径长度.xlsx"), index=False)
    # print("L is:", L)
    metrics["平均路径长度"] = L
    
    # 网络效率
    E, network_efficiency_list = network_efficiency(path_len_list)
    df = pd.DataFrame(network_efficiency_list, columns=["id1", "id2", "最短距离d", "1/d"])
    # df.to_excel("./data/网络效率.xlsx", index=False)
    df.to_excel(os.path.join(args.dir, "网络效率.xlsx"), index=False)
    # print("E is:", E)
    metrics["网络效率"] = E

    # 连通系数
    C = connectivity_factor(L)
    metrics["连通系数"] = C

    # 网络连接度
    M = len(edges)
    J = network_connectivity(N, M)
    # print("J is:", J)
    metrics["网络连接度"] = J

    # 航路网长度
    NL = network_length(edges)
    # print("NL is:", NL)
    metrics["航路网长度"] = NL

    # 非直线系数
    NLC, NLC_list = nonlinear_coefficient(paths, id2coor)
    df = pd.DataFrame(NLC_list, columns=["id1", "id2", "路径", "非直线系数"])
    # df.to_excel("./data/非直线系数.xlsx", index=False)
    df.to_excel(os.path.join(args.dir, "非直线系数.xlsx"), index=False)
    metrics["非直线系数"] = NLC

    metrics_key = list()
    metrics_value = list()
    values = list()
    for key, value in metrics.items():
        metrics_key.append(key)
        values.append(value)
    metrics_value.append(values)

    df = pd.DataFrame(metrics_value, columns=metrics_key)
    # df.to_excel("./data/性能指标.xlsx", index=False)
    df.to_excel(os.path.join(args.dir, "性能指标.xlsx"), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="性能评估")

    parser.add_argument("--dir", default="./data", help="保存所有性能指标的文件夹路径，默认为data文件夹")
    # parser.add_argument("--ke", default="./data/Knn航线信息.xlsx", help="读取knn航线图边信息路径，默认为./data/Knn航线信息.xlsx")
    # parser.add_argument("--pe", default="./data/Prime航线信息.xlsx", help="读取prime航线图边信息路径，默认为./data/Prime航线信息.xlsx")
    args = parser.parse_args()

    evaluation(args)
    