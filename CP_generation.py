"""
文件命名规则：文件名_p_f.xlsx，其中p为算法使用的精度[1,2,3,4,5]，f为选择的算法[1,2](1考虑需求点，2考虑需求量)

算法1和2的实现寻找圆心，其中
算法1不考虑需求量
算法2考虑需求量

用法示例: python CP_generation.py  --dir ./data --fun 2 --pre 2 --jf ./data/jinfeiqu-raw.xlsx --xq ./data/xuqiudian-raw.xlsx
参数说明：
    dir:指定中间生成文件（包括圆心文件、圆内需求点文件、筛选后需求点文件）的保存文件夹，默认为data文件夹
    fun:指定选择算法1或者2, 默认为2
    pre:指定得到圆心坐标的精度（1e-x, x可选值为[1,2,3,4,5], 默认为2）
    fpre:不建议改动: 选择筛选需求点是否位于禁飞区内的精度，默认为1e-5, 精度越高，筛选掉的点越准确
    jf:指定读取禁飞点文件的路径，默认为 ./data/jinfeiqu-raw.xlsx
    xq:指定读取需求点的路径，默认为 ./data/xuqiudian-raw.xlsx

具体用法：
(精度1算法1)：python .\CP_generation.py --dir .\data\1\1_1\ --pre 1 --fun 1
(精度1算法2)：python .\CP_generation.py --dir .\data\1\1_2\ --pre 1 --fun 2
(精度2算法1)：python .\CP_generation.py --dir .\data\2\2_1\ --pre 2 --fun 1
(精度2算法2)：python .\CP_generation.py --dir .\data\2\2_2\ --pre 2 --fun 2
(精度3算法1)：python .\CP_generation.py --dir .\data\3\3_1\ --pre 3 --fun 1
(精度3算法2)：python .\CP_generation.py --dir .\data\3\3_2\ --pre 3 --fun 2
"""

import pandas as pd
from decimal import Decimal
from setuptools import Require
from geopy.distance import geodesic
from isinpolygon2D import isin_multipolygon
import os
import argparse

#精度设置
Precision = [[10, "0.0"], [100, "0.00"], [1000, "0.000"], [10000, "0.0000"], [100000, "0.00000"]]

def in_no_fly(precision, jf_path, xq_path, save_path):
    """读取原需求点坐标后，首先筛选掉位于禁飞区内的需求点，并以保留的需求点作为输入求下一步圆心
    """
    precision_1 = Precision[precision-1][0]
    precision_2 = Precision[precision-1][1]

    jf_set = set()   #set用于禁飞点去重
    jf_lst = list()  #禁飞点集合
    xq_lst = list()  #需求点集合
    precision
    jf_data = pd.read_excel(jf_path, sheet_name=0)
    nrows = jf_data.shape[0]
    for row in range(nrows):
        # print(float(require_data.iloc[row, 1]), float(require_data.iloc[row, 2]))
        longitude = jf_data.iloc[row, 2]
        latitude = jf_data.iloc[row, 1]
        longitude = int(float(Decimal(longitude).quantize(Decimal(precision_2)))*precision_1)
        latitude = int(float(Decimal(latitude).quantize(Decimal(precision_2)))*precision_1)
        # S1.add((float(jf_data.iloc[row, 1]), float(jf_data.iloc[row, 2])))
        if (longitude, latitude) not in jf_set:
            jf_set.add((longitude, latitude))
            jf_lst.append([longitude, latitude])
    jf_lst.append(jf_lst[0])
    # for jf in jf_lst:
    #     print(jf)

    xq_data = pd.read_excel(xq_path, sheet_name=0)
    nrows = xq_data.shape[0]
    for row in range(nrows):
        r_longitude = xq_data.iloc[row, 1]
        r_latitude = xq_data.iloc[row, 2]
        requirement = xq_data.iloc[row, 3]
        longitude = int(float(Decimal(r_longitude).quantize(Decimal(precision_2)))*precision_1)
        latitude = int(float(Decimal(r_latitude).quantize(Decimal(precision_2)))*precision_1)
        requirement = int(requirement)
        xq_lst.append([longitude, latitude, r_longitude, r_latitude, requirement])
    
    # print(jf_lst)
    filter_xq_lst = list()
    id = 1
    for xq in xq_lst:
        # print(xq[0:2], type(xq[0:2]))
        if not isin_multipolygon(xq[0:2], jf_lst, contain_boundary=True):
            # print(xq, type(xq))
            xq.insert(2,id)
            # print(xq[2:])
            filter_xq_lst.append(xq[2:])
            id += 1
    # print(filter_xq_lst)
    save_data(filter_xq_lst, save_path, column_name=["编号","经度","纬度","需求"])
    
    return save_path
    


def read_data(precision, jf_path, xq_path):
    """读取禁飞点、需求点坐标
    """
    precision_1 = Precision[precision-1][0]
    precision_2 = Precision[precision-1][1]
    S1 = set()  #禁飞点集合
    S2 = set()  #需求点集合
    S1_lst = list()

    jf_data = pd.read_excel(jf_path, sheet_name=0)
    nrows = jf_data.shape[0]
    for row in range(nrows):
        # print(float(require_data.iloc[row, 1]), float(require_data.iloc[row, 2]))
        longitude = jf_data.iloc[row, 2]
        latitude = jf_data.iloc[row, 1]
        longitude = float(Decimal(longitude).quantize(Decimal(precision_2)))     #修改精度
        latitude = float(Decimal(latitude).quantize(Decimal(precision_2)))
        # S1.add((float(jf_data.iloc[row, 1]), float(jf_data.iloc[row, 2])))
        if (longitude, latitude) not in S1:
            S1.add((longitude, latitude))
            S1_lst.append([int(longitude*precision_1), int(latitude*precision_1)])                #修改精度


    xq_data = pd.read_excel(xq_path, sheet_name=0)
    nrows = xq_data.shape[0]
    for row in range(nrows):
        longitude = xq_data.iloc[row, 1]
        latitude = xq_data.iloc[row, 2]
        require = xq_data.iloc[row,3]
        longitude = float(Decimal(longitude).quantize(Decimal(precision_2)))     #修改精度
        latitude = float(Decimal(latitude).quantize(Decimal(precision_2)))
        require = int(require)
        S2.add((longitude, latitude, require))
    
    return S1, S2, S1_lst

def save_data(data_list, save_path, column_name):
    """保存坐标
    """
    df = pd.DataFrame(data_list, columns=column_name)
    df.to_excel(save_path, index=False)

def cp_dis(cp1, cp2):
    """判断两个圆心之间的距离:
        <14km 返回 False；否则返回True
    """
    # print((cp1[1], cp1[0]), (cp2[1], cp2[0]))
    dis = geodesic((cp1[1], cp1[0]), (cp2[1], cp2[0])).km
    return dis >= 14.0


def fun1(precision, jf_path, xq_path, save_path):
    """伪代码1实现
    """
    precision_1 = Precision[precision-1][0]
    precision_2 = Precision[precision-1][1]
    # S1, S2, S1_lst = read_data("./jinfeiqu.xlsx", "./xuqiudian.xlsx")
    S1, S2, S1_lst = read_data(precision, jf_path, xq_path)
    S1_lst.append(S1_lst[0])
    
    # print(S1_lst)
    candidate_p = dict()    #候选点坐标
    selected_cp = list()     #选中的圆心
    selected_cp_contain_xq_num = dict()     #选中圆心包含的需求点
    # print(S1)
    # print(S2)
    k = 0
    r = 14.0  #半径

    while S2:
        # print(len(S2))
        for i in range(121*precision_1, 122*precision_1, 1):                                      #修改精度
            for j in range(int(307*precision_1/10), int(315*precision_1/10), 1):
                # print("候选点坐标：({},{})".format(i,j))
                # print(i,j)
                if isin_multipolygon([i, j], S1_lst, contain_boundary=True):
                    # print("({},{}) 在禁飞区内".format(i,j))
                    continue
                p_j = float(Decimal(j/precision_1*1.0).quantize(Decimal(precision_2)))       #修改精度
                p_i = float(Decimal(i/precision_1*1.0).quantize(Decimal(precision_2)))
                flag = False
                for xq_point in S2:
                    # print(xq_point[1], xq_point[0], p_j, p_i)
                    dis = geodesic((xq_point[1], xq_point[0]), (p_j, p_i)).km
                    # print(dis, type(dis))
                    if dis <= r:
                        # print("需求点({},{})位于候选点({},{})范围内".format(xq_point[0],xq_point[1],p_i,p_j))
                        if not flag:
                            candidate_p[(p_i, p_j)] = [xq_point]
                            flag = True
                        else:
                            candidate_p[(p_i, p_j)].append(xq_point)
                        # print((xq_point[1], xq_point[0], p_j, p_i))
        if candidate_p:
            find_cp = False
            while candidate_p and not find_cp:
                center_p = max(candidate_p.items(), key=lambda x:len(x[1]))[0]  #tuple
                dis_flag = True

                for cp in selected_cp:
                    if not cp_dis(tuple(cp), center_p):
                        dis_flag = False
                        del candidate_p[center_p]
                        break

                if dis_flag:
                    selected_cp_contain_xq_num[center_p] = candidate_p[center_p]
                    for removed_p in candidate_p[center_p]:
                        S2.discard(removed_p)
                    
                    print("添加({},{})为圆心".format(center_p[0],center_p[1]))
                    k += 1
                    selected_cp.append(list(center_p))
                    find_cp = True
                    candidate_p.clear()

            if not find_cp:
                break
        else:
            break

    save_data(selected_cp, save_path, column_name=["经度","纬度"])
    cp_and_xq_num = list()
    for cp, xqs in selected_cp_contain_xq_num.items():
        for xq in xqs:
            cp_and_xq_num.append([cp[0], cp[1], len(xqs), xq[0], xq[1], xq[2]])
    
    df = pd.DataFrame(cp_and_xq_num, columns=["圆心经度", "圆心纬度", "圆内需求点数量", "需求点经度", "需求点纬度", "需求点需求量"])
    # df.to_excel("./data/圆内需求点.xlsx", index=False)
    df.to_excel(os.path.join(args.dir, "圆内需求点_{}_{}.xlsx".format(args.pre, args.fun)), index=False)
    return k



def fun2(precision, jf_path, xq_path, save_path):
    """伪代码2实现：考虑需求值
    """
    precision_1 = Precision[precision-1][0]
    precision_2 = Precision[precision-1][1]
    S1, S2, S1_lst = read_data(precision, jf_path, xq_path)
    S1_lst.append(S1_lst[0])
    # for s in S1_lst:
    #     print(s)
    # print(S1_lst)
    candidate_p = dict()    #候选点坐标
    candidate_p_re = dict() #候选点坐标需求值和
    selected_cp = list()     #选中的圆心
    selected_cp_contain_xq_num = dict()     #选中圆心包含的需求点

    # print(S1)
    # print(S2)
    k = 0
    r = 14.0  #半径

    while S2:
        # print(len(S2))
        for i in range(int(121*precision_1), int(122*precision_1), 1):                                     #修改精度
            for j in range(int(307*precision_1/10), int(315*precision_1/10), 1):
                # print("候选点坐标：({},{})".format(i,j))
                # print(i,j)
                if isin_multipolygon([i, j], S1_lst, contain_boundary=True):
                    # print("({},{}) 在禁飞区内".format(i,j))
                    continue
                p_j = float(Decimal(j/precision_1*1.0).quantize(Decimal(precision_2)))      #修改精度
                p_i = float(Decimal(i/precision_1*1.0).quantize(Decimal(precision_2)))
                flag = False
                for xq_point in S2:
                    # print(xq_point[1], xq_point[0], p_j, p_i)
                    dis = geodesic((xq_point[1], xq_point[0]), (p_j, p_i)).km
                    # print(dis, type(dis))
                    if dis <= r:
                        # print("需求点({},{})位于候选点({},{})范围内".format(xq_point[0],xq_point[1],p_i,p_j))
                        if not flag:
                            candidate_p[(p_i, p_j)] = [xq_point]
                            candidate_p_re[(p_i, p_j)] = xq_point[2]
                            flag = True
                        else:
                            candidate_p[(p_i, p_j)].append(xq_point)
                            candidate_p_re[(p_i, p_j)] += xq_point[2]
                        # print((xq_point[1], xq_point[0], p_j, p_i))

        if candidate_p:
            find_cp = False
            while candidate_p and not find_cp:
                center_p = max(candidate_p_re.items(), key=lambda x: x[1])[0]  #tuple
                dis_flag = True

                for cp in selected_cp:
                    if not cp_dis(tuple(cp), center_p):
                        dis_flag = False
                        del candidate_p_re[center_p]
                        del candidate_p[center_p]
                        break

                if dis_flag:
                    selected_cp_contain_xq_num[center_p] = candidate_p[center_p]
                    for removed_p in candidate_p[center_p]:
                        S2.discard(removed_p)
                    print("添加({},{})为圆心".format(center_p[0],center_p[1]))
                    k += 1
                    selected_cp.append(list(center_p))
                    find_cp = True
                    candidate_p.clear()
                    candidate_p_re.clear()
                    
            if not find_cp:
                break
        else:
            break

    save_data(selected_cp, save_path, column_name=["经度","纬度"])
    cp_and_xq_num = list()
    for cp, xqs in selected_cp_contain_xq_num.items():
        for xq in xqs:
            cp_and_xq_num.append([cp[0], cp[1], len(xqs), xq[0], xq[1], xq[2]])
    
    df = pd.DataFrame(cp_and_xq_num, columns=["圆心经度", "圆心纬度", "圆内需求点数量", "需求点经度", "需求点纬度", "需求点需求量"])
    # df.to_excel("./data/圆内需求点.xlsx", index=False)
    df.to_excel(os.path.join(args.dir, "圆内需求点_{}_{}.xlsx".format(args.pre, args.fun)), index=False)
    return k



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="寻找圆心")
    parser.add_argument("--dir", default="./data", help="指定中间生成文件（包括圆心文件、圆内需求点文件、筛选后需求点文件）的保存文件夹，默认为data文件夹")
    parser.add_argument("--fun", choices=[1,2], type=int, default=2, help="选择实现算法，1为不考虑需求量，2为考虑，默认为2")
    parser.add_argument("--pre", choices=[1,2,3,4,5], type=int, default=2, help="选择精度，默认为1e-2")
    parser.add_argument("--fpre", choices=[1,2,3,4,5], type=int, default=5, help="不建议改动: 选择筛选需求点是否位于禁飞区内的精度，默认为1e-5, 精度越高，筛选掉的点越准确")
    parser.add_argument("--jf", default="./data/jinfeiqu.xlsx", help="读取禁飞点文件的路径，默认为'./data/jinfeiqu.xlsx'")
    parser.add_argument("--xq", default="./data/xuqiudian.xlsx", help="读取需求点文件的路径，默认为'./data/xuqiudian.xlsx'")
    args = parser.parse_args()

    # 首先筛选掉位于禁飞区内的需求点, 筛选需求点默认精度为5，如果需要，可以修改精度：
    # xq_path = in_no_fly(precision=args.fpre, jf_path=args.jf, xq_path=args.xq, save_path=args.fsave)
    xq_path = in_no_fly(precision=args.fpre, jf_path=args.jf, xq_path=args.xq, save_path=os.path.join(args.dir, "筛选后需求点_{}_{}.xlsx".format(args.pre, args.fun)))
    # print(xq_path)

    # 选择算法生成圆心
    if args.fun == 1:
        print("Processing Fun-1...")
        # k = fun1(args.pre, args.jf, xq_path, args.save)
        k = fun1(args.pre, args.jf, xq_path, os.path.join(args.dir, "圆心_{}_{}.xlsx".format(args.pre, args.fun)))

    elif args.fun == 2:
        print("Processing Fun-2...")
        k = fun2(args.pre, args.jf, xq_path, os.path.join(args.dir, "圆心_{}_{}.xlsx".format(args.pre, args.fun)))

    print("生成圆心个数为{}个".format(k))
