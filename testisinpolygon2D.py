"""判断圆心是否跨越禁飞区
"""

from isinpolygon2D import isin_multipolygon
import pandas as pd
from decimal import Decimal

def read_data(jf_path, xq_path):
    jf_lst = list()  #禁飞点集合
    xq_lst = list()  #需求点集合

    jf_data = pd.read_excel(jf_path, sheet_name=0)
    nrows = jf_data.shape[0]
    for row in range(nrows):
        # print(float(require_data.iloc[row, 1]), float(require_data.iloc[row, 2]))
        longitude = jf_data.iloc[row, 2]
        latitude = jf_data.iloc[row, 1]
        longitude = int(float(Decimal(longitude).quantize(Decimal("0.000000")))*1000000)
        latitude = int(float(Decimal(latitude).quantize(Decimal("0.000000")))*1000000)
        # S1.add((float(jf_data.iloc[row, 1]), float(jf_data.iloc[row, 2])))
        jf_lst.append([longitude, latitude])


    xq_data = pd.read_excel(xq_path, sheet_name=0)
    nrows = xq_data.shape[0]
    for row in range(nrows):
        longitude = xq_data.iloc[row, 1]
        latitude = xq_data.iloc[row, 2]
        longitude = int(float(Decimal(longitude).quantize(Decimal("0.000000")))*1000000)
        latitude = int(float(Decimal(latitude).quantize(Decimal("0.000000")))*1000000)
        xq_lst.append([longitude, latitude])
    
    return jf_lst, xq_lst


jf_lst, xq_lst = read_data("./jinfeiqu.xlsx", "./xuqiudian.xlsx")
jf_lst.append(jf_lst[0])

id = 1
for xq in xq_lst:
    if isin_multipolygon(xq, jf_lst, contain_boundary=True):
        print(id)
    id += 1


