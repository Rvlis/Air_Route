import pandas as pd
from geopy.distance import geodesic
from decimal import Decimal

def get_dis(data_path):
    longest_dis = 0.0
    shortest_dis = geodesic((31.50, 122.00), (30.70, 121.00)).km
    cps = pd.read_excel(data_path).values
    cps = list(cps)
    for i in range(len(cps)):
        for j in range(i+1, len(cps)):
            dis = geodesic((cps[i][1], cps[i][0]), (cps[j][1], cps[j][0])).km
            if dis > longest_dis:
                longest_dis = dis
            if dis < shortest_dis:
                shortest_dis = dis
    return longest_dis, shortest_dis


longest_dis, shortest_dis = get_dis("./3-1/yuanxin_3_1.xlsx")
print(float(Decimal(longest_dis).quantize(Decimal("0.000"))), float(Decimal(shortest_dis).quantize(Decimal("0.000"))))