import json
import math

import pandas as pd

df = pd.read_excel("./data/手打.xls", usecols=[0],names=None,header=None)  # 读取项目名称列,不要列名

x = ""
print(df.values)
for idx,num in enumerate(df.values):
    if math.isnan(num):
        break
    else:
        x += "{\"label\": " + str(int(num)) + "}\n"
with open('./data/out.json', 'w') as f:
    f.write(x)