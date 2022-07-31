import numpy as np
import pandas as pd
import os

print(os.getcwd())
# xtest= np.array(pd.read_excel(r'../exl/锡槽0.7平衡2.xlsx', sheet_name='test'))
xtest= pd.read_excel(r'../exl/锡槽0.7平衡2.xlsx', sheet_name='test')
print(type(xtest))
print(xtest)
import pandas as pd
import pymysql
con = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='', db='testdb', charset='utf8')
cursor = con.cursor()
sql = 'SELECT * FROM test'
cursor.execute(sql)
# 获得列名
column = [col[0] for col in cursor.description]
# 获得数据
data = cursor.fetchall()
# 获得DataFrame格式的数据
data_df = pd.DataFrame(list(data), columns=column)


# 3 6 Sheet5  锡槽0.7平衡
