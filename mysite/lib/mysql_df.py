import pandas as pd
import pymysql


def read_df(table):
    db = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='', db='testdb', charset='utf8')
    cursor = db.cursor()
    sql = 'SELECT * FROM ' + table
    cursor.execute(sql)
    # 获得列名
    column = [col[0] for col in cursor.description]
    # 获得数据
    data = cursor.fetchall()
    # 获得DataFrame格式的数据
    data_df = pd.DataFrame(list(data), columns=column)
    cursor.close()
    db.close()

    return data_df
