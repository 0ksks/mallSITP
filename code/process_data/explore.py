from sql_engine import *
import pandas as pd
import json
tables = "会员主档资料 会员消费积分报表 会员积分业态表 会员积分明细报表 会员销售分析报表 客流".split()
with open("code/dtypes.json","r") as f:
    dic:dict[str,str] = json.loads(f.read())
def sample(tableName:str, print_:bool=True, amount:int=1):
    query = f"SELECT * FROM {tableName} ORDER BY RAND() LIMIT {amount}"
    df = pd.read_sql(query, engine)
    if print_:
        print(tableName)
        print("="*20)
        if amount==1:
            print(df.iloc[0])
        else:
            print(df)
    return df

if __name__ == "__main__":
    table = tables[0]
    df = sample(table,False,2)
    df = df.astype(dict(zip(df.columns, dic[table].split())))
    data = []
    for col in df.columns:
        data.append(list(map(str,(df[col].name,df[col].dtype,df[col][0]))))
    from html_table import plot_table
    plot_table(data)
    ...