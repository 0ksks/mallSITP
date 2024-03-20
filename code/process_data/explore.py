from sql_engine import *
import pandas as pd

tables = "会员主档资料 会员消费积分报表 会员积分业态表 会员积分明细报表 会员销售分析报表 客流".split()

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
    df = sample(tables[0],False,2)
    for col in df.columns:
        print(df[col].name,df[col].dtype,df[col][0])
    # describe = df.describe()
    # for col in describe.columns:
    #     print(describe[col])
    ...