import pandas as pd
import os
from warnings import filterwarnings
from path_to_data import path_to_data
filterwarnings('ignore')

def combine(path:str)->pd.DataFrame:
    files = os.listdir(path)
    dfs = []
    for idx,file in enumerate(files):
        if "DS_Store" not in file:
            print(f"{file}({idx+1}/{len(files)})")
            df = pd.read_excel(path+file,engine='openpyxl')
            if "行数" in df.columns:
                df = df.drop("行数",axis=1)
        df["资料年份"]=file[-9:-5]
        dfs.append(df)
    dfs:pd.DataFrame = pd.concat(dfs)
    if "会员主档资料" in path:
        dfs["当前积分"].fillna(0, inplace=True)
        types = "category datetime64[ns] category category category category float16 category category datetime64[ns] float16 category category category category category category category category int16 category category category"
    elif "会员消费积分报表" in path:
        types = "category category datetime64[ns] float16 float16 category category category"
    else:
        dfs["年龄"].fillna(0, inplace=True)
        dfs["消费笔数 总消费额 积分兑换次数 积分余额".split()].fillna(0, inplace=True)
        types = "category datetime64[ns] datetime64[ns] category category category category category float16 category category category category float16 float16 float16 float16 category"
    dfs = dfs.astype(dict(zip(df.columns, types.split())))
    return dfs.reset_index().drop("index",axis=1)
def main(engine):
    tables = "会员主档资料 会员消费积分报表 会员销售分析报表".split()
    for table in tables:
        path = f"{path_to_data}/{table}/"
        df = combine(path)
        df.replace([float('inf'), float('-inf')], [9999, -9999], inplace=True)
        df.to_sql(table, con=engine, if_exists='replace', index=False)
if __name__ == "__main__":
    from sql_engine import engine
    main(engine)
    engine.dispose()
    