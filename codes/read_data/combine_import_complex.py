import pandas as pd
import os
from warnings import filterwarnings
from path_to_data import path_to_data
filterwarnings('ignore')
# 会员积分业态表
def hui4yuan2ji1fen1ye4tai4biao3():
    files = os.listdir(f"{path_to_data}/会员积分业态表/")
    dfs = []
    for idx,file in enumerate(files):
        if "DS_Store" not in file:
            print(file,f"({idx+1}/{len(files)})")
            df = pd.read_excel(f"{path_to_data}/会员积分业态表/{file}")
            dfs.append(df)
    dfs:pd.DataFrame = pd.concat(dfs)
    dfs = dfs.reset_index().drop("index",axis=1)
    types = "datetime64[ns] category category category category int16 int16"
    dfs = dfs.astype(dict(zip(df.columns, types.split())))
    dfs.replace([float('inf'), float('-inf')], [9999, -9999], inplace=True)
    return dfs
# 会员积分明细报表
def hui4yuan2ji1fen1ming2xi4bao4biao3():
    path = path_to_data+"/会员积分明细报表/{}/"
    types = ["无感积分","手动积分"]
    dfs = []
    for type in types:
        path_ = path.format(type)
        files = os.listdir(path_)
        for idx,file in enumerate(files):
            if "DS_Store" not in file:
                print(f"{path_}{file}",f"({idx+1}/{len(files)})")
                df = pd.read_excel(f"{path_}{file}")
                df["积分方式"] = type
                dfs.append(df)
    dfs:pd.DataFrame = pd.concat(dfs)
    dfs = dfs.reset_index().drop("index",axis=1)
    types = "category category category datetime64[ns] timedelta64[ns] category category int16 float16 int16 category category category"
    dfs = dfs.astype(dict(zip(df.columns, types.split())))
    dfs.replace([float('inf'), float('-inf')], [9999, -9999], inplace=True)
    return dfs
def main(engine):
    hui4yuan2ji1fen1ye4tai4biao3().to_sql("会员积分业态表", con=engine, if_exists='replace', index=False)
    hui4yuan2ji1fen1ming2xi4bao4biao3().to_sql("会员积分明细报表", con=engine, if_exists='replace', index=False)
if __name__ == "__main__":
    from sql_engine import engine
    main(engine)
    engine.dispose()