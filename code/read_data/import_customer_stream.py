import pandas as pd
import os
import re
from warnings import filterwarnings
from path_to_data import path_to_data
filterwarnings('ignore')
def handle_one_record(path:str,year:int):
    df = pd.read_csv(path,encoding="gbk")
    df = df.drop(["平均","小计","位置名称"],axis=1)
    df = df.T
    df = df.reset_index()
    df = df.drop(0)
    weekdaysDict = "周一 周二 周三 周四 周五 周六 周日"
    weekdaysDict = {k:v for (k,v) in zip(weekdaysDict.split(),range(1,8))}
    extract_weekdays = lambda x:weekdaysDict[re.findall(r'\((.*?)\)', x)[0]]
    extract_dates = lambda x,year:f"{year}-"+re.findall(r'\d{2}-\d{2}', x)[0]
    index = df["index"]
    df = df.drop("index",axis=1)
    df.columns = ["进入客流","离开客流","滞留量"]
    df["星期"] = index.apply(extract_weekdays)
    df["日期"] = index.apply(lambda x:extract_dates(x,year))
    return df
def get_year_and_path(parentPath:str):
    yearList = os.listdir(parentPath)
    def get_files(parentPath:str)->str:
        fileList = os.listdir(parentPath)
        return fileList
    yearAndPath = []
    for year in yearList:
        try:
            eval(year)
            fileList = get_files(f"{parentPath}{year}/")
            for file in fileList:
                yearAndPath.append((year,f"{parentPath}{year}/{file}"))
        except:
            pass
    return yearAndPath
def get_dfs(parentPath:str)->pd.DataFrame:
    yearAndPath = get_year_and_path(parentPath)
    dfs = []
    for idx,(year,path) in enumerate(yearAndPath):
        print(f"{path}({idx+1}/{len(yearAndPath)})")
        dfs.append(handle_one_record(path,year))
    dfs:pd.DataFrame = pd.concat(dfs)
    return dfs
def main(engine):
    dfs = get_dfs(f"{path_to_data}/客流/")
    dfs = dfs.reset_index().drop("index",axis=1)
    dfs = dfs.astype(dict(zip(dfs.columns, "int16 int16 int16 int16 datetime64[ns]".split())))
    dfs.to_sql("客流", con=engine, if_exists='replace', index=False)
if __name__ == "__main__":
    from sql_engine import engine
    main(engine)
    engine.dispose()