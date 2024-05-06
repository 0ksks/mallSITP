from sql_engine import engine
import pandas as pd
def get_incomplete()->pd.DataFrame:
    print("querying")
    SQL = "SELECT `会员卡号`,`加入日期`,`累计消费金额`,`性别`,`移动电话`,`当前积分`,`业态` FROM `待生成数据集_pk`" 
    data = pd.read_sql(SQL,con=engine)
    from datetime import datetime
    fixedDate = datetime.strptime("2018-01-01", "%Y-%m-%d").date()
    timeDelta = lambda date_string:(datetime.strptime(date_string, "%d/%m/%Y %H:%M:%S").date()-fixedDate).days
    data["加入日期"] = data["加入日期"].apply(timeDelta)
    data.iloc[:,1:] = data.iloc[:,1:].astype(float)
    return data
def get_complete()->pd.DataFrame:
    print("querying")
    SQL = "SELECT `加入日期`,`累计消费金额`,`性别`,`移动电话`,`当前积分`,`业态` FROM `训练_测试数据集`"
    data = pd.read_sql(SQL,con=engine)
    from datetime import datetime
    fixedDate = datetime.strptime("2018-01-01", "%Y-%m-%d").date()
    timeDelta = lambda date:(date-fixedDate).days
    data["加入日期"] = data["加入日期"].apply(timeDelta)
    data = data.astype(float)
    return data
def binarycode(df:pd.DataFrame)->pd.DataFrame:
    pk = df.iloc[:,0]
    df = df.iloc[:,1:]
    dfna = df.isna().values
    code = []
    for row_i in range(dfna.shape[0]):
        binarycode_ = 0
        for col_i in range(dfna.shape[1]):
            if dfna[row_i][col_i]:
                binarycode_ += pow(2,col_i)
        code.append(binarycode_)
    df["type"] = code
    df["pk"] = pk
    return df
def get_uni_inc()->pd.DataFrame:
    rawData = get_complete()
    df = binarycode(get_incomplete())
    df_re = df[df["type"]==32].drop("type",axis=1)
    df_re_ = (df_re.iloc[:,:-1] - rawData.mean())/rawData.std()
    df_re_["PK"] = df_re.iloc[:,-1]
    return df_re_
def get_bi_inc()->pd.DataFrame:
    rawData = get_complete()
    df = binarycode(get_incomplete())
    df_re = df[df["type"]==34].drop("type",axis=1)
    df_re_ = (df_re.iloc[:,:-1] - rawData.mean())/rawData.std()
    df_re_["PK"] = df_re.iloc[:,-1]
    return df_re_
if __name__ == "__main__":
    print(get_bi_inc())