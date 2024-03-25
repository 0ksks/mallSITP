from torch.utils.data import Dataset
import torch
from pandas import DataFrame
from typing import Literal
class Norm:
    def __init__(self,data:DataFrame) -> None:
        self.data = data
        self.mean = data.mean()
        self.std = data.std()
        self.norm = (self.data - self.mean)/self.std
    def denormalizer(self,input:DataFrame)->DataFrame:
        output = input*self.std+self.mean
        return output
    def normalizer(self,input:DataFrame)->DataFrame:
        norm = (input - self.mean)/self.std
        return norm
    

class AEDataset(Dataset):
    def __init__(self,type:Literal["raw","norm"]) -> None:
        from pandas import read_sql
        from sql_engine import engine
        SQL = "SELECT `加入日期`,`累计消费金额`,`性别`,`移动电话`,`当前积分`,`业态` FROM `训练_测试数据集`"
        print("querying")
        data = read_sql(SQL,con=engine)
        from datetime import datetime
        fixedDate = datetime.strptime("2018-01-01", "%Y-%m-%d").date()
        timeDelta = lambda date:(date-fixedDate).days
        data["加入日期"] = data["加入日期"].apply(timeDelta)
        self.type = type
        data = data.astype(float)
        if type=="raw":
            self.data = data.values
        elif type=="norm":
            normal = Norm(data)
            self.data = normal.norm.values
            self.denormalizer = normal.denormalizer
    def __len__(self)->int:
        return self.data.shape[0]
    def __getitem__(self, index):
        return torch.tensor(self.data[index],dtype=torch.float).view((-1,1,self.data.shape[1]))
    def denormalizer(self, input:DataFrame=None)->DataFrame:
        if self.type=="norm":
            if input is None:
                return self.denormalizer(self.data)
            return self.denormalizer(input)
        elif self.type=="raw":
            print(f"no need to denormalize, the data is raw")
            if input is None:
                return self.data
            return input
if __name__ == "__main__":
    aeDataset = AEDataset('norm')
    print(len(aeDataset))
    print(aeDataset[:2].shape)