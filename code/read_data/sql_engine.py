__all__ = ["engine"]
db_username = 'root'
db_password = '20030212'
db_host = 'localhost'
db_port = '3306'
db_name = '商场sitp'
print("import engine")
from sqlalchemy import create_engine
engine = create_engine(f'mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}')