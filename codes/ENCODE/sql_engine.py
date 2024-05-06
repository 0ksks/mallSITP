__all__ = ["engine"]
with open("codes/mysql.json","r") as f:
    import json
    config = json.loads(f.read())
dbURL = 'mysql+pymysql://{}:{}@{}:{}/{}'.format(*config.values())
'''
'mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
'''
print("import engine")
from sqlalchemy import create_engine
engine = create_engine(dbURL)