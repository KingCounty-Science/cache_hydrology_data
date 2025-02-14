import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import configparser
from sqlalchemy import create_engine
import urllib
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

config = configparser.ConfigParser()
config.read('gdata_config.ini')

host_name = "KCITSQLPRNRPX01"
db_name = "gData"
server = "KCITSQLPRNRPX01"
server = "10.82.12.39"
driver = "SQL Server"



comparison_sites = configparser.ConfigParser()
comparison_sites.read('gdata_config.ini')


sql_alchemy_connection = urllib.parse.quote_plus('DRIVER={'+driver+'}; SERVER='+server+'; DATABASE='+db_name+'; USERNAME='+user+'; PASSWORD='+password+'; Trusted_Connection=yes;')
sql_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % sql_alchemy_connection)
conn = sql_engine.raw_connection()

sql = f"select SITE_CODE as site_number, G_ID as site_sql_id from {config['parameters']['parameter_table']} WHERE STATUS = 'Active' AND FlowLevel = 'True'"
site_list = pd.read_sql_query(sql, conn)
conn.close()
print(site_list)

def make_sql_engine():
    config = configparser.ConfigParser()
    config.read('gdata_config.ini')

    host_name = "KCITSQLPRNRPX01"
    db_name = "gData"
    server = "KCITSQLPRNRPX01"
    server = "10.82.12.39"
    driver = "SQL Server"
    sql_alchemy_connection = urllib.parse.quote_plus('DRIVER={'+driver+'}; SERVER='+server+'; DATABASE='+db_name+'; USERNAME='+user+'; PASSWORD='+password+'; Trusted_Connection=yes;')
    sql_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % sql_alchemy_connection)
    return sql_engine

def make_access_engine():
    config = configparser.ConfigParser()
    config.read('gdata_config.ini')
    # P:\JimBower\FRMP\CAO database\Backend
    sql_engine = create_engine("access:///?DataSource=P:/JimBower/FRMP/CAO database/Backend.accdb")

    
    #sql_alchemy_connection = urllib.parse.quote_plus('DRIVER={'+driver+'}; SERVER='+server+'; DATABASE='+db_name+'; USERNAME='+user+'; PASSWORD='+password+'; Trusted_Connection=yes;')
    #sql_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % sql_alchemy_connection)
    return sql_engine
'''
host_name = "ianrhigginshydro.mysql.pythonanywhere-services.com"

db_name = "ianrhigginshydro$ahydrologydatabase"
username = "ianrhigginshydro"
password = "4Q@2wertasdf"
driver = "mysql"

sql_engine = "mysql+mysqlconnector://{username}:{password}@{hostname}/{databasename}".format(
    username="ianrhigginshydro",
    password="4Q@2wertasdf",
    hostname="ianrhigginshydro.mysql.pythonanywhere-services.com",
    databasename="ianrhigginshydro$ahydrologydatabase",
)

sql_engine = create_engine("mysql+mysqlconnector://{username}:{password}@{hostname}/{databasename}".format(
    username="ianrhigginshydro",
    password="4Welwitchia",
    hostname="ianrhigginshydro.mysql.pythonanywhere-services.com",
    databasename="ianrhigginshydro$ahydrologydatabase",
))

Base = declarative_base()

sql = "CREATE TABLE Persons (PersonID int, LastName varchar(255), FirstName varchar(255), Address varchar(255), City varchar(255));"

class YourTable(Base):
    __tablename__ = 'your_table_name'

    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    age = Column(Integer)
    # Add more columns as needed

Base.metadata.create_all(sql_engine)

Session = sessionmaker(bind=sql_engine)
session = Session()

new_record = YourTable(name='John', age=25)
session.add(new_record)
session.commit()
'''
