# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 10:02:36 2021

@author: IHiggins
"""

from datetime import datetime
from datetime import timedelta
import urllib
import configparser

import numpy as np
#import win32com.client as win32
#import schedule
import pyodbc
import pandas as pd
from sqlalchemy import create_engine
pd.options.mode.chained_assignment = None  # default='warn', None is no warn
config = configparser.ConfigParser()
# config.read(r'C:\Users\ihiggins\.spyder-py3\gdata_config.ini')
config.read('gdata_config.ini')
#print("Run Start at "+str(pd.to_datetime('today'))+"")
#print("")
#10.82.12.39
server = "KCITSQLPRNRPX01"
driver = "SQL Server"
database = "gData"
trusted_connection = "yes"
conn = pyodbc.connect('Driver={'+driver+'};'
                      'Server='+server+';'
                      'Database='+database+';'
                      'Trusted_Connection='+trusted_connection+';')

# object calling is different then object interpolation
# a variable defined before a function is global for function
#sql_alchemy_connection = urllib.parse.quote_plus('DRIVER={'+driver+'}; SERVER='+server+'; DATABASE='+database+', Trusted_Connection='+trusted_connection)
# DEV Server  KCITSQLDEVNRP01
# Data server KCITSQLPRNRPX01
#gage_lookup = pd.read_sql_query(
#    'select G_ID, SITE_CODE from tblGaugeLLID;', conn)

def clean_file(df, parameter, site_sql_id, utc_offset):
    '''Takes opened telemetry file and cleans data for processing'''

    #  Rename columns - note the value column is dynamic based on what is provided by the SQL we only read the two columns so this index method is fine
    df.rename(columns={df.columns[0]: "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["datetime"] = df["datetime"] + timedelta(hours=utc_offset)
    df.rename(columns={df.columns[1]: "data"}, inplace=True)
    # Round the 'value' column and convert data type
    df['data'] = pd.to_numeric(df['data'], errors='coerce').astype("float")
    df['corrected_data'] = pd.to_numeric(df['corrected_data'], errors='coerce').astype("float")
    df['G_ID'] = str(site_sql_id)
    df.drop_duplicates(subset=["datetime"], inplace=True)
    return df

def delete_data(df, parameter, site_sql_id):
    # data has to be sorted properly for "start date" to be first row
    df = df.sort_values(by='datetime', ascending=True)
    start_date = df.head(1).iloc[0, df.columns.get_loc("datetime")] # -  timedelta(seconds=1)
    end_date = df.tail(1).iloc[0, df.columns.get_loc("datetime")] # +  timedelta(seconds=1)
    #print(df)
    #print(df.dtypes)
    #end_date = df.iloc[0,0] +  timedelta(seconds=61)

    # pull old data + 1 day
    #start_date = df.iloc[-1,0] -  timedelta(seconds=61)
    #print("start date", start_date, "end_date", end_date)
    # new_data = pd.read_sql_query('select '+config[parameter]['datetime']+','+config[parameter]['corrected_data']+','+config[parameter]['discharge']+' from '+config[parameter]['table']+' WHERE G_ID = '+str(site_sql_id)+' AND '+config[parameter]['datetime']+' between ? and ?', conn, params=[str(start_date), str(end_date)])
    # Delete existing data for time peroid in question
    conn.execute(f"delete from {config[parameter]['table']} WHERE G_ID = {site_sql_id} AND {config[parameter]['datetime']} between ? and ?", start_date.strftime('%m/%d/%Y %H:%M'), end_date.strftime('%m/%d/%Y %H:%M'))
    conn.commit()

def upload_data(df, parameter, site_sql_id, utc_offset):
    '''takes data from cut_data, formats it for server and uploads
    different parameters will call different functions for individual
    sql tables as  defined in below if(parameter) statements'''
    # This is from dash
    # df = pd.DataFrame(rows)
    # changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]\\
    #df = df.copy()
    def discharge_column():
        df.rename(columns={"discharge": config[parameter]['discharge']}, inplace=True)
        return df
    
    def est_column():
        df.rename(columns={"estimate": config[parameter]['est']}, inplace=True)
        df[config[parameter]['est']] = df[config[parameter]['est']].astype(bool)
        
        return df

    def lock_column():
        df[config[parameter]['lock']] = "0"
        return df
    
    def depth_column():
        df[config[parameter]['depth']] = "0"
        return df
    
    def ice_column():
        df[config[parameter]['ice']] = "0"
        return df
    def warning_column():
        #df[config[parameter]['warning']] = "0"
        df.rename(columns={"warning": config[parameter]['warning']}, inplace=True)
        df[config[parameter]['warning']] = df[config[parameter]['warning']].astype(bool)
        return df

    def provisional_column():
        df[config[parameter]['provisional']] = "0"
        return df

    def gallons_pumped_column():
        df[config[parameter]['amount_pumped']] = "0"
        return df

    def pump_on_column():
        df[config[parameter]['pump_state']] = "0"
        return df

    def auto_timestamp_column():
        # time_now = pd.to_datetime('today')
        df[config[parameter]['auto_timestamp']] = pd.to_datetime('today')
        df[config[parameter]['auto_timestamp']] = df[config[parameter]['auto_timestamp']].dt.strftime('%m/%d/%Y %H:%M')
        return df

    def snow_column():
        df[config[parameter]['snow']] = "0"
        return df

    def utc_offset_column(utc_offset):
        df[config[parameter]['utc_offset']] = str(utc_offset)
        return df

    def site_id(site_sql_id):
        df["G_ID"] = str(site_sql_id)
        return df

    def sql_time():
        df[config[parameter]['datetime']] = df[config[parameter]['datetime']]
        #df[config[parameter]['datetime']] = df[config[parameter]['datetime']] + timedelta(hours=(utc_offset).astype(float))
        df[config[parameter]['datetime']] = df[config[parameter]['datetime']].dt.strftime('%m/%d/%Y %H:%M:%S')
        return df

    # The actual data upload
    def upload(df):
        # engine is the lowest level object in sqlalchemy, it maintains a pool of connections available for use whenever application needs them
        # engine pooling...the connection has a pool https://docs.sqlalchemy.org/en/20/core/pooling.html#sqlalchemy.pool.QueuePool

        # pyodbc has a longer pooling then sql_alchemy and needs to be reset
        pyodbc.pooling = False
        # not sure this fast execumetry sped things up
        # info on host name connection https://docs.sqlalchemy.org/en/14/dialects/mssql.html#connecting-to-pyodbc
        sql_alchemy_connection = urllib.parse.quote_plus('DRIVER={'+driver+'}; SERVER='+server+'; DATABASE='+database+'; Trusted_Connection='+trusted_connection+';')
        sql_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % sql_alchemy_connection)

        #sql_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % sql_alchemy_connection, fast_executemany=True)

         # connection is that thing that actually does the work of eqecuting an sql query
        # engine.execute() is a convience way of opening a connection, running a command, and closing
        # with to_sql you need to open a connection
        # raw connection bypasses some proxy issues with pooling # https://docs.sqlalchemy.org/en/14/core/connections.html#sqlalchemy.engine.Engine.raw_connection

        cnxn = sql_engine.raw_connection()
   
        #cnxn = sql_engine.connect()
        # setting chunksize should speed things up, there may be a dynamic approperate chunk size based on row count
        # you should be able to add a returning= but I cant figure it out
        # to_sql should be faster, there is also a read option https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-sql-method

        # a sqlalchemy.engine.connection can be passed to conn with engine.begin() as connection:
       # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html
       #with engine.begin() as connection:
        #df1 = pd.DataFrame({'name' : ['User 4', 'User 5']})
        #df1.to_sql('users', con=connection, if_exists='append')
        #   2
       
        df.to_sql(config[parameter]['table'], sql_engine, method=None, if_exists='append', index=False, chunksize=1000, )
       

        
        cnxn.close()
        # but not this returning statement.returning(config[parameter]['table'].config[parameter]['datetime'])
        # when application needs to talk to database .execute() is a convience method that calls conn conn = engine.connect(close_with_result=True)
        #cnxn = engine.raw_connection()
         # connection is that thing that actually does the work of eqecuting an sql query
    
        #for row in cnxn:
            #print(cnxn[config[parameter]['datetime']])
       
        #result = engine.execute('SELECT * FROM tablename;')
        ##what engine.execute() is doing under the hood
        #conn = engine.connect(close_with_result=True)
        #result = conn.execute('SELECT * FROM tablename;')

        ##after you iterate over the results, the result and connection get closed
        #for row in result:
        #    print(result['columnname']

        #or you can explicitly close the result, which also closes the connection
       # result.close()
    

        # connection is that thing that actually does the work of eqecuting an sql query
        #connection = engine.connect()
        #trans = connection.begin()
        #try:
        #connection.execute("INSERT INTO films VALUES ('Comedy', '82 minutes');")
        #connection.execute("INSERT INTO datalog VALUES ('added a comedy');")
        #trans.commit()
        #except:
        #trans.rollback()
        #raise

        #sql_alchemy_connection = urllib.parse.quote_plus(f"DRIVER={{driver}}; SERVER={server}; DATABASE={database}, Trusted_Connection={trusted_connection}")
        #result = engine.execute('SELECT * FROM tablename;')
        #what engine.execute() is doing under the hood
        #   conn = engine.connect(close_with_result=True)
        #result = conn.execute('SELECT * FROM tablename;')

        #after you iterate over the results, the result and connection get closed
        #for row in result:
    #print(result['columnname']

        #or you can explicitly close the result, which also closes the connection
#result.close()
        #### THIS IS THE WORKING CODE
        #sql_alchemy_connection = urllib.parse.quote_plus('DRIVER={'+driver+'}; SERVER='+server+'; DATABASE='+database+'; Trusted_Connection='+trusted_connection+';')
        #sql_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % sql_alchemy_connection)
        #cnxn = sql_engine.raw_connection()
        #
        #df.to_sql(config[parameter]['table'], sql_engine, method=None, if_exists='append', index=False)
        # but not this returning statement.returning(config[parameter]['table'].config[parameter]['datetime'])

        # try method=multi, None works
        # try chunksize int
        #cnxn.close()
        
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce', infer_datetime_format=True)
    # df["G_ID"] = str(site_sql_id)
    df.rename(columns={"datetime": config[parameter]['datetime']}, inplace=True)
    df.rename(columns={"data": config[parameter]['data']}, inplace=True)
    df.rename(columns={"corrected_data": config[parameter]['corrected_data']}, inplace=True)

    if parameter == "air_temperature":
        df = auto_timestamp_column()
        df = est_column()
        df = lock_column()
        df = provisional_column()
        df = site_id(site_sql_id)
        df = utc_offset_column(utc_offset)
        df = warning_column()
        # ONLY USE THIS FOR SQL IMPORT IT ADDS & HOURS
        df = sql_time()
        upload(df)
    
    if parameter == "water_temperature":
        df = auto_timestamp_column()
        df = est_column()
        df = ice_column()
        df = depth_column()
        df = provisional_column()
        df = site_id(site_sql_id)
        df = utc_offset_column(utc_offset)
        # ONLY USE THIS FOR SQL IMPORT IT ADDS & HOURS
        df = sql_time()
        upload(df)

    if parameter == "barometer":
        df = auto_timestamp_column()
        df = est_column()
        df = lock_column()
        df = provisional_column()
        df = site_id(site_sql_id)
        df = utc_offset_column(utc_offset)
        df = warning_column()
        # ONLY USE THIS FOR SQL IMPORT IT ADDS & HOURS
        df = sql_time()
        upload(df)

    if parameter == "Conductivity"or parameter == "conductivity":
        df = auto_timestamp_column()
        df = est_column()
        df = lock_column()
        df = site_id(site_sql_id)
        df = utc_offset_column(utc_offset)
        df = warning_column()
        # ONLY USE THIS FOR SQL IMPORT IT ADDS & HOURS
        df = sql_time()
        upload(df)
        
    if parameter == "FlowLevel":
        df = auto_timestamp_column()
        df = discharge_column()
        df = est_column()
        df = lock_column()
        df = provisional_column()
        df = site_id(site_sql_id)
        # df = utc_offset_column(utc_offset)
        # df = warning_column()

    if parameter == "discharge":
        # print(df)
        df = auto_timestamp_column()
        df = discharge_column()
        df = est_column()
        df = lock_column()
        df = provisional_column()

        df = utc_offset_column(utc_offset)
        df = warning_column()
        df = provisional_column()
        df = site_id(site_sql_id)
        # ONLY USE THIS FOR SQL IMPORT IT ADDS & HOURS
        df = sql_time()

        #df.drop(columns=['stage', 'water_level'], inplace=True)
        upload(df)

    if parameter == "water_level":
       
        df = auto_timestamp_column()
        df = est_column()
        df = lock_column()
        df = provisional_column()
        df = site_id(site_sql_id)
        df = utc_offset_column(utc_offset)
        df = warning_column()
        # ONLY USE THIS FOR SQL IMPORT IT ADDS & HOURS
        df = sql_time()
        
        upload(df)

    if parameter == 'groundwater_level':
        df = auto_timestamp_column()
        df = est_column()
        df = gallons_pumped_column()
        df = lock_column()
        df = provisional_column()
        df = pump_on_column()
        df = site_id(site_sql_id)
        df = utc_offset_column(utc_offset)
        df = warning_column()
        # ONLY USE THIS FOR SQL IMPORT IT ADDS & HOURS
        df = sql_time()
        upload(df)

    if parameter == "rain":
        df = auto_timestamp_column()
        df = est_column()
        df = lock_column()
        df = provisional_column()
        df = site_id(site_sql_id)
        df = snow_column()
        df = utc_offset_column(utc_offset)
        df = warning_column()
        # ONLY USE THIS FOR SQL IMPORT IT ADDS & HOURS
        df = sql_time()
        upload(df)

    if parameter == "rain_tips":
        df = auto_timestamp_column()
        df = est_column()
        df = lock_column()
        df = provisional_column()
        df = site_id(site_sql_id)
        df = snow_column()
        df = utc_offset_column(utc_offset)
        df = warning_column()
        # ONLY USE THIS FOR SQL IMPORT IT ADDS & HOURS
        df = sql_time()
        upload(df)

    if parameter == "turbidity":
        df = auto_timestamp_column()
        df = est_column()
        df = lock_column()
        df = provisional_column()
        df = site_id(site_sql_id)
        df = utc_offset_column(utc_offset)
        df = warning_column()
        # ONLY USE THIS FOR SQL IMPORT IT ADDS & HOURS
        df = sql_time()
        upload(df)

    if parameter == "battery":
        df = site_id(site_sql_id)
        df = sql_time()
        df[config[parameter]['datetime']] = pd.to_datetime(df[config[parameter]['datetime']], format='%Y-%m-%d %H:%M:%S', errors='coerce', infer_datetime_format=True)
        df[config[parameter]['datetime']] = df[config[parameter]['datetime']].dt.strftime('%m/%d/%Y')
        upload(df)

    return df

def daily_table(parameter, site_sql_id, utc_offset):
    '''updates daily table, regardless of weither data was uploaded
    behaves similarly to discharge upload function'''
    # get 15 minute data last value
    # agnostic to actual interval
   
    cursor = conn.cursor()
    existing_data = cursor.execute("select max("+str(config[parameter]['datetime'])+") from "+str(config[parameter]['table'])+" WHERE G_ID = "+str(site_sql_id)+";").fetchval().date()
    cursor.close()
    # get daily table last value
    try:
        cursor = conn.cursor()
        existing_daily_data = cursor.execute("select max("+str(config[parameter]['daily_datetime'])+") from "+str(config[parameter]['daily_table'])+" WHERE G_ID = "+str(site_sql_id)+";").fetchval().date()
        cursor.close()
    except AttributeError:
        # if there is no data present
        existing_daily_data = datetime.strptime("1900-1-1", '%Y-%m-%d').date()
        # def discharge_column():
            # df.rename(columns={"discharge": config[parameter]['discharge']}, inplace=True)
            # return df
    
    def est_column():
        data[config[parameter]["daily_estimate"]] = "0"
        return data
    def depth_column():
        data[config[parameter]["daily_depth"]] = "0"
        return data

    def ice_column():
        data[config[parameter]["daily_ice"]] = "0"
        return data
    
    def lock_column():
        data[config[parameter]["daily_lock"]] = "0"
        return data

    def warning_column():
        data[config[parameter]["daily_warning"]] = "0"
        return data

    def provisional_column():
        data[config[parameter]['daily_provisional']] = "0"
        return data

    def gallons_pumped_column():
        data[config[parameter]['gallons_pumped']] = ""
        return data

    def pump_on_column():
        data[config[parameter]['pump_on']] = "0"
        return data

    def auto_timestamp_column():
        # time_now = pd.to_datetime('today')
        data[config[parameter]['daily_auto_timestamp']] = pd.to_datetime('today')
        data[config[parameter]['daily_auto_timestamp']] = data[config[parameter]['daily_auto_timestamp']].dt.strftime('%m/%d/%Y %H:%M')
        data[config[parameter]["daily_provisional"]] = "-1"
        return data

    def utc_offset_column(utc_offset):
        data[config[parameter]['utc_offset']] = str(utc_offset)
        return data

    def snow_column():
        data[config[parameter]['daily_snow']] = "0"
        return data 
        
    def site_id(site_sql_id):
        data["G_ID"] = str(site_sql_id)
        return data

    def sql_time(utc_offset):

        data[config[parameter]['datetime']] = data[config[parameter]['datetime']].dt.strftime('%m/%d/%Y')
        data[config[parameter]["daily_datetime"]] = data[config[parameter]['datetime']]
        return data

    # The actual daily data upload
    def daily_upload(data):
        sql_alchemy_connection = urllib.parse.quote_plus('DRIVER={'+driver+'}; SERVER='+server+'; DATABASE='+database+'; Trusted_Connection='+trusted_connection+';')
        sql_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % sql_alchemy_connection)
        cnxn = sql_engine.raw_connection()
        data.to_sql(config[parameter]['daily_table'], sql_engine, method=None, if_exists='append', index=False)
        # try method=multi, None works
        # try chunksize int

        cnxn.close()

    # if the daily table needs updating

    if existing_daily_data < existing_data:
        end_date = existing_data
        # pull old data + 1 day
        start_date = existing_daily_data - timedelta(days=2)
        # new_data = pd.read_sql_query('select '+config[parameter]['datetime']+','+config[parameter]['corrected_data']+','+config[parameter]['discharge']+' from '+config[parameter]['table']+' WHERE G_ID = '+str(site_sql_id)+' AND '+config[parameter]['datetime']+' between ? and ?', conn, params=[str(start_date), str(end_date)])
        # Delete existing data for time peroid in question
        conn.execute(f"delete from {config[parameter]['daily_table']} WHERE G_ID = {site_sql_id} AND {config[parameter]['daily_datetime']} between ? and ?", start_date.strftime('%m/%d/%Y'), end_date.strftime('%m/%d/%Y'))
        conn.commit()
        try:
            new_data = pd.read_sql_query('select '+config[parameter]['datetime']+','+config[parameter]['corrected_data']+','+config[parameter]['discharge']+' from '+config[parameter]['table']+' WHERE G_ID = '+str(site_sql_id)+' AND '+config[parameter]['datetime']+' between ? and ?', conn, params=[str(start_date), str(end_date)])
            new_data.rename(columns={
                config[parameter]['datetime']: "datetime",
                config[parameter]['corrected_data']: "corrected_data",
                config[parameter]['discharge']: "discharge",
            }, inplace=True)
        except:
            new_data = pd.read_sql_query('select '+config[parameter]['datetime']+','+config[parameter]['corrected_data']+' from '+config[parameter]['table']+' WHERE G_ID = '+str(site_sql_id)+' AND '+config[parameter]['datetime']+' between ? and ?', conn, params=[str(start_date), str(end_date)])
            new_data.rename(columns={
                config[parameter]['datetime']: "datetime",
                config[parameter]['corrected_data']: "corrected_data",
            }, inplace=True)
        if parameter == "rain":
            # resample 15 minute to daily
            new_data.set_index('datetime', inplace=True)
            corrected_data = new_data.resample('D')['corrected_data'].agg(['sum', 'count'])
            corrected_data.reset_index(inplace=True)
            # corrected_data = corrected_data[["datetime":config[parameter]['datetime'], "mean":config[parameter]['corrected_data_mean'], "max":config[parameter]['corrected_data_max'], "min":config[parameter]['D_MinStage'], "count":config[parameter]['daily_record_count']]].copy
            corrected_data.rename(columns={
                "datetime": config[parameter]["datetime"],
                "sum": config[parameter]["daily_sum"],
                "count": config[parameter]["daily_record_count"],
            }, inplace=True)
        else:
            # resample 15 minute to daily
            new_data.set_index('datetime', inplace=True)
            corrected_data = new_data.resample('D')['corrected_data'].agg(['mean', 'max', 'min', 'count'])
            corrected_data.reset_index(inplace=True)
            # corrected_data = corrected_data[["datetime":config[parameter]['datetime'], "mean":config[parameter]['corrected_data_mean'], "max":config[parameter]['corrected_data_max'], "min":config[parameter]['D_MinStage'], "count":config[parameter]['daily_record_count']]].copy
            corrected_data.rename(columns={
                "datetime": config[parameter]["datetime"],
                "mean": config[parameter]["daily_mean"],
                "max": config[parameter]["daily_max"],
                "min": config[parameter]["daily_min"],
                "count": config[parameter]["daily_record_count"],
            }, inplace=True)

        if parameter == "air_temperature":
            data = corrected_data
            # add other columns
            data = auto_timestamp_column()
            # df = discharge_column()
            data = est_column()
            data = lock_column()
            data = provisional_column()
            data = utc_offset_column(utc_offset)
            data = warning_column()
            data = site_id(site_sql_id)
            # ONLY USE THIS FOR SQL IMPORT IT ADDS & HOURS
            data = sql_time(utc_offset)
            # drop columns
            data.drop(columns=[config[parameter]["datetime"], config[parameter]["utc_offset"]], inplace=True)
            daily_upload(data)
            
        if parameter == "water_temperature":
            data = corrected_data
            # add other columns
            data = auto_timestamp_column()
            # df = discharge_column()
            data = est_column()
            data = depth_column()
            data = ice_column()
            data = lock_column()
            data = provisional_column()
            data = utc_offset_column(utc_offset)
            data = warning_column()
            data = site_id(site_sql_id)
            # ONLY USE THIS FOR SQL IMPORT IT ADDS & HOURS
            data = sql_time(utc_offset)
            # drop columns
            data.drop(columns=[config[parameter]["datetime"], config[parameter]["utc_offset"]], inplace=True)
            daily_upload(data)
        if parameter == "barometer":
            data = corrected_data
            # add other columns
            data = auto_timestamp_column()
            # df = discharge_column()
            data = est_column()
            data = lock_column()
            data = provisional_column()
            data = utc_offset_column(utc_offset)
            data = warning_column()
            data = site_id(site_sql_id)
            # ONLY USE THIS FOR SQL IMPORT IT ADDS & HOURS
            data = sql_time(utc_offset)
            # drop columns
            data.drop(columns=[config[parameter]["datetime"], config[parameter]["utc_offset"]], inplace=True)
            daily_upload(data)

        if parameter == "discharge":
            discharge = new_data.resample('D')['discharge'].agg(['mean', 'max', 'min'])
            discharge.reset_index(inplace=True)
            discharge.rename(columns={
                "datetime": config[parameter]["datetime"],
                "mean": config[parameter]["discharge_mean"],
                "max": config[parameter]["discharge_max"],
                "min": config[parameter]["discharge_min"],
            }, inplace=True)
            data = corrected_data.merge(discharge, left_on=config[parameter]["datetime"], right_on=config[parameter]["datetime"])
            # add other columns
            data = auto_timestamp_column()
            # df = discharge_column()
            data = est_column()
            data = lock_column()
            data = provisional_column()
            data = utc_offset_column(utc_offset)
            data = warning_column()
            data = site_id(site_sql_id)
            # ONLY USE THIS FOR SQL IMPORT IT ADDS & HOURS
            data = sql_time(utc_offset)
            # drop columns
            data.drop(columns=[config[parameter]["datetime"], config[parameter]["utc_offset"]], inplace=True)
            daily_upload(data)

        if parameter == "water_level":
            data = corrected_data
            # add other columns
            data = auto_timestamp_column()
            # df = discharge_column()
            data = est_column()
            data = lock_column()
            data = provisional_column()
            data = utc_offset_column(utc_offset)
            data = warning_column()
            data = site_id(site_sql_id)
            # ONLY USE THIS FOR SQL IMPORT IT ADDS & HOURS
            data = sql_time(utc_offset)
            # drop columns
            data.drop(columns=[config[parameter]["datetime"], config[parameter]["utc_offset"]], inplace=True)
            daily_upload(data)
                
        if parameter == "rain":
            data = corrected_data
            # add other columns
            data = auto_timestamp_column()
            # df = discharge_column()
            data = est_column()
            data = snow_column()
            data = lock_column()
            data = provisional_column()
            data = utc_offset_column(utc_offset)
            data = warning_column()
            data = site_id(site_sql_id)
            # ONLY USE THIS FOR SQL IMPORT IT ADDS & HOURS
            data = sql_time(utc_offset)
            # drop columns
            data.drop(columns=[config[parameter]["datetime"], config[parameter]["utc_offset"]], inplace=True)
            daily_upload(data)

        if parameter == "turbidity":
            data = corrected_data
            # add other columns
            data = auto_timestamp_column()
            # df = discharge_column()
            data = est_column()
            data = lock_column()
            data = provisional_column()
            data = utc_offset_column(utc_offset)
            data = warning_column()
            data = site_id(site_sql_id)
            # ONLY USE THIS FOR SQL IMPORT IT ADDS & HOURS
            data = sql_time(utc_offset)
            # drop columns
            data.drop(columns=[config[parameter]["datetime"], config[parameter]["utc_offset"]], inplace=True)
            daily_upload(data)
    # if the daily table does not need updating
    else:
        # return an empty data frame, a bit hacky but it prevents needless blank sql inserts later
        data = []
        data = pd.DataFrame(data, columns=[])


def full_upload(df, parameter, site_sql_id, utc_offset):
    print("full upload")
    clean_file(df, parameter, site_sql_id, utc_offset)
    print("cleaned")
    delete_data(df, parameter, site_sql_id)
    print("delete")
    upload_data(df, parameter, site_sql_id, utc_offset)
    print("uplad")
    daily_table(parameter, site_sql_id, utc_offset)
    print("daily")
#
#def manual_upload():
#    #parameter_upload_data = pd.read_csv(r"W:\STS\hydro\GAUGE\Temp\Ian's Temp\output.csv")
#    #parameter_upload_data = pd.read_csv(r"C:\Users\ihiggins\Documents\raw.csv")
#    df = pd.read_csv(r"C:\Users\ihiggins\OneDrive - King County\cache_upload\COS_Site02_water_temperature_2022_02_08.csv")
#
#    #parameter_upload_data.rename(columns={"Water_Level_ft": "data"}, inplace=True)
#    #parameter_upload_data.drop(columns=['Estimate'], inplace=True)
#    parameter = "water_temperature"
#    df = df.rename(columns={config[parameter]["datetime"]: "datetime", config[parameter]["data"]: "data", config[parameter]["corrected_data"]: "corrected_data"})
#    df = df[["datetime", "data", "corrected_data"]]
#    offset = 0
#
#    site_name = "wl1509w"
#    site_sql_id = 1899
#    utc_offset = 7
#
#    clean_file(df, parameter, site_sql_id, utc_offset)
#    print(df)
#    delete_data(df, parameter, site_sql_id)
#    upload_data(df, parameter, site_sql_id, utc_offset)
#    daily_table(parameter, site_sql_id, utc_offset)
#
#manual_upload()



# clean_file()
# takes data as parameter_upload_data, parameter, site_name
# delete_data()
# upload_data
