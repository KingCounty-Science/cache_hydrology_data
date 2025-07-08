# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 10:02:36 2021

@author: IHiggins
"""

from datetime import datetime
from datetime import timedelta
import urllib
import configparser
from sqlalchemy.exc import IntegrityError
import numpy as np
#import win32com.client as win32
#import schedule
import pyodbc
import pandas as pd
from sqlalchemy import create_engine, text
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

    def groundwater_temperature_column():
        df[config[parameter]['groundwater_temperature']] = np.nan
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
        df = warning_column()
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
        df = lock_column()
        df = provisional_column()
        df = groundwater_temperature_column()
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



def calculate_daily_values(start_date, end_date, parameter, site_sql_id):
    sql_alchemy_connection = urllib.parse.quote_plus('DRIVER={'+driver+'}; SERVER='+server+'; DATABASE='+database+'; Trusted_Connection='+trusted_connection+';')
    sql_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % sql_alchemy_connection)
    """takes a date range from either daily insufficient data or missing data returns a df calculated daily values"""
    daily_data = []
    if start_date:
        # standard columns
        derived_mean = "" # used for discharge
        derived_max = ""    # used for discharge
        derived_min = ""    # used for discharge

        daily_sum = ""

        provisional = f"MAX( CAST({config[parameter]['provisional']} AS INT)) AS {config[parameter]['daily_provisional']}, " # standard provisional
        snow = "" # used for rain

        depth = "" # used for rain
        ice = "" # used for rain
        daily_sum = "" # used for rain

        # parameter specific specially formatted columns
        if parameter == "discharge":
                derived_mean = f"ROUND(AVG({config[parameter]['discharge']}), 2) AS {config[parameter]['discharge_mean']}, "
                derived_max = f"ROUND(MAX({config[parameter]['discharge']}), 2) AS {config[parameter]['discharge_max']}, "
                derived_min = f"ROUND(MAX({config[parameter]['discharge']}), 2) AS {config[parameter]['discharge_min']}, "
                
        if parameter == "conductivity":
                provisional = ""

        if parameter == "rain": # this progrma will calculate min/mix/avg but those columns will be removed by column managment
                daily_sum = f"ROUND(SUM({config[parameter]['corrected_data']}), 2) AS {config[parameter]['daily_sum']}, "
                snow =  f"MAX( CAST({config[parameter]['snow']} AS INT)) AS {config[parameter]['daily_snow']}, "
    
        if parameter == "water_temperature":
                depth = f"ROUND( AVG( CAST({config[parameter]['ice']} AS INT) ) , 2) AS {config[parameter]['daily_depth']}, "
                ice =  f"MAX( CAST({config[parameter]['ice']} AS INT)) AS {config[parameter]['daily_ice']}, "
        groundwater_temperature = ""
        if parameter == "groundwater_level":
                groundwater_temperature =  f"ROUND(AVG({config[parameter]['groundwater_temperature']}), 2) AS {config[parameter]['groundwater_temperature']}, "
        with sql_engine.begin() as conn:
                # 120 is yyyy-mm-dd hh:mi:ss
                # 105 dd-mm-yyyy
                #new_data = pd.read_sql_query('select '+config[parameter]['datetime']+','+config[parameter]['corrected_data']+' from '+config[parameter]['table']+' WHERE G_ID = '+str(site_sql_id)+' AND '+config[parameter]['datetime']+' between ? and ?', conn, params=[str(start_date), str(end_date)])            
            
                daily_data = pd.read_sql_query(f"SELECT CAST({config[parameter]['datetime']} AS DATE) AS {config[parameter]['daily_datetime']}, "
                                                f"{site_sql_id} AS {config[parameter]['site_sql_id']}, "
                                                f"ROUND(AVG({config[parameter]['corrected_data']}), 2) AS {config[parameter]['daily_mean']}, ROUND(MAX({config[parameter]['corrected_data']}), 2) AS {config[parameter]['daily_max']}, ROUND(MIN({config[parameter]['corrected_data']}), 2) AS {config[parameter]['daily_min']}, "
                                                f"{derived_mean}{derived_max}{derived_min}"
                                                f"COUNT(*) AS {config[parameter]['daily_record_count']}, "
                                                f"MAX( CAST({config[parameter]['estimate']} AS INT)) AS {config[parameter]['daily_estimate']}, "
                                                f"MAX( CAST({config[parameter]['warning']} AS INT)) AS {config[parameter]['daily_warning']}, "
                                                f"{daily_sum}"
                                                f"{groundwater_temperature}"
                                                f"{snow}"
                                                f"{ice}"
                                                f"{depth}"
                                                f"{provisional}"
                                                f"MAX( CAST({config[parameter]['lock']} AS INT)) AS {config[parameter]['daily_lock']} "
                                                f"FROM {config[parameter]['table']} "
                                                f"WHERE G_ID = {site_sql_id} AND CAST({config[parameter]['datetime']} AS DATE) BETWEEN '{start_date}' AND '{end_date}' "
                                                f"GROUP BY CAST({config[parameter]['datetime']} AS DATE) ", conn)
                daily_data[f"{config[parameter]['daily_auto_timestamp']}"] = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
                print("daily data")
                print(daily_data)
        # arrange columns in desired order
        with sql_engine.begin() as conn:
            desired_order = pd.read_sql_query(f"SELECT TOP 1 * "
                                                f"FROM {config[parameter]['daily_table']} "
                                                f"WHERE G_ID = {site_sql_id} ", conn)
            
        desired_order = desired_order.columns.tolist()
        existing_columns = [col for col in desired_order if col in daily_data.columns]         # Filter out columns that exist in the DataFrame
        daily_data = daily_data[existing_columns]

        for index, row in daily_data.iterrows():
            try:
                daily_data.loc[daily_data.index == index].to_sql(config[parameter]['daily_table'], sql_engine, method=None, if_exists='append', index=False)
            except IntegrityError:
               
                row_dict = row.to_dict()
                
                update_cols = [col for col in row_dict if col not in ["G_ID", config[parameter]['daily_datetime']]]
                key_cols = ["G_ID", config[parameter]['daily_datetime']]

                set_clause = ",\n    ".join([f"{col} = ?" for col in update_cols])
                where_clause = " AND ".join([f"{col} = ?" for col in key_cols])

                update_sql = f"""
                UPDATE {config[parameter]['daily_table']}
                SET
                    {set_clause}
                WHERE {where_clause}
                """
                

                #update_values =  [row_dict[col] for col in key_cols] + [row_dict[col] for col in update_cols]
                update_values = [row_dict[col] for col in update_cols] + [row_dict[col] for col in key_cols]
           
                with sql_engine.begin() as cnn:
                    cnn.execute(update_sql, update_values)

    return daily_data


def full_upload(df, parameter, site_sql_id, utc_offset):
    print("full upload")
    clean_file(df, parameter, site_sql_id, utc_offset)
    print("cleaned")
    delete_data(df, parameter, site_sql_id)
    print("delete")
    upload_data(df, parameter, site_sql_id, utc_offset)
    print("uplad")
    df[config[parameter]['datetime']] = pd.to_datetime(df[config[parameter]['datetime']], errors='coerce')

    # Get max date, strip time, add 2 days
    start_date = df[config[parameter]['datetime']].min().normalize() - pd.Timedelta(days=2)
    end_date = df[config[parameter]['datetime']].max().normalize() + pd.Timedelta(days=2)
    calculate_daily_values(start_date, end_date, parameter, site_sql_id)
   
    print("daily")

def check_if_rating_exists(site_sql_id, rating):
    print("rating check")
    """sql_alchemy_connection = urllib.parse.quote_plus('DRIVER={'+driver+'}; SERVER='+server+'; DATABASE='+database+'; Trusted_Connection='+trusted_connection+';')
    sql_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % sql_alchemy_connection)
    with sql_engine.connect() as conn:
    #    result = conn.execute(text("SELECT COUNT(*) FROM tblFlowRatings WHERE G_ID = :site_id AND RTRIM(RatingNumber) = :rating"), {"site_id": site_sql_id, "rating": rating})
    print("results", result)
    count = result.scalar_one()

    print("count", count)"""
    count = 0 
    return count


def upload_rating(data):
        sql_alchemy_connection = urllib.parse.quote_plus('DRIVER={'+driver+'}; SERVER='+server+'; DATABASE='+database+'; Trusted_Connection='+trusted_connection+';')
        sql_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % sql_alchemy_connection)
        cnxn = sql_engine.raw_connection()
        data.to_sql("tblFlowRatings", sql_engine, method=None, if_exists='append', index=False)
        # try method=multi, None works
        # try chunksize int

        cnxn.close()

def upload_rating_notes(data):
        sql_alchemy_connection = urllib.parse.quote_plus('DRIVER={'+driver+'}; SERVER='+server+'; DATABASE='+database+'; Trusted_Connection='+trusted_connection+';')
        sql_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % sql_alchemy_connection)
        cnxn = sql_engine.raw_connection()
        data.to_sql("tblFlowRating_Stats", sql_engine, method=None, if_exists='append', index=False)
        # try method=multi, None works
        # try chunksize int

        cnxn.close()

def delete_rating(site_sql_id, rating):
    sql_alchemy_connection = urllib.parse.quote_plus('DRIVER={'+driver+'}; SERVER='+server+'; DATABASE='+database+'; Trusted_Connection='+trusted_connection+';')
    sql_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % sql_alchemy_connection)
    with sql_engine.begin() as conn:
        conn.execute(text("""DELETE FROM tblFlowRatings WHERE G_ID = :site_id AND RTRIM(RatingNumber) = :rating"""), {"site_id": site_sql_id, "rating": rating})
        conn.execute(text("""DELETE FROM tblFlowRating_Stats WHERE RTRIM(Rating_Number) = :rating"""),{"rating": rating})
    
