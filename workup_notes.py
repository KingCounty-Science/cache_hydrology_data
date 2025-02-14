import os
from datetime import datetime
from datetime import timedelta
import urllib
import configparser
import numpy as np
#import win32com.client as win32
#import schedule
import pyodbc
import pandas as pd
from datetime import date
from sqlalchemy import create_engine
pd.options.mode.chained_assignment = None  # default='warn', None is no warn
config = configparser.ConfigParser()
# config.read(r'C:\Users\ihiggins\.spyder-py3\gdata_config.ini')
config.read('gdata_config.ini')
print("Run Start at "+str(pd.to_datetime('today'))+"")
print("")
#10.82.12.39
server = "KCITSQLPRNRPX01"
driver = "SQL Server"
database = "gData"
trusted_connection = "yes"
conn = pyodbc.connect('Driver={'+driver+'};'
                      'Server='+server+';'
                      'Database='+database+';'
                      'Trusted_Connection='+trusted_connection+';')


#def discharge_workup_notes
#sql_alchemy_connection = urllib.parse.quote_plus('DRIVER={'+driver+'}; SERVER='+server+'; DATABASE='+database+'; Trusted_Connection='+trusted_connection+';')
#        sql_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % sql_alchemy_connection)
#        cnxn = sql_engine.raw_connection()
#        df.to_sql(config[parameter]['table'], sql_engine, method=None, if_exists='append', index=False)
#        # try method=multi, None works
#        # try chunksize int
#        cnxn.close()

def q_workup_notes(q_observation, site_sql_id, site):
    q_table = "tblFlowWorkUpRatingTracker"
    # sql connection to gdata
    sql_alchemy_connection = urllib.parse.quote_plus('DRIVER={'+driver+'}; SERVER='+server+'; DATABASE='+database+'; Trusted_Connection='+trusted_connection+';')
    sql_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % sql_alchemy_connection)
    conn = sql_engine.raw_connection()
    cur = conn.cursor()
    # get workup notes table info, this is for dev only
    #cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{q_table}'")
    #df = pd.DataFrame(cur.fetchall())
   
    # get rating offset
    cur.execute(f"SELECT Offset FROM tblFlowRating_Stats WHERE Rating_Number = '{q_observation['rating_number'].iloc[0]}'")
    rating_offset = pd.DataFrame(cur.fetchall())
    rating_offset = rating_offset[0].iloc[0]
    rating_offset = rating_offset[0]
    # create dataframe for workup notes
    d = {'G_ID': [site_sql_id], 
        'FileName': [f"{site}_discharge_{q_observation['datetime'].iloc[-1]}.csv"],
        'StartTime': q_observation['datetime'].iloc[0],
        'EndTime': q_observation['datetime'].iloc[-1],
        'WorkUpDate': date.today(),
        'RatingOffset': q_observation['q_offset'].iloc[0],
        'RatingCorrLog': q_observation['q_offset'].iloc[0]-q_observation['q_offset'].iloc[-1],
        'Rating': q_observation['rating_number'].iloc[0],
        'GZF': rating_offset,
        "AutoDTStamp": datetime.today(),
        "WorkUp_Notes": "based on first and last obs",
        "WorkedUp_By": " "
        }
    df = pd.DataFrame(data=d)
    # upload
    df.to_sql(q_table, sql_engine, method=None, if_exists='append', index=False)

    cur.close()
    conn.close()
    
def stage_workup_notes(observation_stage, site_sql_id, site):
    stage_table = "tblFlowWorkUpStageTracker"
    #sql connection
    sql_alchemy_connection = urllib.parse.quote_plus('DRIVER={'+driver+'}; SERVER='+server+'; DATABASE='+database+'; Trusted_Connection='+trusted_connection+';')
    sql_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % sql_alchemy_connection)
    conn = sql_engine.raw_connection()
    cur = conn.cursor()
    # read table info, for dev only
    #cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{stage_table}'")
    #df = pd.DataFrame(cur.fetchall())
   
    # create df for workup notes
    d = {'G_ID': [site_sql_id], 
        'FileName': [f"{site}_discharge_{observation_stage['datetime'].iloc[-1]}.csv"],
        'Start_Time': observation_stage['datetime'].iloc[0],
        'End_Time': observation_stage['datetime'].iloc[-1],
        'WorkUpDate': date.today(),
        'SensorOffset': observation_stage['q_offset'].iloc[0],
        'SensorCorrLog': observation_stage['q_offset'].iloc[0]-observation_stage['q_offset'].iloc[-1],
        "AutoDTStamp": datetime.today(),
        "Comments": "based on first and last obs",
        "WorkedUp_By": " "
        }
    df = pd.DataFrame(data=d)
    # upload df
    df.to_sql(stage_table, sql_engine, method=None, if_exists='append', index=False)
    
    cur.close()
    conn.close()

def workup_transactions(observation, site_sql_id, site, parameter):
    observation_table = "tblWorkUpTransactions"
 
    sql_alchemy_connection = urllib.parse.quote_plus('DRIVER={'+driver+'}; SERVER='+server+'; DATABASE='+database+'; Trusted_Connection='+trusted_connection+';')
    sql_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % sql_alchemy_connection)
    conn = sql_engine.raw_connection()
    cur = conn.cursor()
    # get table info dev only
    #cur.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{observation_table}'")
    #df = pd.DataFrame(cur.fetchall())

    # parameter numbers 
    if parameter == "AirTemp" or parameter == "air_temperature" or parameter == "Air_Temperature":
        parameter_number = 4
    elif parameter == "barometer" or parameter == "Barometer":
        parameter_number = 10
    elif parameter == "Conductivity" or parameter == "conductivity":
        parameter_number = 6
    elif parameter == "discharge" or parameter == "FlowLevel":
        parameter_number = 2
    elif parameter == "DO" or parameter == "dissolved_oxygen":
        parameter_number = 5
    elif parameter == "LakeLevel" or parameter == "water_level" or parameter == "lake_level":
        parameter_number = 39
    elif parameter == "water_temperature":
        parameter_number = 3
    elif parameter == "Piezometer" or parameter == "piezometer" or parameter == "groundwater_level":
        parameter_number = 36
    elif parameter == "Precip" or parameter == "precipitation":
        parameter_number = 1
    elif parameter == "Turbidity" or parameter == "turbidity":
        parameter_number = 2
    else:
        parameter_number = " "

# observation dataframe
    d = {'G_ID': [site_sql_id], 
        'WorkUp_Date': date.today(),
        "WorkedUp_By": " ",
        'WorkUp_notes': [f"{site}_discharge_{observation['datetime'].iloc[-1]}.csv"],
        'Start_Time': observation['datetime'].iloc[0],
        'End_Time': observation['datetime'].iloc[-1],
        'Parameter': parameter_number
        
        }
    df = pd.DataFrame(data=d)
    df.to_sql(observation_table, sql_engine, method=None, if_exists='append', index=False)

    cur.close()
    conn.close()

def workup_notes_main(notes_df, parameter, site_sql_id, site):
    # if discharge
    if parameter == "discharge" or parameter == "FlowLevel":
        # filter by observations
        observation_stage = notes_df.dropna(subset=['observation_stage'])
        q_observation = notes_df.dropna(subset=['q_observation'])
        observation = notes_df.dropna(subset=['observation_stage'])
        q_workup_notes(q_observation, site_sql_id, site)
        stage_workup_notes(observation_stage, site_sql_id, site)
        workup_transactions(observation, site_sql_id, site, parameter)
        #set table name
        #stage_table = "tblFlowWorkupStageTracker"
        #q_table = "tblFlowWorkupRatingTracker"
    elif parameter == "LakeLevel" or parameter == "Piezometer" or parameter == "water_level" or parameter == "lake_level" or parameter == "groundwater_level":
        observation = notes_df.dropna(subset=['observation_stage'])
        workup_transactions(observation, site_sql_id, site, parameter)
    else:
        observation = notes_df.dropna(subset=['parameter_observation'])
        workup_transactions(observation, site_sql_id, site, parameter)
        


'''
notes_df = pd.read_csv(r"W:/STS/hydro/GAUGE/Temp/Ian's Temp/31i_discharge_2022_02_17.csv")
parameter = "discharge"
site_sql_id = 103
site = "31i"
workup_notes_main(notes_df, parameter, site_sql_id, site)
'''
#7
from import_data import get_site_sql_id, sql_import, get_horizontal_datum
#import pyexcel


def excel_export(project, site_list, start_date, end_date):
    all_df = pd.DataFrame(columns=['datetime'])
    for item in site_list:
        site_sql_id = get_site_sql_id(item)
        site_df = sql_import("LakeLevel", site_sql_id, start_date, end_date)
        if site_df.empty:
           site_df = sql_import("Piezometer", site_sql_id, start_date, end_date) 
        site_df = site_df[["datetime", "data", "corrected_data"]]
        site_df = site_df.rename(columns={'data': f"{item}_data"})
        site_df = site_df.rename(columns={'corrected_data': f"{item}_corrected_data"})
        #ground_ele = get_horizontal_datum(site_sql_id)
        #all_df[f"{item}_ground_ele"] = ground_ele
        #all_df = pd.concat([all_df, site_df], axis=0, ignore_index=True)
        site_df = site_df.sort_values(by='datetime', ascending=True)
        all_df = pd.merge(all_df, site_df, on='datetime', how='outer')

    #all_df.set_index('datetime', inplace=True)
    #all_df = round(all_df.resample('D').mean(), 2)
    #all_df.reset_index(inplace=True)
    ## add ground elevation
    #for item in site_list:
    #    ground_ele = get_horizontal_datum(site_sql_id)
    #    all_df[f"{item}_ground_ele"] = ground_ele
        
    # alphabetical order
    all_df.set_index('datetime', inplace=True)
    all_df = all_df.reindex(sorted(all_df.columns), axis=1)
    all_df.reset_index(inplace=True)
    all_df = all_df.sort_values(by='datetime', ascending=True)
    print(all_df)
    all_df.to_csv(f"W:/STS/hydro/GAUGE/Temp/Ian's Temp/{project}_export_{datetime.today().strftime('%Y_%m_%d')}.csv", index=False, header= True)
    #all_df.py_tocsv().save_as(records=all_df.to_dict(orient='records'), dest_file_name=r"W:/STS/hydro/GAUGE/Temp/Ian's Temp/taylor_creek.xlsx")
    #all_df.to_excel(, index=False)
    
project = "Lones"
site_list = ["Lones_02", "Lones_06", "Lones_13", "Lones_14"]
start_date = "01/01/2020 0:00" #"%m/%d/%Y %H:%M"
end_date = "08/16/2024 0:00" #"%m/%d/%Y %H:%M"
#excel_export(project, site_list, start_date, end_date)

