import base64
import datetime as dt
from datetime import timedelta
from datetime import datetime

import pyodbc
import configparser
import pandas as pd
from datetime import date
import numpy as np


from plotly.subplots import make_subplots
from sqlalchemy import create_engine
import urllib

config = configparser.ConfigParser()
config.read('gdata_config.ini')

SQL_String = pyodbc.connect('Driver={'+config['sql_connection']['Driver']+'};'
                            'Server='+config['sql_connection']['Server']+';'
                            'Database=' +
                            config['sql_connection']['Database']+';'
                            'Trusted_Connection='+config['sql_connection']['Trusted_Connection']+';')

# new sql alchemy connection
server = config['sql_connection']['Server']
driver = config['sql_connection']['Driver']
database = config['sql_connection']['Database']
trusted_connection = config['sql_connection']['Trusted_Connection']
   

# pyodbc has a longer pooling then sql_alchemy and needs to be reset
pyodbc.pooling = False
# not sure this fast execumetry sped things up
# info on host name connection https://docs.sqlalchemy.org/en/14/dialects/mssql.html#connecting-to-pyodbc
sql_alchemy_connection = urllib.parse.quote_plus('DRIVER={'+driver+'}; SERVER='+server+'; DATABASE='+database+'; Trusted_Connection='+trusted_connection+';')
sql_engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % sql_alchemy_connection)

def sql_import(parameter, site_sql_id, start_date, end_date):
    """selects data fro site based on sql id; selects all data or accepts start date and end date
    accepts start and end date in PDT, converts to UTC for query, converts result to PDT"""
    if start_date != '' and end_date != '': # if there is a startdate or all obs generate query string

        if parameter == "FlowLevel" or parameter == "discharge":
                select_statement = f"SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config[parameter]['datetime']}, 120)) as datetime, {config[parameter]['data']} as data, {config[parameter]['corrected_data']} as corrected_data, {config[parameter]['discharge']} as discharge, {config[parameter]['est']} as estimate, {config[parameter]['warning']} as warning, {config[parameter]['provisional']} as provisional  "
        elif parameter == "barometer":
                # barometer (only has "data column")
                select_statement = f"SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config[parameter]['datetime']}, 120)) as datetime, {config[parameter]['corrected_data']} as corrected_data, {config[parameter]['est']} as estimate, {config[parameter]['warning']} as warning "
        elif parameter == "water_level" or parameter == "groundwater_level": # get dry indicator
                select_statement = f"SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config[parameter]['datetime']}, 120)) as datetime, {config[parameter]['data']} as data, {config[parameter]['corrected_data']} as corrected_data, {config[parameter]['est']} as estimate, {config[parameter]['warning']} as warning, {config[parameter]['non_detect']} as non_detect, {config[parameter]['provisional']} as provisional   "
        else:
                select_statement = f"SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config[parameter]['datetime']}, 120)) as datetime, {config[parameter]['data']} as data, {config[parameter]['corrected_data']} as corrected_data, {config[parameter]['est']} as estimate, {config[parameter]['warning']} as warning, {config[parameter]['provisional']} as provisional   "
                
        if start_date == '*' and end_date == "*": # if querying all datetimes you dont need a between
                with sql_engine.begin() as conn:
                        df = pd.read_sql_query(f"{select_statement}"
                                f"FROM {config[parameter]['table']} "
                                F"WHERE G_ID = {str(site_sql_id)} "
                                f"ORDER BY {config[parameter]['datetime']} DESC", conn)

        else: # covert start and end date to datetime       
                start_date = pd.to_datetime(start_date).to_pydatetime()
                start_date = (start_date + timedelta(hours=(7))).strftime("%m/%d/%Y %H:%M")
                end_date = pd.to_datetime(end_date).to_pydatetime()
                end_date = (end_date + timedelta(hours=(7))).strftime("%m/%d/%Y %H:%M")
            
        
                with sql_engine.begin() as conn: # quert qith a between statement
                        df = pd.read_sql_query(f"{select_statement}"
                                f"FROM {config[parameter]['table']} "
                                F"WHERE G_ID = {str(site_sql_id)} "
                                f"AND {config[parameter]['datetime']} BETWEEN ? and ? "
                                f"ORDER BY {config[parameter]['datetime']} DESC", conn, params=[str(start_date), str(end_date)])
           
    
            
        return df


def sql_parameter_import(parameter, start_date, end_date):
    # gData autoDTStamp is in PDD
    # gData datetime is in UTC
    if start_date != '' and end_date != '':
            start_date = pd.to_datetime(start_date).to_pydatetime()
            start_date = (start_date + timedelta(hours=(0))).strftime("%m/%d/%Y %H:%M")
            end_date = pd.to_datetime(end_date).to_pydatetime()
            end_date = (end_date + timedelta(hours=(0))).strftime("%m/%d/%Y %H:%M")
            if parameter == "FlowLevel" or parameter == "discharge":
        
                select_statement = f"SELECT s.{config['site_identification']['site']} as site, "\
                                f"DATEADD(HOUR, -7, CONVERT(DATETIME, t.{config[parameter]['datetime']}, 120)) as datetime, "\
                                f"t.{config[parameter]['data']} as data, "\
                                f"t.{config[parameter]['corrected_data']} as corrected_data, "\
                                f"t.{config[parameter]['discharge']} as discharge, "\
                                f"t.{config[parameter]['est']} as estimate, "\
                                f"t.{config[parameter]['warning']} as warning, "\
                                f"t.{config[parameter]['provisional']} as provisional, "\
                                f"DATEADD(HOUR, -0, CONVERT(DATETIME, t.{config[parameter]['update_timestamp']}, 120)) as update_timestamp "
                        

            elif parameter == "barometer": # barometer (only has "data column")
                select_statement = f"SELECT {config['site_identification']['site_sql_id']} as site_sql_id, DATEADD(HOUR, -7, CONVERT(DATETIME, {config[parameter]['datetime']}, 120)) as datetime, {config[parameter]['corrected_data']} as corrected_data, {config[parameter]['est']} as estimate, {config[parameter]['warning']} as warning, {config[parameter]['provisional']} as provisional, {config[parameter]['upload_timestamp']} as upload_timestamp "
            else:
        
                
                
                select_statement = f"SELECT s.{config['site_identification']['site']} as site, "\
                                f"DATEADD(HOUR, -7, CONVERT(DATETIME, t.{config[parameter]['datetime']}, 120)) as datetime, "\
                                f"t.{config[parameter]['data']} as data, "\
                                f"t.{config[parameter]['corrected_data']} as corrected_data, "\
                                f"t.{config[parameter]['est']} as estimate, "\
                                f"t.{config[parameter]['warning']} as warning, "\
                                f"t.{config[parameter]['provisional']} as provisional, "\
                                f"DATEADD(HOUR, -0, CONVERT(DATETIME, t.{config[parameter]['update_timestamp']}, 120)) as update_timestamp "#select_statement = f"SELECT {config['site_identification']['site_sql_id']} as site_sql_id, DATEADD(HOUR, -7, CONVERT(DATETIME, {config[parameter]['datetime']}, 120)) as datetime, {config[parameter]['data']} as data, {config[parameter]['corrected_data']} as corrected_data, {config[parameter]['est']} as estimate, {config[parameter]['warning']} as warning, {config[parameter]['provisional']} as provisional, {config[parameter]['upload_timestamp']} as upload_timestamp "
            with sql_engine.begin() as conn:
                df = pd.read_sql_query(f"{select_statement}"
                                        f"FROM {config[parameter]['table']} as t "
                                        f"INNER JOIN {config['site_identification']['table']} as s "
                                        f"ON t.{config['site_identification']['site_sql_id']} = s.{config['site_identification']['site_sql_id']} "
                                        f"WHERE t.{config[parameter]['update_timestamp']} BETWEEN ? AND ? "
                                        f"ORDER BY {config[parameter]['datetime']} DESC;", conn, params=[str(start_date), str(end_date)])       
                             
                df = df.drop(columns=["update_timestamp"])
                                  
                                    
                                    #F"WHERE G_ID = {str(site_sql_id)} "
                                    #f"AND {config[parameter]['datetime']} BETWEEN ? and ? "
    else:
            df = pd.DataFrame()
    return df
def sql_statistics(parameter, site_sql_id):
        select_statement = f"""
        SELECT 
        MIN(DATEADD(HOUR, -7, CONVERT(DATETIME, {config[parameter]['datetime']}, 120))) as first_datetime,
        MAX(DATEADD(HOUR, -7, CONVERT(DATETIME, {config[parameter]['datetime']}, 120))) as last_datetime
        
        FROM {config[parameter]['table']}
        WHERE G_ID = {str(site_sql_id)}
        """
    
        # Execute the query and fetch the first and last datetime
        with sql_engine.begin() as conn:
            result = conn.execute(select_statement).fetchone()

        first_datetime = result['first_datetime']
        last_datetime = result['last_datetime']

        # ger q 05 precentile
        if parameter == "discharge" or parameter == "FlowLevel":
                select_statement = f"""SELECT 
                        PERCENTILE_CONT(0.05) 
                        WITHIN GROUP (ORDER BY {config[parameter]['discharge_mean']})
                        OVER () AS percentile_95
                        FROM {config[parameter]['daily_table']}
                        WHERE G_ID = {str(site_sql_id)};"""
                with sql_engine.begin() as conn:
                        result = conn.execute(select_statement).fetchone()

                percentile_95_q = result['percentile_95']
        else:
                percentile_95_q = 0

        # get 05 percentile water level (05 is the lowest 05 flow)
        try:
                select_statement = f"""SELECT 
                                PERCENTILE_CONT(0.05) 
                                WITHIN GROUP (ORDER BY {config[parameter]['daily_mean']})
                                OVER () AS percentile_95
                                FROM {config[parameter]['daily_table']}
                                WHERE G_ID = {str(site_sql_id)};"""
                with sql_engine.begin() as conn:
                        result = conn.execute(select_statement).fetchone()

                percentile_95 = result['percentile_95']
        except:
              percentile_95 = 0

        # Get both min and max in one query stage/parameter
        if parameter != "discharge":
                try:
                        select_statement = f"""SELECT 
                                                MIN({config[parameter]['daily_min']}) AS min_value,
                                                MAX({config[parameter]['daily_max']}) AS max_value
                                                FROM {config[parameter]['daily_table']}
                                                WHERE G_ID = {str(site_sql_id)};"""
                        
                        with sql_engine.begin() as conn:
                                result = conn.execute(select_statement).fetchone()
                       
                        min_corrected_data = result['min_value']
                        max_corrected_data = result['max_value']
                except Exception as e:
                        print(f"Error occurred: {e}")  # This will show you what's wrong
                        min_corrected_data = np.nan
                        max_corrected_data = np.nan
                min_derived_parameter = np.nan
                max_derived_parameter = np.nan
        elif parameter == "discharge":
                try:
                        select_statement = f"""SELECT 
                                                MIN({config[parameter]['daily_min']}) AS min_value,
                                                MAX({config[parameter]['daily_max']}) AS max_value,
                                                MIN({config[parameter]['discharge_min']}) AS min_derived_parameter,
                                                MAX({config[parameter]['discharge_max']}) AS max_derived_parameter
                                                FROM {config[parameter]['daily_table']}
                                                WHERE G_ID = {str(site_sql_id)};"""
                        with sql_engine.begin() as conn:
                                result = conn.execute(select_statement).fetchone()
                        
                        min_corrected_data = result['min_value']
                        max_corrected_data = result['max_value']
                        min_derived_parameter = result['min_derived_parameter']  # Fixed
                        max_derived_parameter = result['max_derived_parameter']  # Fixed
                except Exception as e:
                        print(f"Error occurred: {e}")  # Added error printing
                        min_corrected_data = np.nan
                        max_corrected_data = np.nan
                        min_derived_parameter = np.nan
                        max_derived_parameter = np.nan
               
        # Extract first and last datetime from the result
        statistics = {
        "first_datetime": first_datetime,
        "last_datetime": last_datetime,
        "percentile_05_q": percentile_95_q,
        "percentile_05": percentile_95,
        "min_corrected_data": min_corrected_data,
        "max_corrected_data": max_corrected_data,
        "min_derived_parameter": min_derived_parameter,
        "max_derived_parameter": max_derived_parameter,
        }
        
        return statistics

def sql_get_closest_datetime(parameter, site_sql_id, date):
        date = pd.to_datetime(date).to_pydatetime()
        date = (date + timedelta(hours=(7))).strftime("%m/%d/%Y %H:%M")
        select_statement = f"""SELECT TOP 1 CONVERT(DATETIME, {config[parameter]['datetime']}, 120) FROM {config[parameter]['table']} WHERE G_ID = {str(site_sql_id)} ORDER BY ABS(DATEDIFF(SECOND, DATEADD(HOUR, 0, CONVERT(DATETIME, {config[parameter]['datetime']}, 120)), '{date}'))"""
         # Execute the query and fetch the first and last datetime
        with sql_engine.begin() as conn:
                result_a = conn.execute(select_statement).fetchone()
        # finds closest
        result_a = result_a[0]
        result_a = (result_a - timedelta(hours=(7)))
        # finds value below
        select_statement = f"""SELECT TOP 1 CONVERT(DATETIME, {config[parameter]['datetime']}, 120) FROM {config[parameter]['table']} WHERE G_ID = {str(site_sql_id)} AND CONVERT(DATETIME, {config[parameter]['datetime']}, 120) < '{date}' ORDER BY ABS(DATEDIFF(SECOND, DATEADD(HOUR, 0, CONVERT(DATETIME, {config[parameter]['datetime']}, 120)), '{date}'))"""
         # Execute the query and fetch the first and last datetime
        with sql_engine.begin() as conn:
                result_b = conn.execute(select_statement).fetchone()
        
        result_b = result_b[0]
        result_b = (result_b - timedelta(hours=(7)))

        # finds value above
        # if there is no c observation return A which is the closest
        try:
                select_statement = f"""SELECT TOP 1 CONVERT(DATETIME, {config[parameter]['datetime']}, 120) FROM {config[parameter]['table']} WHERE G_ID = {str(site_sql_id)} AND CONVERT(DATETIME, {config[parameter]['datetime']}, 120) > '{date}' ORDER BY ABS(DATEDIFF(SECOND, DATEADD(HOUR, 0, CONVERT(DATETIME, {config[parameter]['datetime']}, 120)), '{date}'))"""
                # Execute the query and fetch the first and last datetime
                with sql_engine.begin() as conn:
                        result_c = conn.execute(select_statement).fetchone()
                
                result_c = result_c[0]
                result_c = (result_c - timedelta(hours=(7)))
        except:
               result_c = result_a
      
        #select_statement = f"""SELECT TOP 1 CONVERT(DATETIME, {config[parameter]['datetime']}, 120) FROM {config[parameter]['table']} WHERE G_ID = {str(site_sql_id)} ORDER BY -ABS(DATEDIFF(SECOND,  '{date}', DATEADD(HOUR, 0, CONVERT(DATETIME, {config[parameter]['datetime']}, 120)) ))"""
         # Execute the query and fetch the first and last datetime
        #with sql_engine.begin() as conn:
        #        result_b = conn.execute(select_statement).fetchone()
        #result_b = result_b[0]
        #result_b = (result_b - timedelta(hours=(7)))
        return result_a, result_b, result_c


def get_observations_join(parameter, site_sql_id, startDate, endDate):
        added_time_window = 12 # you want to pull in observations from before and after the start of the record as the observation could be taken 1 minute before start of record
        
        if startDate != "*" and endDate != "*":
                startDate = startDate + timedelta(hours=(7))
                endDate = endDate + timedelta(hours=(7))
                

        if parameter == "water_level" or parameter == "LakeLevel" or parameter == "groundwater_level" or parameter == "piezometer" or parameter == "Piezometer": # no parameter value to join with
                with sql_engine.begin() as conn: 
                        observations = pd.read_sql_query(f"""   SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config['observation']['datetime']}, 120)) as datetime, ROUND({config['observation']['observation_stage']}, 2) as observation_stage, Comments as comments
                                                    FROM tblFieldVisitInfo 
                                                    WHERE tblFieldVisitInfo.G_ID = {site_sql_id} ORDER BY {config['observation']['datetime']} DESC;""", conn)    
                observations["observation_stage"] = observations["observation_stage"].round(2)
        else:
                with sql_engine.begin() as conn:                                      
                        parameter_observations = pd.read_sql_query(f"""
                                                SELECT 
                                                        DATEADD(HOUR, -7, CONVERT(DATETIME, {config['observation']['datetime']}, 120)) AS datetime, 
                                                        {config['observation']['observation_number']} AS observation_number, 
                                                        ROUND({config['observation']['observation_stage']}, 2) AS observation_stage, 
                                                        ROUND(tblFieldData.Parameter_Value, 2) AS parameter_observation, 
                                                        Comments AS comments
                                                FROM 
                                                        tblFieldVisitInfo 
                                                LEFT OUTER JOIN 
                                                        tblFieldData ON (tblFieldVisitInfo.FieldVisit_ID = tblFieldData.FieldVisit_ID)
                                                WHERE 
                                                        tblFieldVisitInfo.G_ID = {site_sql_id} 
                                                        AND tblFieldData.Parameter = {config[parameter]['observation_type']}
                                                ORDER BY 
                                                        {config['observation']['datetime']} DESC;
                                                """, conn)
                        parameter_observations["observation_stage"] = parameter_observations["observation_stage"].round(2)
                       
                        if not parameter_observations.empty: # your shouldnt really need this but the above join seems to leave out stage measurements without discharge nomatter how i do the join
                                observations = pd.read_sql_query(f"""   SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config['observation']['datetime']}, 120)) as datetime, ROUND({config['observation']['observation_stage']}, 2) as observation_stage, Comments as comments
                                                        FROM tblFieldVisitInfo 
                                                        WHERE tblFieldVisitInfo.G_ID = {site_sql_id} ORDER BY {config['observation']['datetime']} DESC;""", conn)   
                                observations = observations.merge(parameter_observations, on = ["datetime", "observation_stage", "comments"], how = "outer")
                                
                if observations.empty: # if there are no parameter observations ie a waterlevel site
                        with sql_engine.begin() as conn: 
                                observations = pd.read_sql_query(f"""   SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config['observation']['datetime']}, 120)) as datetime, ROUND({config['observation']['observation_stage']}, 2) as observation_stage, Comments as comments
                                                        FROM tblFieldVisitInfo 
                                                        WHERE tblFieldVisitInfo.G_ID = {site_sql_id} ORDER BY {config['observation']['datetime']} DESC;""", conn)    
                        observations["observation_stage"] = observations["observation_stage"].round(2)
        
        return observations



def get_all_observations():
        parameter = "discharge"
        #added_time_window = 12 # you want to pull in observations from before and after the start of the record as the observation could be taken 1 minute before start of record
        # convert to datetime
        #startDate = pd.to_datetime(startDate)
        #endDate = pd.to_datetime(endDate)

        # convert start/end date to utc time
        #startDate = startDate + timedelta(hours=(7))
        #endDate = endDate + timedelta(hours=(7))

        # add data window
        #startDate = startDate - timedelta(hours=(added_time_window))
        #endDate = endDate + timedelta(hours=(added_time_window))

        # convert to string
        #startDate = startDate.strftime("%m/%d/%Y %H:%M")
        #endDate = endDate.strftime("%m/%d/%Y %H:%M")
        if parameter == "water_level" or parameter == "LakeLevel" or parameter == "groundwater_level" or parameter == "piezometer" or parameter == "Piezometer": # no parameter value to join with
              with sql_engine.begin() as conn: 
                     #observations = pd.read_sql_query(f"""   SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config['observation']['datetime']}, 120)) as datetime, {config['observation']['observation_stage']} as observation_stage, Comments as comments
                     #                               FROM tblFieldVisitInfo 
                     #                               WHERE tblFieldVisitInfo.G_ID = {site_sql_id} AND tblFieldVisitInfo.Date_Time BETWEEN ? AND ?;""", conn, params=[str(startDate), str(endDate)])   
                     observations = pd.read_sql_query(f"""   SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config['observation']['datetime']}, 120)) as datetime, {config['observation']['observation_stage']} as observation_stage, Comments as comments
                                                    FROM tblFieldVisitInfo 
                                                    WHERE tblFieldVisitInfo.G_ID = {site_sql_id};""", conn)    
        if parameter == "discharge":
                with sql_engine.begin() as conn:                                      
                #observations = pd.read_sql_query(f"""  SELECT Measurement_Number,                                                  Date_Time AS date,                                                                         Stage_Feet,                                                         tblFieldData.Parameter_Value, Comments FROM tblFieldVisitInfo INNER JOIN tblFieldData ON (tblFieldVisitInfo.FieldVisit_ID = tblFieldData.FieldVisit_ID) WHERE tblFieldVisitInfo.G_ID = {site_sql_id} AND tblFieldVisitInfo.Date_Time BETWEEN ? AND ? AND tblFieldData.Parameter = 2;""", conn, params=[str(startDate), str(endDate)])
                #observations = pd.read_sql_query(f"""   SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config['observation']['datetime']}, 120)) as datetime, {config['observation']['observation_number']} as observation_number, {config['observation']['observation_stage']} as observation_stage, tblFieldData.Parameter_Value as parameter_observation, Comments as comments
                #                                        FROM tblFieldVisitInfo INNER JOIN tblFieldData ON (tblFieldVisitInfo.FieldVisit_ID = tblFieldData.FieldVisit_ID) 
                #                                        WHERE tblFieldVisitInfo.G_ID = {site_sql_id} AND tblFieldVisitInfo.Date_Time BETWEEN ? AND ? AND tblFieldData.Parameter = {config[parameter_value]['observation_type']};""", conn, params=[str(startDate), str(endDate)])
                        observations = pd.read_sql_query(f"""   SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config['observation']['datetime']}, 120)) as datetime, {config['observation']['observation_number']} as observation_number, {config['observation']['observation_stage']} as observation_stage, tblFieldData.Parameter_Value as parameter_observation, tblFieldVisitInfo.G_ID as site_sql_id, Comments as comments
                                                        FROM tblFieldVisitInfo INNER JOIN tblFieldData ON (tblFieldVisitInfo.FieldVisit_ID = tblFieldData.FieldVisit_ID) 
                                                        WHERE tblFieldData.Parameter = {config[parameter]['observation_type']};""", conn)
                if not parameter_observations.empty:
                                observations = pd.read_sql_query(f"""   SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config['observation']['datetime']}, 120)) as datetime, {config['observation']['observation_stage']} as observation_stage, Comments as comments
                                                        FROM tblFieldVisitInfo 
                                                        WHERE tblFieldVisitInfo.G_ID = {site_sql_id};""", conn)   
                                observations = observations.merge(parameter_observations, on = ["datetime", "observation_stage", "comments"], how = "outer")
                
                if observations.empty: # if there are no parameter observations ie a waterlevel site
                        with sql_engine.begin() as conn: 
                        #observations = pd.read_sql_query(f"""   SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config['observation']['datetime']}, 120)) as datetime, {config['observation']['observation_stage']} as observation_stage, Comments as comments
                        #                               FROM tblFieldVisitInfo 
                        #                               WHERE tblFieldVisitInfo.G_ID = {site_sql_id} AND tblFieldVisitInfo.Date_Time BETWEEN ? AND ?;""", conn, params=[str(startDate), str(endDate)])   
                                observations = pd.read_sql_query(f"""   SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config['observation']['datetime']}, 120)) as datetime, {config['observation']['observation_stage']} as observation_stage, tblFieldVisitInfo.G_ID as site_sql_id, Comments as comments
                                                        FROM tblFieldVisitInfo;""", conn)    
     
        parameter = "discharge"
        with sql_engine.begin() as conn:
                sites = pd.read_sql_query(f"select SITE_CODE as site_id, G_ID as site_sql_id from tblGaugeLLID WHERE STATUS = 'Active' AND FlowLevel = 'True' ORDER BY SITE_CODE ASC;", conn)
        

        sites.to_csv("example_data/sites.csv")


        observations = observations.merge(sites, on = "site_sql_id")
        observations = observations.drop(columns = ["site_sql_id"])
       
        observations.to_csv("example_data/observations.csv")
        #return observations



def usgs_data_import(site_name, parameter, start_date, end_date):
        from data_cleaning import reformat_data
        if start_date != "" and end_date != "":
                start_date = start_date.strftime("%Y-%m-%dT%H:%M")
                end_date = end_date.strftime("%Y-%m-%dT%H:%M")
                #site_number = site_name


                usgs_site_codes = pd.read_csv("external_comaprison_sites.csv", skipinitialspace=True)
                site_number = usgs_site_codes.loc[usgs_site_codes["site"] == site_name, "site_sql_id"].item()

                #site_number = pd.read_csv("external_comaprison_sites.csv", skipinitialspace=True)
                #site_number = site_number["site_sql_id"].loc[site_number["site"] == f"USGS {site_name}"].item()
                # convert comparison parameter to number
                # https://help.waterdata.usgs.gov/parameter_cd?group_cd=PHY we should use these paramters for gdata
                if parameter == "discharge":
                        parameter_number = "00060"
                elif parameter == "stage" or parameter == "water_level":
                       parameter_number = "00065"
                elif parameter == "water_temperature":
                        parameter_number = "00010"
                elif parameter == "wind_speed":
                        parameter_number = "00038"
                elif parameter == "precipitation":
                        parameter_number = "00045"
                elif parameter == "relative_humidity":
                        parameter_number = "00025"
                elif parameter == "air_temperature":
                        parameter_number = "00020"
                elif parameter == "solar_radiation":
                        parameter_number = "00036"
                else:
                       parameter_number = "00060"
                # Example query parameters (replace with your values)
                #site_number = '12119000'
                #start_date = '2022-01-10T00:00'
                #end_date = '2023-01-10T00:00'
                
                # USGS API endpoint for streamflow data
                #api_url = f'https://waterdata.usgs.gov/nwis/dv?site_no={site_number}&format=json&startDT={start_date}&endDT={end_date}'
                #https://waterdata.usgs.gov/nwis/dv?site_no=12119000&format=json&startDT=2022-01-10&endDT=2024-01-20'
                # https://waterdata.usgs.gov/monitoring-location/12119000/#parameterCode=00065&period=P7D&showMedian=false
                #00060 = discharge
                #00065 = water level
                url = f'https://waterservices.usgs.gov/nwis/iv/?sites={site_number}&parameterCd={parameter_number}&startDT={start_date}-07:00&endDT={end_date}-07:00&siteStatus=all&format=rdb'
                df = pd.read_csv(url, delimiter='\t', comment='#', skiprows=30, header=None, names=['site_number', 'datetime', 'timezone', 'comparison', 'status'])
                
               # print(pd.read_csv(url, delimiter='\t', comment='#', skiprows=1, header=None, names=['site_number', 'datetime', 'timezone', 'comparison', 'status']))
                df = df.reset_index(drop=True)
              # Convert to UTC
                # data is read in PST PDT 
                #df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce', infer_datetime_format=True)
                #df['datetime'] = df['datetime'].dt.tz_localize('America/Los_Angeles').dt.tz_convert('UTC')
                df = df[["datetime", "comparison"]]
                df.rename(columns={'comparison': 'corrected_data'}, inplace=True)
                df["site"] = site_name
                df["parameter"] = parameter
                desired_order = ["site", "parameter", "datetime", "corrected_data"]
                existing_columns = [col for col in desired_order if col in df.columns] 
                #df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce', infer_datetime_format=True)
                #df['datetime'] = (pd.to_datetime(startDate).to_pydatetime()) - timedelta(hours=(7))

                df = reformat_data(df)
                return df
                # Make the API request
                #response = requests.get(api_url)

                # Check if the request was successful
                #if response.status_code == 200:
                        #data = response.json()
                        # Process the data as needed
                #        print(response.text)
                #else:
                #        print(f"Error: {response.status_code}")
#usgs_data_import()

def noaa_data_import(site_name, parameter, start_date, end_date):
       
        token = "eNexYxFXopAZHtKttvNTbxxScXLOHNSN"
        # datasets
        # BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/datasets"
        """GHCND: Daily Summaries
        GSOM: Global Summary of the Month
        GSOY: Global Summary of the Year
        NORMAL_ANN: Normals Annual/Seasonal
        NORMAL_DLY: Normals Daily
        NORMAL_HLY: Normals Hourly
        NORMAL_MLY: Normals Monthly"""
        import requests

        # Your NOAA API token (replace with your actual token)
        API_TOKEN = "eNexYxFXopAZHtKttvNTbxxScXLOHNSN"

        # Station ID for KBFI in the GHCND dataset
        STATION_ID = "GHCND:USW00024233"

        # Base URL for the NOAA datatypes endpoint
        BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/datatypes"
        
        # Set up query parameters
        params = {
        "stationid": STATION_ID,
        "datasetid": "GHCND",
        "limit": 1000
        }

        # Set up request headers
        headers = {
        "token": API_TOKEN
        }

        # Make the GET request
        response = requests.get(BASE_URL, headers=headers, params=params)

        # Check for success
        if response.status_code == 200:
                data = response.json()
                print(f"Datatypes measured at {STATION_ID}:\n")
        for dt in data.get("results", []):
                print(f"{dt['id']}: {dt['name']}")
        else:
                print(f"Error {response.status_code}: {response.text}")

        # renton
        STATION_ID = "GHCND:USW00024233"  # Or switch to ISD station ID if needed\
        ## boeing field
       # STATION_ID = "GHCND:USW00024233"
        
        DATASET_ID = "GHCND"  # Or "LCD", "ISD" depending on what's available
        DATATYPE_ID = "ACSH"  # Replace with actual sunshine datatype from step 1
        START_DATE = "2024-05-01"
        END_DATE = "2024-05-31"

        url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"

        params = {
        "datasetid": DATASET_ID,
        "datatypeid": DATATYPE_ID,
        "stationid": STATION_ID,
        "startdate": START_DATE,
        "enddate": END_DATE,
        "units": "standard",  # or "metric"
        "limit": 1000
        }

        headers = {
        "token": API_TOKEN
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                        print("Date\t\tSunshine Value")
                        for item in results:
                                date = item['date'][:10]
                                value = item['value']
                                print(f"{date}\t{value}")
                else:
                        print("No sunshine data found for this period.")
        else:
                print(f"Error {response.status_code}: {response.text}")


#noaa_data_import("", "", "", "")

def get_site_sql_id(site_id):
    with sql_engine.begin() as conn:
        #gage_lookup = pd.read_sql_query('select G_ID, SITE_CODE from tblGaugeLLID;', conn)
        site_sql_id = pd.read_sql_query(
                        f"SELECT {config['site_identification']['site_sql_id']} "
                        f"FROM {config['site_identification']['table']} WHERE {config['site_identification']['site_id']} = '{site_id}';", conn)
    site_sql_id = site_sql_id.iloc[0, 0]
    
    return site_sql_id

def get_horizontal_datum(site_sql_id):
       #Horiz_datum = datum on ground
        with sql_engine.begin() as conn:
        #gage_lookup = pd.read_sql_query('select G_ID, SITE_CODE from tblGaugeLLID;', conn)
                ground_ele = pd.read_sql_query(
                        f"SELECT Horiz_datum "
                        f"FROM {config['site_identification']['table']} WHERE {config['site_identification']['site_sql_id']} = '{site_sql_id}';", conn)
        ground_ele = ground_ele.iloc[0, 0]
    
        return ground_ele


def get_rating_points_and_list(site_sql_id): # for rating analysis
        with sql_engine.begin() as conn:
            rating_points = pd.read_sql_query(f"""
            SELECT 
                r.WaterLevel as stage_rating, 
                CAST(r.Discharge AS float) as discharge_rating, 
                r.RatingNumber as rating,
                s.Offset as gzf
            FROM 
                tblFlowRatings r
            JOIN 
                tblFlowRating_Stats s
            ON 
                r.RatingNumber = s.Rating_Number
            WHERE 
                r.G_ID = '{str(site_sql_id)}';
        """, conn)
        print(rating_points)
        rating_points = rating_points.sort_values(by = ["rating", "stage_rating"])
        rating_points = rating_points.set_index("rating")
        

        #grouped = rating_points.groupby('RatingNumber')
        #individual_dfs = {rating_number: group for rating_number, group in grouped}
        rating_list = list(rating_points.index.unique())
    
        return rating_points, rating_list
def get_all_rating_points_and_list(): # for rating analysis
        with sql_engine.begin() as conn:
            rating_points = pd.read_sql_query(f"""
            SELECT 
                r.WaterLevel as stage_rating, 
                CAST(r.Discharge AS float) as discharge_rating, 
                r.RatingNumber as rating,
                s.Offset as gzf,
                r.G_ID as site_sql_id                              
            FROM 
                tblFlowRatings r
            JOIN 
                tblFlowRating_Stats s
            ON 
                r.RatingNumber = s.Rating_Number;""", conn)
        #print(rating_points)
        rating_points = rating_points.sort_values(by = ["rating", "stage_rating"])
        #rating_points = rating_points.set_index("rating")
        
        # get sites
        parameter = "discharge"
        with sql_engine.begin() as conn:
                sites = pd.read_sql_query(f"select SITE_CODE as site_id, G_ID as site_sql_id from tblGaugeLLID WHERE STATUS = 'Active' AND FlowLevel = 'True' ORDER BY SITE_CODE ASC;", conn)
        

        sites.to_csv("example_data/sites.csv")


        rating_points = rating_points.merge(sites, on = "site_sql_id")
        rating_points = rating_points.drop(columns = ["site_sql_id"])
        rating_points.to_csv("example_data/rating_points.csv")


def rating_calculator(Ratings_value, site_sql_id): # for cache...should merge with others
            
        with sql_engine.begin() as conn:
                rating_points = pd.read_sql_query(f"SELECT WaterLevel as water_level_rating, Discharge as discharge, RatingNumber as rating_number "
                                            f"FROM tblFlowRatings "
                                            f"WHERE G_ID = '{str(site_sql_id)}' "
                                            f"AND RatingNumber = '{Ratings_value}' ;", conn)
        
        rating_points = rating_points.dropna()
        rating_points.sort_values(by=['water_level_rating'], inplace = True)
        rating_points["water_level_rating"] = round(rating_points["water_level_rating"], 2)
        rating_points["discharge"] = round(rating_points["discharge"], 2)
          # rating offset

        with sql_engine.begin() as conn:
                gzf = pd.read_sql_query(f"SELECT Offset as gzf "
                                            f"FROM tblFlowRating_Stats "
                                            f"WHERE Rating_Number = '{Ratings_value}';", conn)
           
            

        gzf = gzf.iloc[0, 0].astype(float)
        
        return rating_points, gzf

def get_sites(parameter):
        if parameter == "discharge":
                with sql_engine.begin() as conn:
                        available_sites = pd.read_sql_query(f"select SITE_CODE as site_id, G_ID as site_sql_id from tblGaugeLLID WHERE STATUS = 'Active' AND FlowLevel = 'True' ORDER BY SITE_CODE ASC;", conn)
    # site_sql_id = pd.read_sql_query(f"select G_ID as site_sql_id from tblGaugeLLID WHERE SITE_CODE = {site_number};", conn)
# this will need to change when there is more then just flowlevel
        #available_sites = available_sites["site_id"].values.tolist()
        return available_sites
#sites = get_sites("discharge")
#