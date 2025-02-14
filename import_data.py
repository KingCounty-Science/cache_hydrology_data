import base64
import datetime as dt
from datetime import timedelta
from datetime import datetime

import pyodbc
import configparser
import pandas as pd
from datetime import date

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
    if start_date != '' and end_date != '':
        
            
            start_date = pd.to_datetime(start_date).to_pydatetime()
            start_date = (start_date + timedelta(hours=(7))).strftime("%m/%d/%Y %H:%M")
            end_date = pd.to_datetime(end_date).to_pydatetime()
            end_date = (end_date + timedelta(hours=(7))).strftime("%m/%d/%Y %H:%M")
            
            if parameter == "FlowLevel" or parameter == "discharge":
              
                # QUERY Discharge
                select_statement = f"SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config[parameter]['datetime']}, 120)) as datetime, {config[parameter]['data']} as data, {config[parameter]['corrected_data']} as corrected_data, {config[parameter]['discharge']} as discharge, {config[parameter]['est']} as estimate, {config[parameter]['warning']} as warning "
            elif parameter == "barometer":
                # barometer (only has "data column")
                select_statement = f"SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config[parameter]['datetime']}, 120)) as datetime, {config[parameter]['corrected_data']} as corrected_data, {config[parameter]['est']} as estimate, {config[parameter]['warning']} as warning "
            
            # precipitation
            #elif parameter == "Precip" or parameter == "precip" or parameter == "rain" or parameter == "Rain":
            else:
            
                select_statement = f"SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config[parameter]['datetime']}, 120)) as datetime, {config[parameter]['data']} as data, {config[parameter]['corrected_data']} as corrected_data, {config[parameter]['est']} as estimate, {config[parameter]['warning']} as warning "
                
                # NEW CONN STRING
            with sql_engine.begin() as conn:
               
                df = pd.read_sql_query(f"{select_statement}"
                                    f"FROM {config[parameter]['table']} "
                                    F"WHERE G_ID = {str(site_sql_id)} "
                                    f"AND {config[parameter]['datetime']} BETWEEN ? and ? "
                                    f"ORDER BY {config[parameter]['datetime']} DESC", conn, params=[str(start_date), str(end_date)])
           
    else:
            df = pd.DataFrame()
            
    return df

def sql_import_all_datetimes(parameter, site_sql_id):
        if parameter == "FlowLevel" or parameter == "discharge":
                # QUERY Discharge
                select_statement = f"SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config[parameter]['datetime']}, 120)) as datetime, {config[parameter]['data']} as data, {config[parameter]['corrected_data']} as corrected_data, {config[parameter]['discharge']} as discharge, {config[parameter]['est']} as estimate "
        elif parameter == "barometer" or parameter == "Barometer":
                # barometer (only has "data column")
                select_statement = f"SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config[parameter]['datetime']}, 120)) as datetime, {config[parameter]['corrected_data']} as corrected_data, {config[parameter]['est']} as estimate "
            
        else:
            
                select_statement = f"SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config[parameter]['datetime']}, 120)) as datetime, {config[parameter]['data']} as data, {config[parameter]['corrected_data']} as corrected_data, {config[parameter]['est']} as estimate  "
                
                # NEW CONN STRING
        with sql_engine.begin() as conn:
            
                df = pd.read_sql_query(f"{select_statement}"
                                    f"FROM {config[parameter]['table']} "
                                    F"WHERE G_ID = {str(site_sql_id)} "
                                    f"ORDER BY {config[parameter]['datetime']} DESC", conn)
            # THIS ISNT THE COPY OF A Slice, datetime converted in sql statement
            #df["datetime"] = df["datetime"] - timedelta(hours=7)
        #else:
        #    df = pd.DataFrame()
            
        return df


def get_observations_join(parameter, site_sql_id, startDate, endDate):
        added_time_window = 12 # you want to pull in observations from before and after the start of the record as the observation could be taken 1 minute before start of record
        # convert to datetime
        #startDate = pd.to_datetime(startDate)
        #endDate = pd.to_datetime(endDate)

        # convert start/end date to utc time
        startDate = startDate + timedelta(hours=(7))
        endDate = endDate + timedelta(hours=(7))

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
        else:
                with sql_engine.begin() as conn:                                      
                #observations = pd.read_sql_query(f"""  SELECT Measurement_Number,                                                  Date_Time AS date,                                                                         Stage_Feet,                                                         tblFieldData.Parameter_Value, Comments FROM tblFieldVisitInfo INNER JOIN tblFieldData ON (tblFieldVisitInfo.FieldVisit_ID = tblFieldData.FieldVisit_ID) WHERE tblFieldVisitInfo.G_ID = {site_sql_id} AND tblFieldVisitInfo.Date_Time BETWEEN ? AND ? AND tblFieldData.Parameter = 2;""", conn, params=[str(startDate), str(endDate)])
                #observations = pd.read_sql_query(f"""   SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config['observation']['datetime']}, 120)) as datetime, {config['observation']['observation_number']} as observation_number, {config['observation']['observation_stage']} as observation_stage, tblFieldData.Parameter_Value as parameter_observation, Comments as comments
                #                                        FROM tblFieldVisitInfo INNER JOIN tblFieldData ON (tblFieldVisitInfo.FieldVisit_ID = tblFieldData.FieldVisit_ID) 
                #                                        WHERE tblFieldVisitInfo.G_ID = {site_sql_id} AND tblFieldVisitInfo.Date_Time BETWEEN ? AND ? AND tblFieldData.Parameter = {config[parameter_value]['observation_type']};""", conn, params=[str(startDate), str(endDate)])
                        observations = pd.read_sql_query(f"""   SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config['observation']['datetime']}, 120)) as datetime, {config['observation']['observation_number']} as observation_number, {config['observation']['observation_stage']} as observation_stage, tblFieldData.Parameter_Value as parameter_observation, Comments as comments
                                                        FROM tblFieldVisitInfo INNER JOIN tblFieldData ON (tblFieldVisitInfo.FieldVisit_ID = tblFieldData.FieldVisit_ID) 
                                                        WHERE tblFieldVisitInfo.G_ID = {site_sql_id} AND tblFieldData.Parameter = {config[parameter]['observation_type']};""", conn)
                if observations.empty: # if there are no parameter observations ie a waterlevel site
                        with sql_engine.begin() as conn: 
                        #observations = pd.read_sql_query(f"""   SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config['observation']['datetime']}, 120)) as datetime, {config['observation']['observation_stage']} as observation_stage, Comments as comments
                        #                               FROM tblFieldVisitInfo 
                        #                               WHERE tblFieldVisitInfo.G_ID = {site_sql_id} AND tblFieldVisitInfo.Date_Time BETWEEN ? AND ?;""", conn, params=[str(startDate), str(endDate)])   
                                observations = pd.read_sql_query(f"""   SELECT DATEADD(HOUR, -7, CONVERT(DATETIME, {config['observation']['datetime']}, 120)) as datetime, {config['observation']['observation_stage']} as observation_stage, Comments as comments
                                                        FROM tblFieldVisitInfo 
                                                        WHERE tblFieldVisitInfo.G_ID = {site_sql_id};""", conn)    
        return observations

def usgs_data_import(site_number, start_date, end_date):
        from data_cleaning import reformat_data
        if start_date != "" and end_date != "":
                print("start date", start_date)
                #start_date = datetime.fromisoformat(start_date.replace("Z", ""))
                #print("start date", start_date)
                start_date = start_date.strftime("%Y-%m-%dT%H:%M")
                print("start date", start_date)
                #end_date = datetime.fromisoformat(end_date.replace("Z", ""))
                end_date = end_date.strftime("%Y-%m-%dT%H:%M")
        
                # Example query parameters (replace with your values)
                #site_number = '12119000'
                #start_date = '2022-01-10T00:00'
                #end_date = '2023-01-10T00:00'
                
                # USGS API endpoint for streamflow data
                #api_url = f'https://waterdata.usgs.gov/nwis/dv?site_no={site_number}&format=json&startDT={start_date}&endDT={end_date}'
                #https://waterdata.usgs.gov/nwis/dv?site_no=12119000&format=json&startDT=2022-01-10&endDT=2024-01-20'
                # https://waterdata.usgs.gov/monitoring-location/12119000/#parameterCode=00065&period=P7D&showMedian=false
                
                url = f'https://waterservices.usgs.gov/nwis/iv/?sites={site_number}&parameterCd=00065&startDT={start_date}-07:00&endDT={end_date}-07:00&siteStatus=all&format=rdb'
                df = pd.read_csv(url, delimiter='\t', comment='#', skiprows=30, header=None, names=['site_number', 'datetime', 'timezone', 'comparison', 'status'])
                df = df.reset_index(drop=True)
              
                df = df[["datetime", "comparison"]]
                
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
        rating_points = rating_points.sort_values(by = ["rating", "stage_rating"])
        rating_points = rating_points.set_index("rating")
        

        #grouped = rating_points.groupby('RatingNumber')
        #individual_dfs = {rating_number: group for rating_number, group in grouped}
        rating_list = list(rating_points.index.unique())
    
        return rating_points, rating_list

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
        rating_calculation_status = Ratings_value
        return rating_calculation_status, rating_points, gzf

def get_sites(parameter):
        if parameter == "discharge":
                with sql_engine.begin() as conn:
                        available_sites = pd.read_sql_query(f"select SITE_CODE as site_id, G_ID as site_sql_id from tblGaugeLLID WHERE STATUS = 'Active' AND FlowLevel = 'True' ORDER BY SITE_CODE ASC;", conn)
    # site_sql_id = pd.read_sql_query(f"select G_ID as site_sql_id from tblGaugeLLID WHERE SITE_CODE = {site_number};", conn)
# this will need to change when there is more then just flowlevel
        available_sites = available_sites["site_id"].values.tolist()
        return available_sites
