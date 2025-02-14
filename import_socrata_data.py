import base64
import io
import pyodbc
import configparser
import dash
from dash import callback_context
from dash import html
from dash.dependencies import Input, Output, State
from dash import dcc
#from dash import html
from dash import dash_table
import pandas as pd
import dash_datetimepicker
import dash_daq as daq
from datetime import timedelta
from datetime import datetime
import numpy as np
from sqlalchemy import create_engine
import urllib
import plotly.graph_objs as go
import datetime as dt
from dash.exceptions import PreventUpdate
# long call back 
# https://dash.plotly.com/long-callbacks
## launch a new web browser
from web_browser import launch_web_broswer
# launch_web_broswer()
import dash_bootstrap_components as dbc
from data_cleaning import reformat_data, parameter_calculation, column_managment
from import_data import sql_statistics
import os
import pandas as pd
from sodapy import Socrata
import requests
import base64
from urllib.parse import urlencode
from datetime import datetime, timedelta
import plotly.express as px
#from plotly import graph_objs as go
#from plotly.graph_objs import *

#import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from io import StringIO
import os
from zoneinfo import ZoneInfo


from dotenv import load_dotenv

load_dotenv()
socrata_api_id = os.getenv("socrata_api_id")
socrata_api_secret = os.getenv("socrata_api_secret")

def site_metadata():
    #ith sql_engine.begin() as conn:
    #Available_Sites = pd.read_sql_query("select SITE_CODE, G_ID from tblGaugeLLID WHERE STATUS = 'Active' ORDER BY SITE_CODE ASC;", conn)

    #ef site_metadata(n_clicks):
    """reads site meta data, returns sites and gager list"""
    socrata_database_id = "g7er-dgc7"
    dataset_url = f"https://data.kingcounty.gov/resource/{socrata_database_id}.json"
    socrataUserPw = (f"{socrata_api_id}:{socrata_api_secret}").encode('utf-8')
    base64AuthToken = base64.b64encode(socrataUserPw)
    headers = {'accept': '*/*', 'Authorization': 'Basic ' + base64AuthToken.decode('utf-8')}
    #  "site, site_sql_id, latitude, longitude, gager",
    
    query_params = {
        "$select": "site, parameter",
    }
    
    encoded_query = urlencode(query_params)
    dataset_url = f"{dataset_url}?{encoded_query}"
    
    response = requests.get(dataset_url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        # create site/parameter list
        desired_order = ["site", "parameter"] # observation and observation_stage are kinda redundent at some point and should be clarified
        existing_columns = [col for col in desired_order if col in df.columns] 
        # Filter out columns that exist in the DataFrame   
        site_list = df[existing_columns].copy()
        # split parameter colum into list
        site_list["parameter"] = site_list["parameter"].str.split(", ")
        # create a row for each site/parameter column
        site_list = site_list.explode("parameter", ignore_index=True)
        
        # remove duplicates incase a parameter is repeated
        site_list = site_list.drop_duplicates()
        
        barometer_list = site_list.loc[site_list["parameter"] == "barometer"].copy()
        
        # convert to directory
        #df = df.groupby("site")["parameter"].apply(list).to_dict()
        # convert to list
        site_list = [f"{site} {parameter}" for site, parameter in zip(site_list["site"], site_list["parameter"])]
    
        barometer_list = barometer_list["site"].tolist()
        
        
        #gager_list = df["gager"].drop_duplicates().tolist()
        #df = df.to_json(orient="split")
    else:
        return dash.no_update 
    return site_list, barometer_list

#site_metadata()