# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 08:22:14 2022

@author: IHiggins
"""

import datetime as dt
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from scipy import stats, interpolate


def fill_timeseries(data, data_interval):
    data.drop_duplicates(subset=['datetime'], keep='first', inplace=True)
    data.dropna(subset=['datetime'], inplace=True)
    #data.to_csv(r"W:/STS/hydro/GAUGE/Temp/Ian's Temp/cache_error_df.csv")
    #data['estimate'] = 0
    # make sure it is in datetime
    data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce', infer_datetime_format=True)
    data['datetime'] = data['datetime'].map(lambda x: dt.datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))
    data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce', infer_datetime_format=True)
  
    
    # resample
    #data['data'] = data['data'].astype(float, errors="ignore")
    interval = (60/len((data["datetime"].dt.strftime('%M').copy()).drop_duplicates()))
  
    if interval < 5:
        interval = 1
    elif interval >= 5 and interval < 10:
        interval = 5
    elif interval >= 10 and interval < 20:
        interval = 15
    elif interval <=20 and interval <= 50:
        interval = 30
    elif interval > 50:
        interval = 60
    print("interval ", interval)
    print("interval base")
    print(((data["datetime"].dt.strftime('%M').copy()).drop_duplicates()).sort_values(ascending=False))
    print("interval len")
    print(len((data["datetime"].dt.strftime('%M').copy()).drop_duplicates()))
    data.set_index("datetime", inplace=True)
    data = data.resample(f'{interval}T').asfreq(fill_value="NaN")
    #data = data.interpolate(method='linear', limit=4)
    #data = data.resample(f'{data_interval}T').interpolate(method='linear', limit=4)
    #data = data.interpolate(method='linear', limit=4)
    data.reset_index(level=None, drop=False, inplace=True)
   
    #data['estimate'] = 0

    """data.set_index('datetime', inplace=True)

    # Interpolate missing values
    data = data.interpolate(method='linear', limit=4)

    # Reset the index if needed
    data.reset_index(inplace=True)"""
 
    if "estimate" not in data.columns:
        data["estimate"] = "0"

    if "warning" not in data.columns:
        data["warning"] = "0"
    

    def f(x):
        if x['data'] == "NaN": return str(1)
        else: return x['estimate']
    data['estimate'] = data.apply(f, axis=1)
    data['data'] = data['data'].astype(float, errors="ignore")
    #data['data'].interpolate(inplace=True)
    
    return data




def data_conversion(df, parameter):
    # water level conversion
    # if water level is centegrade
    if parameter == "WaterTemp" or parameter == "water_temperature":
        if df['data'].mean() < 20:
            df = df
           
        else:
            df['data'] = (df['data']-32)*(5/9)
    else:
        df = df
    
    return df

def reformat_data(df):
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce', infer_datetime_format=True)
    if 'data' in df.columns:
        df['data'] = df['data'].astype(float, errors="ignore")
        df['data'] = df['data'].round(2)
    if 'corrected_data' in df.columns:
        df['corrected_data'] = df['corrected_data'].astype(float, errors="ignore")
        df['corrected_data'] = df['corrected_data'].round(2)
    if 'observation' in df.columns:
        df['observation'] = df['observation'].astype(float, errors="ignore")
    if 'observation_stage' in df.columns:
        df['observation_stage'] = df['observation_stage'].astype(float, errors="ignore")
    if 'parameter_observation' in df.columns:
        df['parameter_observation'] = df['parameter_observation'].astype(float, errors="ignore")
    if 'offset' in df.columns:
        df['offset'] = df['offset'].astype(float, errors="ignore")
    if 'discharge' in df.columns:
        df['discharge'] = df['discharge'].astype(float, errors="ignore")
    if 'q_observation' in df.columns:
        df['q_observation'] = df['q_observation'].astype(float, errors="ignore")
    if 'Discharge_Rating' in df.columns:
        df['Discharge_Rating'] = df['Discharge_Rating'].astype(float, errors="ignore")
    if 'q_offset' in df.columns:
        df['q_offset'] = df['q_offset'].astype(float, errors="ignore")
    if 'comparison' in df.columns:
        df['comparison'] = df['comparison'].astype(float, errors="ignore")
    if 'estimate' in df.columns:
        df['estimate'] = df['estimate'].astype(int, errors="ignore")
    if 'warning' in df.columns:
        df['warning'] = df['warning'].astype(int, errors="ignore")
    if 'measurement_number' in df.columns:
        df['measurement_number'] = df['measurement_number'].astype(float, errors="ignore")
    if 'discharge_observation' in df.columns:
        df['discharge_observation'] = df['discharge_observation'].astype(float, errors="ignore")

    return df

def style_formatting(): # I think this is for an ag grid and dont use this
    style_data_conditional = ({'if': {'column_id': 'comparison',}, 'backgroundColor': 'rgb(222,203,228)','color': 'black'},
                              {'if': {'filter_query': '{parameter_observation} > 0','column_id': 'parameter_observation'},  'backgroundColor': 'rgb(179,226,205)','color': 'black'},
                              {'if': {'filter_query': '{parameter_observation} > 0','column_id': 'datetime'},  'backgroundColor': 'rgb(179,226,205)','color': 'black'},
                              #{'if': {'filter_query': '{parameter_observation} > 0','column_id': 'offset'},  'backgroundColor': 'rgb(179,226,205)','color': 'black'},
                              {'if': {'filter_query': '{observation_stage} > 0','column_id': 'observation_stage'},  'backgroundColor': 'rgb(179,226,205)','color': 'black'},
                              {'if': {'filter_query': '{observation_stage} > 0','column_id': 'datetime'},  'backgroundColor': 'rgb(179,226,205)','color': 'black'},
                              {'if': {'filter_query': '{observation_stage} > 0','column_id': 'offset'},  'backgroundColor': 'rgb(179,226,205)','color': 'black'},)
                                   
                                   # {'if': {'filter_query': '{{parameter_observation}} > {0}'),'backgroundColor': '#FF4136','color': 'white'},
               

    return style_data_conditional

def parameter_calculation(df, observation, data_level):
      
        if observation not in df.columns:
            df[observation] = "nan"
        if 'offset' not in df.columns:
            df["offset"] = "nan"
        df['offset'] = df[observation] - df[data_level]
        df['offset'].interpolate( method='linear', inplace=True, axis=0, limit_direction='both')
        df['corrected_data'] = (df[data_level]+df['offset']).round(2)
        df['offset'] = (df[observation] - df["data"]).round(2)
       
        return df

def add_comparison_site(comparison_site_sql_id, comparison_parameter, df):
     # add comparison df

        from import_data import sql_import
        #print("compare site", comparison_site_sql_id, "compare parm", comparison_parameter)
        start_date = df['datetime'].min()
        end_date = df['datetime'].max()
        df_comparison = sql_import(comparison_parameter, comparison_site_sql_id, start_date, end_date) # fx converts to PST and out of PST
        #sql_import(parameter, site_sql_id, start_date, end_date) # can accept '' as start and end date
        df_comparison = df_comparison[['datetime', "corrected_data"]]
        df_comparison.rename(columns={"corrected_data": "comparison"}, inplace=True)
        df = df.merge(df_comparison, on="datetime", how = "outer")
        #df = df_comparison.merge(df, on="datetime", how = "inner")
        return df

def rating_curve_equations(df, gzf):
    df_sort = df.sort_values(by=['observation_stage']).copy()
    
    # poly fit
    #poly_fit_equation = np.poly1d(np.polyfit(((df_sort['observation_stage']-gzf)).to_numpy(), (df_sort['discharge_observation']).to_numpy(), 2))
   # df_sort['poly_fit_line'] = poly_fit_equation(df_sort['observation_stage']-gzf)

    # linear regression
    # linregressor (x,y) x = discharge y =
    #inear_regression_equation =  stats.linregress((df_sort['observation_stage']-gzf).to_numpy(), (df_sort['discharge_observation']).to_numpy())
    #df_sort['linear_regression_line']  = (((inear_regression_equation.intercept)-gzf) + (inear_regression_equation.slope*(df_sort['observation_stage'])-gzf))


    #linear_regression_log =  stats.linregress(np.log(df_sort['observation_stage']).to_numpy(), np.log(df_sort['discharge_observation']).to_numpy())
    #df_sort['linear_regression_log']  = (((linear_regression_log.intercept)) + ( linear_regression_log.slope*(df_sort['observation_stage'])))
    #linear_regression_log_gzf =  stats.linregress(np.log(df_sort['observation_stage']-gzf).to_numpy(), np.log(df_sort['discharge_observation']).to_numpy())
    #df_sort['linear_regression_log_gzf']  = (((linear_regression_log_gzf.intercept)) + (linear_regression_log_gzf.slope*(df_sort['observation_stage'])))

   



    print("stats")
    #print(f" Linear Regression intercept {linear_regression_equation.intercept} slope {linear_regression_equation.slope}")
    #print(f" Linear Regression Log intercept {linear_regression_log.intercept} slope {linear_regression_log.slope}")
    #print(f" Linear Regression Log  GZF intercept {linear_regression_log_gzf.intercept} slope {linear_regression_log_gzf.slope}")
     # line between points
    
    #interpolate_function = interpolate.interp1d(((df_sort['observation_stage']-gzf).to_numpy()), df_sort['discharge_observation'].to_numpy(), bounds_error = True)
    df_sort = df_sort.sort_values(by=['observation_stage']).copy()
    #df_x = df_sort.groupby('observation_stage', as_index=False).mean()
    interpolate_function = interpolate.interp1d(((df_sort['observation_stage']-gzf).to_numpy()), df_sort['discharge_observation'].to_numpy(), bounds_error = False)
    interpolate_stage = interpolate.interp1d(df_sort['discharge_observation'].to_numpy(), ((df_sort['observation_stage']-gzf).to_numpy()), bounds_error = False)
    df_sort['interpolate'] = interpolate_function((df_sort['observation_stage']-gzf))
    #df_x = df_sort.sort_values(by = ['observation_stage', 'discharge_observation'], ascending = [True, True], na_position = 'first')
    
    #df_x = df_x.reindex(['observation_stage'])
    #df_x = df_x.sort_values(by = 'discharge_observation')

    #df_sort = df_sort.sort_values(by=['measurement_number']).copy()
    return df_sort, interpolate_function, interpolate_stage
'''
data = pd.read_csv(r"W:/STS/hydro/GAUGE/Temp/Ian's Temp/clean_check.csv", index_col=0)
fill_timeseries(data)
print(f"delta {fill_timeseries.delta}")
print(f"interval {fill_timeseries.interval}")
'''