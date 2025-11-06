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
from datetime import timedelta


def fill_timeseries(data, data_interval):
    data.drop_duplicates(subset=['datetime'], keep='first', inplace=True)
    data.dropna(subset=['datetime'], inplace=True)
    
    data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce', infer_datetime_format=True)
    data['datetime'] = data['datetime'].map(lambda x: dt.datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))
    data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce', infer_datetime_format=True)
  
    
    """data.set_index("datetime", inplace=True)
    data = data.resample(f'{data_interval}T').asfreq(fill_value="NaN")
    data.reset_index(level=None, drop=False, inplace=True)"""
   
 
    if "estimate" not in data.columns:
        data["estimate"] = "0"

    if "warning" not in data.columns:
        data["warning"] = "0"

    if "corrected_data" not in data.columns:
        data["corrected_data"] = np.nan

    def f(x):
        if x['data'] == "NaN": return str(1)
        else: return x['estimate']
    data['estimate'] = data.apply(f, axis=1)
    data['data'] = data['data'].astype(float, errors="ignore")
 
    
    return data


def to_observations(df, query_start_date, query_end_date):
    """adds query start date and query end date to df, thus adding observations to dataframe"""
    if query_start_date and pd.to_datetime(query_start_date).to_pydatetime() < df['datetime'].min():
            query_start_date =  pd.to_datetime(query_start_date).to_pydatetime()
            new_row = {'datetime': query_start_date}  # Replace 'A' with your desired column
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df = df.sort_values(by='datetime', ascending=True) 
            df.reset_index(level=None, drop=True, inplace=True)

    if query_end_date: #and pd.to_datetime(query_end_date).to_pydatetime() > df['datetime'].max():
            query_end_date =  pd.to_datetime(query_end_date).to_pydatetime()
            new_row = {'datetime': query_end_date}  # Replace '' with your desired column
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df = df.sort_values(by='datetime', ascending=True) 
            df.reset_index(level=None, drop=True, inplace=True)
    
    return df


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
        df = df.sort_values(by='datetime', ascending=True).reset_index(drop=True)
    if 'data' in df.columns:
        #df['data'] = df['data'].astype(float, errors="ignore")
        #df['data'] = df['data'].round(2)
        df['data'] = pd.to_numeric(df['data'], errors='coerce').round(2)
    #if 'corrected_data' in df.columns:
    #    df['corrected_data'] = df['corrected_data'].astype(float, errors="ignore")
    #    df['corrected_data'] = df['corrected_data'].round(2)
    if 'corrected_data' in df.columns:
        # Convert to float first
        df['corrected_data'] = pd.to_numeric(df['corrected_data'], errors='coerce')
        # Round, but preserve -999 exactly
        mask = df['corrected_data'] != -99
        df.loc[mask, 'corrected_data'] = df.loc[mask, 'corrected_data'].round(2)
    if 'observation' in df.columns:
        df['observation'] = df['observation'].astype(float, errors="ignore")
        df['observation'] = df['observation'].replace("", np.nan)
      
    if 'observation_stage' in df.columns:
        df['observation_stage'] = df['observation_stage'].astype(float, errors="ignore")
        df['observation_stage'] = df['observation_stage'].replace("", np.nan)
        df['observation_stage'] = df['observation_stage'].round(2)
        #df.loc[~pd.isna(df['observation_stage']), 'observation_stage']\
        #df['observation_stage'] = df['observation_stage'].round("", np.nan)
    if 'parameter_observation' in df.columns:
        df['parameter_observation'] = df['parameter_observation'].astype(float, errors="ignore")
        df['parameter_observation'] = df['parameter_observation'].replace("", np.nan)
    if 'offset' in df.columns:
        df['offset'] = df['offset'].astype(float, errors="ignore")
    if 'discharge' in df.columns:
        df['discharge'] = df['discharge'].astype(float, errors="ignore")
      
    if 'q_observation' in df.columns:
        df['q_observation'] = df['q_observation'].astype(float, errors="ignore")
        df['q_observation'] = df['q_observation'].replace("", np.nan)
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


def initial_column_managment(df):
      #search for columns and re arrange if present
        desired_order = ["datetime", "data", "corrected_data", "discharge", "estimate", "warning"]  # observation and observation_stage are kinda redundent at some point and should be clarified
        #comparison_columns = (df.columns[df.columns.str.contains('comparison')]).values.tolist()
        #comparison_columns = df.columns[df.columns.str.contains('comparison')].tolist()
        
        #if comparison_columns:
            
        #    desired_order.extend(comparison_columns)
     
        existing_columns = [col for col in desired_order if col in df.columns]         # Filter out columns that exist in the DataFrame   
        # Reorder the DataFrame columns
        df = df[existing_columns]
       
        return df
     
def column_managment(df):
     
    #search for columns and re arrange if present
        desired_order = ["datetime", "c_stage", "c_corrected_data", "c_water_level", "c_water_temperature", "c_discharge", "data", "corrected_data", "discharge", "observation", "observation_stage", "parameter_observation", "q_observation", "offset", "q_offset", "precent_q_change", "rating_number", "estimate", "warning", "comparison", "dry_indicator", "comments", "mean", "interpolated_data"]  # observation and observation_stage are kinda redundent at some point and should be clarified
        #comparison_columns = (df.columns[df.columns.str.contains('comparison')]).values.tolist()
        comparison_columns = df.columns[df.columns.str.contains('comparison')].tolist()
        
        if comparison_columns:
            
            desired_order.extend(comparison_columns)
     
        existing_columns = [col for col in desired_order if col in df.columns]         # Filter out columns that exist in the DataFrame   
        # Reorder the DataFrame columns
        df = df[existing_columns]
       
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

def parameter_calculation(df, data_level):
    
    if "field_observations" in df.columns:
            obs = "field_observations"
    elif "observations" in df.columns:
            obs = "observations"
    elif "observation" in df.columns:
            obs = "observation"
    elif "observation_stage" in df.columns:
            obs = "observation_stage"
    elif "observation_stage" not in df.columns and "parameter_observation" in df.columns:
            obs = "parameter_observation"
        #"""if observation not in df.columns:
        #        df[observation] = np.nan
        #if 'offset' not in df.columns:
        #        df["offset"] = np.nan"""
    else:
        obs = "no observation"
    
    if obs != "no observation":
        # Calculate offset only from valid data points
        valid_for_offset = (
            (df[data_level] != -999) & 
            (df.get('warning', 0) != 1) &
            (~df[obs].isna()) &  # Make sure obs data exists
            (~df[data_level].isna())  # Make sure sensor data exists
        )

        # Initialize offset series
        df['offset'] = np.nan

        # Calculate offset only where both obs and sensor data are valid
        df.loc[valid_for_offset, 'offset'] = (
            df.loc[valid_for_offset, obs] - df.loc[valid_for_offset, data_level]
        ).round(2)

        # Interpolate the offset to fill gaps
        df['offset'] = df['offset'].interpolate(method='linear', axis=0, limit_direction='both')

        # Apply correction only to valid data points
        valid_for_correction = (
            (df[data_level] != -999) & 
            (df.get('warning', 0) != 1) &
            (~df['offset'].isna())  # Make sure we have an offset to apply
        )

        # Initialize corrected_data with original sensor data
        if 'corrected_data' not in df.columns:
            df['corrected_data'] = df[data_level]

        # Apply correction only where valid
        df.loc[valid_for_correction, 'corrected_data'] = (
            df.loc[valid_for_correction, data_level] + 
            df.loc[valid_for_correction, 'offset']
        ).round(2)

        # Final offset calculation based on raw data (for metadata)
        # This should use the original raw data column, not corrected_data
        df['offset'] = (df[obs] - df["data"]).round(2)  # Assuming "data" is your raw sensor 
        """# calculate offset based on data level (data/corrected data)
        df['offset'] = df[obs] - df[data_level]
        df['offset'].interpolate( method='linear', inplace=True, axis=0, limit_direction='both')
        # this applys offset over the whole df but only corrects when not dry
        # Assuming you want to exclude -999 (dry/no-data) and warning flags
        valid_data = (df["corrected_data"] != -999) & (df.get('warning', 0) != 1)
        
        # Apply correction only where conditions are met
        df['corrected_data'] = np.where(
            valid_data,
            (df[data_level] + df['offset']).round(2),
            df["corrected_data"]  # Keep original value where conditions not met
        )
        # df['corrected_data'] = (df[data_level]+df['offset']).round(2) calculate corrected data ignoring dry data
        # re-calculate offset based on raw data for metadata
        df['offset'] = (df[obs] - df["data"]).round(2)"""
    df = df.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)
    
       
    return df

def add_comparison_site(comparison_site, comparison_site_sql_id, comparison_parameter, df, startDate, endDate):
     # add comparison df
        #start_date = df['datetime'].min()
        #start_date = (pd.to_datetime(startDate).to_pydatetime()) - timedelta(hours=(7))
        ##end_date = df['datetime'].max()
        #(pd.to_datetime(startDate).to_pydatetime()) - timedelta(hours=(7))
        
        #sql_import(parameter, site_sql_id, start_date, end_date) # can accept '' as start and end date

        if comparison_site.startswith("USGS"):
            from import_data import usgs_data_import
            
            df_comp = usgs_data_import(comparison_site, comparison_parameter, (pd.to_datetime(startDate)) - timedelta(hours=(0)), (pd.to_datetime(endDate)) - timedelta(hours=(0))) # convert start/end date from utc to pdt
            df_comp.rename(columns={"comparison": f"comparison {comparison_site} {comparison_parameter}"}, inplace=True)
            df = df.merge(df_comp, on="datetime", how = "outer")

        else:
            from import_data import sql_import
            if comparison_parameter == "stage":
                df_comp = sql_import("discharge", comparison_site_sql_id, startDate, endDate) # fx converts to PST and out of PST    
                df_comp = df_comp[['datetime', "corrected_data"]]
                df_comp.rename(columns={"corrected_data": f"comparison {comparison_site} {comparison_parameter}"}, inplace=True)

            elif comparison_parameter == "FlowLevel" or comparison_parameter == "discharge":
                df_comp = sql_import("discharge", comparison_site_sql_id, startDate, endDate) # fx converts to PST and out of PST  
                df_comp = df_comp[['datetime', "discharge"]]
                df_comp.rename(columns={"discharge": f"comparison {comparison_site} {comparison_parameter}"}, inplace=True)

            elif comparison_parameter == "rain" or comparison_parameter == "Precip":
                df_comp = sql_import(comparison_parameter, comparison_site_sql_id, startDate, endDate) # fx converts to PST and out of PST 
                df_comp = df_comp[['datetime', "corrected_data"]]
                #df_comp['datetime'] = pd.to_datetime(df_comp['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce', infer_datetime_format=True)
                df_comp['corrected_data'] = df_comp['corrected_data'].astype(float, errors="ignore")
                #df_comp['rolling_daily'] = round(df_comp['corrected_data'].rolling(window=96, min_periods = 1).mean(),2)
                #df_comp['rolling_daily'] = df_comp['corrected_data'].rolling(window=96, center = True, min_periods = 48, closed = 'both').mean()
                #df_comp['rolling_daily'] = df_comp['rolling_daily'].ffill()
                #df_comp['rolling_daily'] = df_comp['rolling_daily'].bfill()
                df_comp['rolling_bidaily'] = df_comp['corrected_data'].rolling(window=192, center = True, min_periods = 96, closed = 'both').mean()
                df_comp['rolling_bidaily'] = df_comp['rolling_bidaily'].ffill()
                df_comp['rolling_bidaily'] = df_comp['rolling_bidaily'].bfill()
                
                df_comp['rolling_weekly'] = df_comp['corrected_data'].rolling(window=754, center = True, min_periods = 377, closed = 'both').mean()
                df_comp['rolling_weekly'] = df_comp['rolling_weekly'].ffill()
                df_comp['rolling_weekly'] = df_comp['rolling_weekly'].bfill()

                df_comp['rolling_biweekly'] = df_comp['corrected_data'].rolling(window=1460, center = True, min_periods = 754, closed = 'both').mean()
                df_comp['rolling_biweekly'] = df_comp['rolling_biweekly'].ffill()
                df_comp['rolling_biweekly'] = df_comp['rolling_biweekly'].bfill()
                df_comp['rolling_monthly'] = df_comp['corrected_data'].rolling(window=2920, center = True, min_periods = 1460, closed = 'both').mean()
                df_comp['rolling_monthly'] = df_comp['rolling_monthly'].ffill()
                df_comp['rolling_monthly'] = df_comp['rolling_monthly'].bfill()
                #df[f'water_year_mean7d_{parameter}_rolling_average'] = df.groupby(["watershed", "water_year"])[f"{parameter}_daily_mean"].transform(lambda x: x.rolling(window = 7, min_periods = 7, center = True, closed = 'both').mean()).round(2)
                #df_comp['mean'] = df_comp['corrected_data'].transform(lambda x: x.rolling(window = 754, min_periods = 1, center = True, closed = 'both').mean()).round(2)
                df_comp['mean'] = df_comp[['rolling_bidaily','rolling_weekly', 'rolling_biweekly', 'rolling_monthly']].mean(axis = 1)
        
                df_comp = df_comp[["datetime", "mean"]]

                df_comp.rename(columns={"mean": f"comparison {comparison_site} {comparison_parameter}"}, inplace=True)
              


            else:
                df_comp = sql_import(comparison_parameter, comparison_site_sql_id, startDate, endDate) # fx converts to PST and out of PST 
                df_comp = df_comp[['datetime', "corrected_data"]]
                df_comp.rename(columns={"corrected_data": f"comparison {comparison_site} {comparison_parameter}"}, inplace=True)
            #df_comp.rename(columns={"corrected_data": f"comparison {comparison_site} {comparison_parameter}"}, inplace=True) 
           
            df = df.merge(df_comp, on="datetime", how = "outer")
    
       
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

### dry data
# sets corrected data to -99 and warning to 1
def set_dry_indicator_function(data, start_dry_indicator_range, end_dry_indicator_range):
    try:
        # Combine date and time
        start_dry_indicator_range = pd.to_datetime(start_dry_indicator_range)
        end_dry_indicator_range = pd.to_datetime(end_dry_indicator_range)
                        
        if start_dry_indicator_range >= end_dry_indicator_range:
            return data
                        
        mask = (data['datetime'] >= start_dry_indicator_range) & (data['datetime'] <= end_dry_indicator_range)
        affected_rows = mask.sum()
                        
        if affected_rows == 0:
            return data
                        
        # Apply dry warning
        data.loc[mask, 'warning'] = 1
        data.loc[mask, 'corrected_data'] = -99
        return data
    except Exception as e:
        return data
                        
def apply_dry_threshold_raw_function(data, dry_threshold_raw_input):
    data.loc[data["data"] <= dry_threshold_raw_input, "corrected_data"] = -99
    data.loc[data["data"] <= dry_threshold_raw_input, "warning"] = 1
    return data

def apply_dry_threshold_corrected_data_function(data, dry_threshold_corrected_data_input):
    data.loc[data["corrected_data"] <= dry_threshold_corrected_data_input, "corrected_data"] = -99
    data.loc[data["corrected_data"] <= dry_threshold_corrected_data_input, "warning"] = 1
    return data

def clear_dry_indicator_function(data, start_dry_indicator_range, end_dry_indicator_range):
    mask = (data['datetime'] >= start_dry_indicator_range) & (data['datetime'] <= end_dry_indicator_range)
    data.loc[mask, 'warning'] = 0
    data.loc[mask, 'corrected_data'] = np.nan  # Clear corrected_data
    return data