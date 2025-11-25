import pandas as pd
#pio.kaleido.scope.default_format = "svg"
import plotly.graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from scipy.stats import gmean, hmean

color_map = {
    'north_seidel_creek': r'#EF553B',
    'south_seidel_creek': r'#FFA15A',
    'webster_creek': r'#EECA3B',
    'cherry_trib': r'#636EFA',
    'fisher_creek': r'#AB63FA',
    'judd_creek': r'#19D3FA',
    'tahlequah_creek': r'#7E7DCD',
    'taylor_creek': r'#00CC96',
    'weiss_creek': r'#1CFFCE',
    1: r'#72B7B2',
    2: r'#F8A19F',
    'mean_discharge' : r'#316395', # dark blue
    "min7q_rolling_helper" : r"#2DE9FF",
    'min7q' : r"#00B5F7",
    'mean_temperature' : r"#D62728",
    'max_temperature' : r'#AF0038',
    'min_temperature' : r"#FF9DA6",

    'mean_conductivity' : r"#FECB52",
    'max_conductivity' : r'#FEAA16',
    'min_conductivity' : r"#F7E1A0",

    'mean_discharge' : r"#00B5F7",
    'max_discharge' : r'#2E91E5',
    'min_discharge' : r"rgb(179, 225, 207)",

    "high_pulse" : r"#DC587D",
    "low_pulse" : r"#F7E1A0",
   

    'mean_conductivity' : r'#FEAF16',
    'low_flow_peroid_water_temperature' : r"#F8A19F",
    'low_flow_peroid_box' : r'rgba(99, 110, 250, 0.3)',
    'summer_season_box' : r'rgba(99, 110, 250, 0.1)',

    #"water_year_7q" : r"rgba(204, 204, 204, 0.1)",
    "water_year_7q" : r"rgba(127, 60, 141, 0.9)",
    "min_7d" :  r"rgba(222, 172, 242, 0.9)",
    
    "low_flow_peroid_7q" : r"rgba(204, 204, 204, 0.3)",
    "summer_season_7q" : r"rgba(204, 204, 204, 0.6)"
    # Add more mappings as needed
    }

def site_interpolate():
    df = pd.read_csv("interpolate\\58a_daily.csv", parse_dates=[0], usecols=[0,1,2,3])
    df = df.rename(columns={df.columns[0]: "date"})
    df = df.rename(columns={df.columns[1]: "mean_flow"})
    df = df.rename(columns={df.columns[2]: "max_flow"})
    df = df.rename(columns={df.columns[3]: "min_flow"})
    df['mean_flow'] = pd.to_numeric(df['mean_flow'], errors='coerce')
    df['max_flow'] = pd.to_numeric(df['max_flow'], errors='coerce')
    df['min_flow'] = pd.to_numeric(df['min_flow'], errors='coerce')
    df = df.sort_values(by="date")
    #df.set_index('date', inplace=True)
  
    #fig = make_subplots(rows=1, cols=1, subplot_titles=df.index.unique(), specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]])
    #fig = make_subplots(rows=3, cols=2, specs=[[{"secondary_y": True}] * columns] * rows)
    fig = make_subplots(rows=3, cols=2, specs=[[{"secondary_y": True}] * 2] * 3)
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    #fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    #fig.update_layout(autosize=False,width=2000,height=3000)

    fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df["mean_flow"],
                    line=dict(color = color_map.get("mean_discharge", 'black'), width = 1),
                    name="mean_flow",showlegend=True,),row=1, col=1, secondary_y=False),
    fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df["max_flow"],
                    line=dict(color = color_map.get('max_discharge', 'black'), width = 1),
                    name="max",showlegend=True,),row=1, col=1, secondary_y=False),
    #
    fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df["min_flow"],
                    line=dict(color = color_map.get('min_discharge', 'black'), width = 1),
                    name="min",showlegend=True,),row=1, col=1, secondary_y=False),

    # long term week ie average flow for first week in august across dataset
    df["lt_week"] = df['date'].dt.strftime('%U')
    df["lt_week_avg"] = df.groupby(["lt_week"])["mean_flow"].transform("mean").round(2)
    # year week is average for for ie first week of month for the year august week 1,2,3,4 for year
    df["y_week"] = df['date'].dt.strftime('%Y-%U')
    df["y_week_avg"] = df.groupby(["y_week"])["mean_flow"].transform("mean").round(2)


    #
    fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df["lt_week_avg"],
                    line=dict(color = color_map.get('mean_temperature', 'black'), width = 1),
                    name="min",showlegend=True,),row=2, col=1, secondary_y=False),
    fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df["y_week_avg"],
                    line=dict(color = color_map.get('max_temperature', 'black'), width = 1),
                    name="min",showlegend=True,),row=2, col=1, secondary_y=False),

    # long term month ie average august value across dataset
    df["lt_month"] = df.date.dt.strftime('%m')
    df["lt_month_avg"] = df.groupby(["lt_month"])["mean_flow"].transform("mean").round(2)
    # year month ie monthly averages per year
    df["y_month"] = df.date.dt.strftime('%Y-%m')
    df["y_month_avg"] = df.groupby(["y_month"])["mean_flow"].transform("mean").round(2)


    #
    fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df["lt_month_avg"],
                    line=dict(color = color_map.get('mean_temperature', 'black'), width = 1),
                    name="min",showlegend=True,),row=3, col=1, secondary_y=False),
    fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df["y_month_avg"],
                    line=dict(color = color_map.get('max_temperature', 'black'), width = 1),
                    name="min",showlegend=True,),row=3, col=1, secondary_y=False),

    # long term mont
    df_month = df.drop_duplicates(subset='lt_month')
    min_month = df_month.loc[df_month["lt_month_avg"] == df_month["lt_month_avg"].min(), "lt_month"].item()
    #print(min_month)
    df_month['relative_wy'] = df_month['y_month'].apply(lambda x: pd.Period(x, "M") - (int(min_month) -1) )
    df_month['relative_wy'] = df_month['relative_wy'].dt.strftime('%m')
    df_month = df_month.sort_values(by = ['relative_wy'])
    #irst_month = df_month.loc[df_month['lt_month_avg'].min(), "lt_month"]
    #print(first_month)
    #df_month = df_month.sort_values(by="lt_month_avg")
    fig.add_trace(go.Scatter(
                    x=df_month['relative_wy'],
                    y=df_month["lt_month_avg"],
                    line=dict(color = color_map.get('mean_temperature', 'black'), width = 1),
                    name="min",showlegend=True,),row=3, col=2, secondary_y=False),
    #print(df_month)
    fig.show()

def average_data(df, site, site_sql_id, parameter, start_date, end_date):
    #columns = ["corrected_data"] # observation and observation_stage are kinda redundent at some point and should be clarified
    #comparison_columns = (df.columns[df.columns.str.contains('comparison')]).values.tolist()
    #comparison_columns = df.columns[df.columns.str.contains('comparison')].tolist()
        
    #if comparison_columns:
    #    columns.extend(comparison_columns)
    # 
    #    columns = [col for col in columns if col in df.columns]         # Filter out columns that exist in the DataFrame
    #  
    #    # Reorder the DataFrame columns
    #    df['average'] = df[columns].mean(axis=1)

    
    # normalize comparison sites to site value
    comparison_columns = df.columns[df.columns.str.contains('comparison')].tolist()
    if comparison_columns:
        for item in comparison_columns:
           
            mean = df[item].mean(skipna=True)
            stdev = df[item].std(skipna=True)
            df[f"standardized_{item}"] = (df[item] - mean) / stdev
    # standardize corrected data
        mean = df["corrected_data"].mean(skipna=True)
        stdev = df["corrected_data"].std(skipna=True)
        df[f"standardized_corrected_data"] = (df['corrected_data'] - mean) / stdev        
   
    # create average from corrected data and normalized comparison
        
   
        columns = ["standardized_corrected_data"] # observation and observation_stage are kinda redundent at some point and should be clarified
        extend_columns = df.columns[df.columns.str.contains('standardized_comparison')].tolist()
        columns.extend(extend_columns)
        columns = [col for col in columns if col in df.columns]         # Filter out columns that exist in the DataFrame
        # Reorder the DataFrame columns
        df['mean'] = round(df[columns].mean(axis=1),2)
        #df['average'] = gmean(df[columns], na_policy = 'omit', axis = 1)
        #df['gmean'] = df[columns].apply(lambda row: gmean(row, axis=0, nan_policy='omit'), axis=1)
        #df['hmean'] = df[columns].apply(lambda row: hmean(row, axis=0, nan_policy='omit'), axis=1)
        # drop standardized columns
        #comparison_columns = df.columns[df.columns.str.contains('standardized')].tolist()
        for item in columns:
            df = df.drop(columns=[item])

            #df.loc[pd.isna(df["data"]), 'data'] = df["comparison"]-df["interpolation_offset"]
    

        #df.loc[(~pd.isna(df["average"]) & ~pd.isna(df["corrected_data"])), "interpolation_offset"] = df["average"] - df['corrected_data']
        # fill offset
        #df["interpolation_offset"].interpolate( method='linear', inplace=True, axis=0, limit_direction='both')
        #df["average"] = round(df["average"] - df["interpolation_offset"], 2)
        #df.loc[pd.isna(df["corrected_data"]), 'average'] = df["average"]-df["interpolation_offset"]
        #df = df.drop(columns=["interpolation_offset"])
    

    from data_cleaning import column_managment
    
    
    df = column_managment(df)
      

    df = df.sort_values(by='datetime', ascending=True)

    return df

def cache_standardize_interpolation(df, site, site_sql_id, parameter, start_date, end_date):

    df = average_data(df, site, site_sql_id, parameter, start_date, end_date)
    if "interpolated_data" in df.columns:
        df = df.drop(columns=["interpolated_data"])
    # create new column of estimated data and corrected_data
    # Copy 'corrected_data' into 'interpolated_data'
    df["interpolated_data"] = df["corrected_data"]
    df['diff'] = np.nan
    df.loc[df["corrected_data"].notna(), 'diff'] = df['mean'] - df['corrected_data']
    df['diff'] = df['diff'].interpolate(method='linear', limit_direction='both', imit_area = "inside")
    #df['diff'] = df['diff'].ffill()
    #df['diff'] = df['mean'] - df['corrected_data']
    #df['diff']
    #df['diff'] = df['diff'].round(0)

    # Sort by 'mean' in descending order
    df = df.sort_values(["mean"], ascending=False)
    df['diff'] = df['diff'].round(1)
    df["interpolated_data"] = df[df['diff'].notna()].groupby(["diff"])["interpolated_data"].apply(lambda x: x.interpolate(method='linear', limit_direction='both', imit_area = "inside"))
    
    # imit_area = "inside"
    #df['diff'] = df['diff'].round(1)
    #df["interpolated_data"] = df.groupby(["diff"])["interpolated_data"].apply(lambda x: x.interpolate(method='linear', limit_direction='both', imit_area = "inside"))
    #df['diff'] = df['diff'].round(0)
    #df["interpolated_data"] = df.groupby(["diff"])["interpolated_data"].apply(lambda x: x.interpolate(method='linear', limit_direction='both', imit_area = "inside"))
    #  imit_area = "inside"
    # Interpolate only where 'mean' is not NA
    #df["interpolated_data"] = df["interpolated_data"].interpolate(method='linear', limit_direction='both')

    # Sort back by 'datetime' in ascending order
    df = df.sort_values(["datetime"], ascending=True)  



    # correct data
    df.loc[(~pd.isna(df["interpolated_data"]) & ~pd.isna(df["corrected_data"])), "interpolation_offset"] = df["interpolated_data"] - df['corrected_data']
    # fill offset
    df["interpolation_offset"].interpolate( method='linear', inplace=True, axis=0, limit_direction='both')
    df["interpolated_data"] = round(df["interpolated_data"] - df["interpolation_offset"], 2)
    #df.loc[pd.isna(df["corrected_data"]), 'interpolated_data'] = df["interpolated_data"]-df["interpolation_offset"]
    df['interpolated_data'] = df["interpolated_data"]-df["interpolation_offset"]

    df = df.drop(columns=["interpolation_offset"])
    return df

def save_interpolation(df, site, site_sql_id, parameter, start_date, end_date):
    # create offset
    df.loc[(~pd.isna(df["interpolated_data"]) & ~pd.isna(df["data"])), "interpolation_offset"] = df["interpolated_data"] - df['data']
    df["interpolation_offset"].interpolate( method='linear', inplace=True, axis=0, limit_direction='both')
    df.loc[(pd.isna(df["data"])) & (~pd.isna(df["interpolated_data"])), "data"] = df["interpolated_data"] - df["interpolation_offset"]
    df["data"] = round(df["data"], 2)
    df = df.drop(columns=["interpolation_offset"])
    #df["interpolated_data"] = round(df["interpolated_data"] - df["interpolation_offset"], 2)
    #df.loc[pd.isna(df["corrected_data"]), 'interpolated_data'] = df["interpolated_data"]-df["interpolation_offset"]
    #df['interpolated_data'] = df["interpolated_data"]-df["interpolation_offset"]

    #df = df.drop(columns=["diff"])
    #df.loc[pd.isna(df["corrected_data"]), "corrected_data"] = df["interpolated_data"]

    return df



def cache_comparison_interpolation(df, site, site_sql_id, parameter, start_date, end_date):






    #from import_data import sql_import_all_datetimes
    #df_historical = sql_import_all_datetimes(parameter, site_sql_id)
    #df = df.sort_values(by=["datetime"])
    
    ## calculate long term data
    #df_historical['long_term_mean'] = df_historical['corrected_data'].mean()
    #df_historical['long_term_max'] = df_historical['corrected_data'].max()
    #df_historical['long_term_min'] = df_historical['corrected_data'].min()
   
    ##calculate monthly data
    #df_historical["month"] = df_historical.datetime.dt.strftime('%m')
    #df_historical["month_mean"] = df_historical.groupby(["month"])["corrected_data"].transform("mean")
    #df_historical["month_max"] = df_historical.groupby(["month"])["corrected_data"].transform("max")
    #df_historical["month_min"] = df_historical.groupby(["month"])["corrected_data"].transform("min")

    #calculate monthly data
    #df_historical["month"] = df_historical.datetime.dt.strftime('%m')
    #df_historical["month_mean"] = df_historical.groupby(["month"])["corrected_data"].transform("mean")
    #df_historical["month_max"] = df_historical.groupby(["month"])["corrected_data"].transform("max")
    #df_historical["month_min"] = df_historical.groupby(["month"])["corrected_data"].transform("min")

        ### calculate weekly data
        ##calculate monthly data
        ##df_historical["week"] = df_historical.datetime.dt.strftime('%U')
        ##df_historical["week_mean"] = df_historical.groupby(["week"])["corrected_data"].transform("mean")
        ##df_historical["week_max"] = df_historical.groupby(["week"])["corrected_data"].transform("max")
        #df_historical["week_min"] = df_historical.groupby(["week"])["corrected_data"].transform("min")
    
    

    # calculate relative water year: while water year is oct-sep. we wnat to start with lowest longg term month somewhere between aug and oct
    #df_month = df_historical.drop_duplicates(subset='month')
    #min_month = df_month.loc[df_month["month_mean"] == df_month["month_mean"].min(), "month"].item()
    
    #df_historical['relative_month'] = df_historical['datetime'].apply(lambda x: pd.Period(x, "M") - (int(min_month) -1) )
    #df_historical['relative_month'] = df_historical['relative_month'].dt.strftime('%m')
    #df_historical = df_historical.sort_values(by = ['relative_month'])
   


    #df["month"] = df.datetime.dt.strftime('%m')
    # df relative wy
    #df['relative_month'] = df['datetime'].apply(lambda x: pd.Period(x, "M") - (int(min_month) -1) )
    #df['relative_month'] = df['relative_month'].dt.strftime('%m')
    #df = df.sort_values(by = ['relative_month'])


    #df_historical = df_historical.drop_duplicates(subset='month')   
    #df_historical = df_historical[["month", "relative_month", "month_mean", "month_min", "month_max"]]
   
    #df = df.merge(df_historical, left_on = ["month", "relative_month"], right_on = ["month",  "relative_month"], how = "outer")
   #comparison
    if "comparison" in df.columns:# if there is something to compare with
        df['data'] = pd.to_numeric(df['data'], errors='coerce')
        df['corrected_data'] = pd.to_numeric(df['corrected_data'], errors='coerce')
        df['comparison'] = pd.to_numeric(df['comparison'], errors='coerce')
        # create year
        #df["year"] = df.datetime.dt.strftime('%m')
        # create year column
        df["year"] = df.datetime.dt.strftime('%Y')
        # create month column
        df["month"] = df.datetime.dt.strftime('%m')
        # create week column
        df["week"] = df.datetime.dt.strftime('%U')

        # create day column
        df["day"] = df.datetime.dt.strftime('%j')
        
        # calculate month data
        df["month_mean"] = df.groupby(["month"])["corrected_data"].transform("mean")
        df["c_month_mean"] = df.groupby(["month"])["comparison"].transform("mean")

        # calculate daily data
        df["day_mean"] = df.groupby(["day"])["corrected_data"].transform("mean")
        df["c_day_mean"] = df.groupby(["day"])["comparison"].transform("mean")

        # calculate relative day wy
        # Create day column
        df["day"] = df.datetime.dt.strftime('%j')

        

        # Find minimum value in c_day_mean column
        #c_min_day = df.loc[df["c_day_mean"] == df["c_day_mean"].min(), "day"].item()

        # Calculate relative day
        # Create day column


        # Find minimum value in c_day_mean column
        c_min_month = df.loc[df["c_month_mean"] == df["c_month_mean"].min(), "month"].iloc[0]
        # Calculate relative day
        df['c_relative_month'] = df['datetime'].apply(lambda x: pd.Period(x, "M") - (int(c_min_month) - 1))
        df['c_relative_month'] = df['c_relative_month'].dt.strftime('%m')

        df['c_relative_water_year'] = (df['datetime'] + pd.DateOffset(months=12 - (int(c_min_month) - 1))).dt.strftime('%Y')
        df['water_year'] = (df['datetime'] + pd.DateOffset(months=3)).dt.strftime('%Y')
    #df = df.sort_values(by = ['relative_month'])

        # calculate difference
        #df['difference'] = abs(df['corrected_data']-df['comparison'])
        #df.loc[pd.isna(df['corrected_data']), "difference"] = np.nan
        #f.loc[pd.isna(df['comparison']), "difference"] = np.nan


        #df["90_precentile_year_month"] = df.groupby(["year", "month"])["corrected_data"].transform(lambda x: x.quantile(0.1)).round(2)
        #df["c_90_precentile_year_month"] = df.groupby(["year","month"])["corrected_data"].transform(lambda x: x.quantile(0.1)).round(2)
        
        
        
        # precent change
        # Calculate percentage change
        #df['pct_change'] = df['corrected_data'].pct_change(fill_method = None)
        #df['avg_pct_change'] = df['pct_change'].mean()

        #df['c_pct_change'] = df['comparison'].pct_change(fill_method = None)
        #df['c_avg_pct_change'] = df['c_pct_change'].mean()

        #df.loc[df['pct_change'] == 0, "direction"] = 0
        #df.loc[df['pct_change'] > 0, "direction"] = 1
        #df.loc[df['pct_change'] < 0, "direction"] = -1

        #df.loc[df['c_pct_change'] == 0, "c_direction"] = 0
        #df.loc[df['c_pct_change'] > 0, "c_direction"] = 1
        #df.loc[df['c_pct_change'] < 0, "c_direction"] = -1
       
        #df["long_term_mean"] = df.loc[~df["corrected_data"].isnull(), "long_term_mean"] = df["corrected_data"].mean()
        #df["c_long_term_mean"] = df.loc[~df["comparison"].isnull(), "c_long_term_mean"] = df["comparison"].mean()

        # high/low pulse count
        # identify pulses above long-term-mean *2 and long-term-mean/2 and stores to 'pulse'
        #df.loc[(~df["corrected_data"].isnull()) & (df['corrected_data'] <= df['long_term_mean']/2), "pulse"] = "low_pulse"
        #df.loc[(~df["corrected_data"].isnull()) & (df['corrected_data'] >= df['long_term_mean']*2), "pulse"] = "high_pulse"
        #df.loc[(~df["corrected_data"].isnull()) & (df['corrected_data'] >= df['long_term_mean']*2) & (df["pulse"].isnull()) , "pulse"] = "false"
        #df["pulse"].fillna('false', inplace=True)

        #df.loc[(~df["comparison"].isnull()) & (df['comparison'] <= df['c_long_term_mean']/2), "c_pulse"] = "low_pulse"
        #df.loc[(~df["comparison"].isnull()) & (df['comparison'] >= df['c_long_term_mean']*2), "c_pulse"] = "high_pulse"
        #df.loc[(~df["comparison"].isnull()) & (df['comparison'] >= df['c_long_term_mean']*2) & (df["c_pulse"].isnull()) , "c_pulse"] = "false"
        
        
        
        
        #["c_pulse"].fillna('false', inplace=True)

        #df["90_precentile_difference"] = df["90_precentile"]-df["c_90_precentile"]
        #### 95th precentile (low)
      
        #### 90th precentile (low)
        #df["90_precentile"] = df["corrected_data"].quantile(0.1).round(2)
        #df["c_90_precentile"] = df["comparison"].quantile(0.1).round(2)
       
        #### 90th precentile (low)
        #df["90_precentile_monthly"] = df.groupby(["month"])["corrected_data"].transform(lambda x: x.quantile(0.1)).round(2)
        #df["c_90_precentile_monthly"] = df.groupby(["month"])["comparison"].transform(lambda x: x.quantile(0.1)).round(2)
        #df["90_precentile_difference"] = (df["90_precentile_monthly"]-df["c_90_precentile_monthly"]).round(2)



        ### 10th precentile (high flow)
        #df["05_precentile_monthly"] = df.groupby(["month"])["corrected_data"].transform(lambda x: x.quantile(0.95)).round(2)
        #df["c_05_precentile_monthly"] = df.groupby(["month"])["comparison"].transform(lambda x: x.quantile(0.95)).round(2)
        #df["05_precentile_difference"] = (df["05_precentile_monthly"]-df["c_05_precentile_monthly"]).round(2)
       
      

        #print("search")
        #df["fill"] = np.nan
        #df.loc[(df["fill"].isnull()) & (~df['comparison'].isnull()) & (df['comparison'] <= df["c_90_precentile_monthly"]), "fill"] = df["comparison"] + df["90_precentile_difference"]
        #df.loc[(~df['comparison'].isnull()) & (df['comparison'] <= df["c_90_precentile_monthly"]), "90_precentile_fill"] = df["comparison"] + df["90_precentile_difference"]
        
        #df.loc[(df["fill"].isnull()) & (~df['comparison'].isnull()) & (df['comparison'] <= df["c_80_precentile_monthly_rolling"]), "fill"] = df["comparison"] + df["80_precentile_difference"]
        #df.loc[(~df['comparison'].isnull()) & (df['comparison'] <= df["c_80_precentile_monthly_rolling"]), "80_precentile_fill"] = df["comparison"] + df["80_precentile_difference"]

        #df.loc[(df["fill"].isnull()) & (~df['comparison'].isnull()) & (df['comparison'] >= df["c_05_precentile_monthly"]), "fill"] = df["comparison"] + df["05_precentile_difference"]
        #df.loc[(~df['comparison'].isnull()) & (df['comparison'] >= df["c_05_precentile_monthly"]), "05_precentile_fill"] = df["comparison"] + df["05_precentile_difference"]
        # rolling sum
        #df['rolling_sum'] = df['companion'].rolling(window=3, min_periods=1).sum()
        #df = df.sort_values(by=['c_relative_water_year', 'c_relative_month', "datetime"], ascending=True)
        df['c_rolling_sum_relative_water_year'] = df.groupby(['c_relative_water_year'])['comparison'].apply(lambda x: x.cumsum())
        #df['mean_rolling_sum'] = df.groupby(['day'])['rolling_sum'].transform('mean')

        #df = df.sort_values(["watershed", "water_year", 'datetime'], ascending=True)

        # grouping by watershed doesnt do anything because zero values for watershed wont be filled
        #df[f'average_{parameter}'] = df.groupby(["datetime"])[parameter].transform("mean").round(2)
       
        # this works the best so far because anything at day resolution cuts off peaks 
        # sort by average parameter then month
        # peaks
        from scipy.signal import find_peaks
        """# Find peaks
        peaks, properties = find_peaks(df['corrected_data'], prominence=True)
        df['is_peak'] = False
        df.loc[peaks, 'is_peak'] = True

        # Add a column for peak prominence
        df['peak_prominence'] = 0
        df.loc[peaks, 'peak_prominence'] = properties['prominences']
        """
        #df = df.sort_values([f"datetime"], ascending=True)
        peaks, properties = find_peaks(df['corrected_data'], prominence=True)

        # Create a column to mark peaks
        df['is_peak'] = False
        df.loc[peaks, 'is_peak'] = True

        # Add a column for peak prominence
        df['peak_prominence'] = np.nan
        df.loc[peaks, 'peak_prominence'] = properties['prominences']

        # Create columns to mark left and right tails
        df['tail'] = 'not a tail'

        # Iterate through each peak
        for peak, left_base, right_base in zip(peaks, properties['left_bases'], properties['right_bases']):
            df.loc[left_base:peak, 'tail'] = 'left tail'
            df.loc[peak:right_base, 'tail'] = 'right tail'


        """ peaks, properties = find_peaks(df['comparison'], prominence=True)
        df['c_is_peak'] = False
        df.loc[peaks, 'c_is_peak'] = True

        # Add a column for peak prominence
        df['c_peak_prominence'] = 0
        df.loc[peaks, 'c_peak_prominence'] = properties['prominences'] """
    
        peaks, properties = find_peaks(df['comparison'], prominence=True)

        # Create a column to mark peaks
        df['c_is_peak'] = False
        df.loc[peaks, 'c_is_peak'] = True

        # Add a column for peak prominence
        df['c_peak_prominence'] = np.nan
        df.loc[peaks, 'c_peak_prominence'] = properties['prominences']

        # Create columns to mark left and right tails
        df['c_tail'] = 'not a tail'

        # Iterate through each peak
        for peak, left_base, right_base in zip(peaks, properties['left_bases'], properties['right_bases']):
            df.loc[left_base:peak, 'c_tail'] = 'left tail'
            df.loc[peak:right_base, 'c_tail'] = 'right tail'
        
        df['estimate'] = pd.to_numeric(df['estimate'], errors='coerce')
        # set estimate to true
        df.loc[pd.isna(df["data"]), 'estimate'] = 1
        # delete estimated data
        df.loc[(df['estimate'].isnull() | df['estimate'] == 1), 'data'] = np.nan
        df.loc[(df['estimate'].isnull() | df['estimate'] == 1), 'corrected_data'] = np.nan
     
        df = df.sort_values([f"datetime"], ascending=True)
        # Fill NaN values where 'tail' is 'right_tail' using backward fill
        df.loc[(df['tail'] == 'right tail'), 'peak_prominence'] = df.loc[(df['tail'] == 'right tail'), 'peak_prominence'].bfill()

        # Fill NaN values where 'c_tail' is 'right_tail' using backward fill
        df.loc[(df['c_tail'] == 'right tail'), 'c_peak_prominence'] = df.loc[(df['c_tail'] == 'right tail'), 'c_peak_prominence'].bfill()

        # Fill NaN values where 'tail' is not 'not a tail' using forward fill
        df.loc[(df['tail'] != 'not a tail'), 'peak_prominence'] = df.loc[(df['tail'] != 'not a tail'), 'peak_prominence'].ffill()

        # Fill NaN values where 'c_tail' is not 'not a tail' using forward fill
        df.loc[(df['c_tail'] != 'not a tail'), 'c_peak_prominence'] = df.loc[(df['c_tail'] != 'not a tail'), 'c_peak_prominence'].ffill()

        df.loc[pd.isna(df["corrected_data"]), 'tail'] = np.nan
        df.loc[pd.isna(df["corrected_data"]), 'peak_prominence'] = np.nan
        df.loc[pd.isna(df["corrected_data"]), 'is_peak'] = np.nan
        

        df.loc[pd.isna(df["comparison"]), 'c_tail'] = np.nan
        df.loc[pd.isna(df["comparison"]), 'c_peak_prominence'] = np.nan
        df.loc[pd.isna(df["comparison"]), 'c_is_peak'] = np.nan

        df.loc[df["tail"] == "not a tail", "tail"] = np.nan
        df.loc[df["c_tail"] == "not a tail", "c_tail"] = np.nan
   


        #df.loc[df["is_peak"] == True, 'is_peak'] = df["corrected_data"]
        #df.loc[df["c_is_peak"] == True, 'c_is_peak'] = df["comparison"]

        # create a test fill column
        df["initial df"] = df["corrected_data"]
        df["test"] = df["corrected_data"]
        #df["peak_compare"] = df["corrected_data"]
        #df["highflow_compare"] = df["corrected_data"]


        
       
        
        # find peaks or close to them in dfs
        df = df.sort_values(["comparison"], ascending=False)
        df.loc[df["c_is_peak"] == True, "test"] = df.loc[df["c_is_peak"] == True, "test"].interpolate(method='linear', limit_direction='both')
        df = df.sort_values(["datetime"], ascending=True)
        # calculate interpolation offset
        #df.loc[(~pd.isna(df["comparison"]) & ~pd.isna(df["corrected_data"])), "interpolation_offset"] = df["comparison"] - df['corrected_data']
        # fill offset
        #df["interpolation_offset"].interpolate( method='linear', inplace=True, axis=0, limit_direction='both')
        #df.loc[pd.isna(df["corrected_data"]), "peak_compare"] = df["peak_compare"]-df["interpolation_offset"]
        #df = df.drop(columns=["interpolation_offset"])

        

        # Step 2: Update 'test' only where it is NaN
        #df['test'] = df['test'].combine_first(df['peak_fill'])
        #df.drop(columns=['peak_fill'], inplace=True)

        #df["interpolation_offset"].interpolate( method='linear', inplace=True, axis=0, limit_direction='both')
        #df.loc[pd.isna(df["corrected_data"]), 'peak_fill'] = df["peak_fill"]-df["interpolation_offset"]
        #df = df.drop(columns=["interpolation_offset"])
        
        # works well "c_rise_fall", 'c_difference' but still kinda jumpy
        # works well "c_rise_fall", 'c_difference', 'comparison' but a bit jumpy and very restrictive
        # test fill
        
        # fill values in peaks
        df = df.sort_values(["comparison"], ascending=True)

        df.loc[df["c_tail"] == "left tail", "test"] = df.loc[df["c_tail"] == "left tail", "test"].interpolate(method='linear', limit_direction='both')
        df.loc[df["c_tail"] == "right tail", "test"] = df.loc[df["c_tail"] == "right tail", "test"].interpolate(method='linear', limit_direction='both')
        



        #df["test"] = df.groupby(["c_tail"])["test"].apply(lambda x: x.interpolate(method='linear', limit_direction='both', limit_area = "inside"))
        
        df = df.sort_values([f"datetime"], ascending=True)

        # calculate interpolation offset test
        #df.loc[(~pd.isna(df["comparison"]) & ~pd.isna(df["corrected_data"])), "interpolation_offset"] = df["comparison"] - df['corrected_data']
        # fill offset
        #df["interpolation_offset"].interpolate( method='linear', inplace=True, axis=0, limit_direction='both')
        #df.loc[pd.isna(df["corrected_data"]), 'test'] = df["test"]-df["interpolation_offset"]
        #df = df.drop(columns=["interpolation_offset"])

        
        ### non peak

        
        # if row estimate is going from zero to 1 set 
        #update_observation_stage = 

        # Apply the lambda function to update observation_stage column
        #update_observation_stage = lambda x: x['corrected_data'] if x['estimate'] == 1 and x.shift(1)['estimate'] == 0 else x['observation_stage']
        # calculate interpolation offset
        df.loc[(~pd.isna(df["comparison"]) & ~pd.isna(df["data"])), "interpolation_offset"] = df["comparison"] - df['data']
        # fill offset
        df["interpolation_offset"].interpolate( method='linear', inplace=True, axis=0, limit_direction='both')
        df.loc[pd.isna(df["data"]), 'data'] = df["comparison"]-df["interpolation_offset"]
        df = df.drop(columns=["interpolation_offset"])

        # calculate interpolation offset
        df.loc[(~pd.isna(df["comparison"]) & ~pd.isna(df["corrected_data"])), "interpolation_offset"] = df["comparison"] - df['corrected_data']
        # fill offset
        df["interpolation_offset"].interpolate( method='linear', inplace=True, axis=0, limit_direction='both')
        df.loc[pd.isna(df["corrected_data"]), 'corrected_data'] = df["comparison"]-df["interpolation_offset"]
        df = df.drop(columns=["interpolation_offset"])
        
        #df = df.sort_values([f"comparison", "month"], ascending=True)
        #df["corrected_data"] = df["corrected_data"].fillna(method = "bfill")
        

        
        #df = df.sort_values(["c_relative_month", "comparison"], ascending=True)
        #df = df.sort_values(["comparison", "c_relative_month", "datetime"], ascending=True)

        ### the most stringent filter
        ## first sort by comparison relative month this creates a trend for the data to follow
        ## then sort by compariosn values within that month
        #df.sort_values(by=['c_relative_month', 'comparison'], ascending=[True, True], inplace=True)
        #df["corrected_data"].interpolate( method='linear', inplace=True, axis=0, limit_direction='both', limit_area = "inside")

        # limit_area = "inside"
       
        ## this works pretty well but jups at the start of wy
        #df.sort_values(by=['c_relative_month', 'comparison'], ascending=[True, True], inplace=True)
        #df.sort_values(by=['c_relative_month', 'comparison', 'c_day_mean'], ascending=[True, True, True], inplace=True)
        #df['corrected_data'] = df.groupby(['c_relative_water_year', 'c_relative_month'])['corrected_data'].apply(lambda x: x.interpolate(method='linear', limit_direction='both', limit_area = "inside"))
        #df.sort_values(by=['comparison', 'c_relative_month'], ascending=[True, True], inplace=True)
        
        #df['corrected_data'] = df.groupby(['c_relative_water_year'])['corrected_data'].apply(lambda x: x.interpolate(method='linear', limit_direction='both', limit_area = "inside"))
        
        # meh
        #df.sort_values(by=['c_relative_month', 'comparison'], ascending=[True, True], inplace=True)
        #df.sort_values(by=['c_relative_month', 'comparison', 'c_day_mean'], ascending=[True, True, True], inplace=True)
        #df['corrected_data'] = df.groupby(['c_relative_month', 'c_relative_water_year'])['corrected_data'].apply(lambda x: x.interpolate(method='linear', limit_direction='both', limit_area = "inside"))

        #df.sort_values(by=['comparison', 'c_relative_month'], ascending=[True, True], inplace=True)
        #df['corrected_data'] = df.groupby(['c_relative_water_year'])['corrected_data'].apply(lambda x: x.interpolate(method='linear', limit_direction='both', limit_area = "inside"))

        #df.sort_values(by=['comparison'], ascending=[True, True], inplace=True)
        #df['corrected_data'] = df.groupby(['c_relative_water_year'])['corrected_data'].apply(lambda x: x.interpolate(method='linear', limit_direction='both', limit_area = "inside"))

        # fill daily holes with rolling sum
        #df.sort_values(by=['datetime'], ascending=True, inplace=True)
        #df.sort_values(by=['mean_rolling_sum'], ascending=True, inplace=True)
        #df['corrected_data'] = df.groupby(['c_relative_water_year'])['corrected_data'].apply(lambda x: x.interpolate(method='linear', limit_direction='both', limit_area = "inside"))
        #df.sort_values(by=['comparison', 'c_relative_month'], ascending=[True, True], inplace=True)
        #df.sort_values(by=['comparison', 'corrected_data'], ascending=[True, True], inplace=True)
        #df['corrected_data'] = df.groupby(['c_relative_water_year'])['corrected_data'].apply(lambda x: x.interpolate(method='linear', limit_direction='both', limit_area = "inside"))
        #df['corrected_data'] = df['corrected_data'].interpolate(method='linear', limit_direction='both', limit_area = "inside"))
        # try only sorting by relative wy now

        #df['corrected_data'] = df.groupby(['c_relative_water_year'])['corrected_data'].apply(lambda x: x.ffill().bfill())
        #df['corrected_data'] = df.groupby(['c_relative_month'])['corrected_data'].apply(lambda x: x.interpolate(method='linear', limit_direction='both', limit_area = "inside"))
        
        # df["corrected_data"] = df["corrected_data"].fillna(method = "bfill")
        #df['corrected_data'] = df.groupby(['c_relative_water_year'])['corrected_data'].apply(lambda x: x.ffill().bfill())
        
        #df.to_csv(r"C:/Users/ihiggins/OneDrive - King County/Documents/compare.csv")
        
        df.to_csv("W:/STS/hydro/GAUGE/Temp/Ian's Temp/compare.csv")
        df = df.sort_values([f"datetime"], ascending=True)
        # drop year column
        df = df.drop(columns=['year'])
        # drop month column
        df = df.drop(columns=['month'])
        # drop week column
        df = df.drop(columns=['week'])
        # drop day column
        df = df.drop(columns=['day'])
        # drop month data
        df = df.drop(columns=['month_mean'])
        df = df.drop(columns=['c_month_mean'])
        # drop day data
        df = df.drop(columns=["day_mean"])
        df = df.drop(columns=["c_day_mean"])

        df = df.drop(columns=['c_relative_month'])
        


        df = df.drop(columns=['c_relative_water_year'])
        df = df.drop(columns=['water_year'])
        
        df = df.drop(columns=["is_peak"])
        df = df.drop(columns=["peak_prominence"])
        df = df.drop(columns=["tail"])
        df = df.drop(columns=["c_is_peak"])
        df = df.drop(columns=["c_peak_prominence"])
        df = df.drop(columns=["c_tail"])
        df = df.drop(columns=['c_rolling_sum_relative_water_year'])
      
        df = df.drop(columns=['test'])
        
 
        #df = df.drop(columns=[f'quantile'])
        #df = df.drop(columns=[f'c_quantile'])

        #df = df.drop(columns=[f"90_precentile"])
        #df = df.drop(columns=[f"c_90_precentile"])

        
    

        #df = df.drop(columns=[f"90_precentile_monthly"])
        #df = df.drop(columns=[f"c_90_precentile_monthly"])
        #df = df.drop(columns=[f"90_precentile_difference"])

        
        #df = df.drop(columns=[f"05_precentile_monthly"])
        #df = df.drop(columns=[f"c_05_precentile_monthly"])
        #df = df.drop(columns=[f"05_precentile_difference"])

        #df = df.drop(columns=[f"fill"])
        #df = df.drop(columns=[f"90_precentile_fill"])
        #df = df.drop(columns=[f"80_precentile_fill"])
        #df = df.drop(columns=[f"05_precentile_fill"])

        

       
       

    df = df.sort_values(by='datetime', ascending=True)
    #df.to_csv(r"C:/Users/ihiggins/OneDrive - King County/Documents/df_relative_wy.csv")
    #df = df.drop(columns=['month', 'relative_month', 'month_mean', 'month_min', 'month_max'])
    #if "comparison" in df.columns():
    #    df = df.drop(columns=['difference', 'c_month_mean', 'week', 'c_week_mean'])
    print(f"interpolate {parameter} complete")
    
    return(df)

def resample_data(df, data_interval):
    # once I added the comparison twice which caused duplicate datetimes and resulted in this failing
    """ reamples datetime to specified interval, interpolates raw data (data) and fills in estimate and warning"""
    df.set_index("datetime", inplace=True)
    df = df.resample(f'{data_interval}T').asfreq(fill_value=np.nan)
    #df = df.resample(f'{data_interval}T').interpolate(method='linear')
    df.reset_index(level=None, drop=False, inplace=True)

    df["data"] = df['data'].interpolate(method='linear')
    df["data"] = round(df["data"], 2)
    if 'estimate' in df.columns:
        df["estimate"] = df["estimate"].ffill()
        df["estimate"] = df["estimate"].bfill()

    if "warning" in df.columns:
        df["warning"] = df["warning"].ffill()
        df["warning"] = df["warning"].bfill()
    if "non_detect" in df.columns:
        df["non_detect"] = df["non_detect"].ffill()
        df["non_detect"] = df["non_detect"].bfill()
    return df

def basic_interpolation(df, method, set_limit, limit_number, direction, area, interp_data, interp_corrected_data):
    if 'on' in interp_data:
        if set_limit == "limit": # limit consecutive na values
            df["data"] = df["data"].interpolate(method=method, limit = limit_number, limit_direction = direction, limit_area = area)
        if set_limit == "no limit": # do not limit consecutive na values
            df["data"] = df["data"].interpolate(method=method, limit_direction = direction, limit_area = area)
        df["data"] = round(df["data"], 2)
        if 'estimate' in df.columns:
            df["estimate"] = df["estimate"].ffill()
            df["estimate"] = df["estimate"].bfill()

        if "warning" in df.columns:
            df["warning"] = df["warning"].ffill()
            df["warning"] = df["warning"].bfill()
        if "non_detect" in df.columns:
            df["non_detect"] = df["non_detect"].ffill()
            df["non_detect"] = df["non_detect"].bfill()
    if 'on' in interp_corrected_data:
        if set_limit == "limit": # limit consecutive na values
            df["corrected_data"] = df["corrected_data"].interpolate(method=method, limit = limit_number, limit_direction = direction, limit_area = area)
        if set_limit == "no limit": # do not limit consecutive na values
            df["corrected_data"] = df["corrected_data"].interpolate(method=method, limit_direction = direction, limit_area = area)
        df["corrected_data"] = round(df["corrected_data"], 2)
    return df

def run_basic_forward_fill(df, fill_limit, fill_limit_number, fill_area, interp_data, interp_corrected_data):
    if 'on' in interp_data:
        if fill_limit == "limit":
            df["data"].ffill(inplace = True, limit = fill_limit_number)
        if fill_limit == "no limit":
            df["data"].ffill(inplace = True)

        df["data"] = round(df["data"], 2)
        if 'estimate' in df.columns:
            df["estimate"] = df["estimate"].ffill()

        if "warning" in df.columns:
            df["warning"] = df["warning"].ffill()
        if "non_detect" in df.columns:
            df["non_detect"] = df["non_detect"].ffill()
    if 'on' in interp_corrected_data:
        if fill_limit == "limit":
            df["corrected_data"].ffill(inplace = True, limit = fill_limit_number)
        if fill_limit == "no limit":
            df["corrected_data"].ffill(inplace = True)

    #df.loc[df["corrected_data"] != -99, "corrected_data"].round(2)
  
    return df
def run_basic_backward_fill(df, fill_limit, fill_limit_number, fill_area, interp_data, interp_corrected_data):
    if 'on' in interp_data:
        if fill_limit == "limit":
            df["data"].bfill(inplace = True, limit = fill_limit_number)
        if fill_limit == "no limit":
            df["data"].bfill(inplace = True)

        df["data"] = round(df["data"], 2)
        if 'estimate' in df.columns:
            df["estimate"] = df["estimate"].bfill()

        if "warning" in df.columns:
            df["warning"] = df["warning"].bfill()

        if "non_detect" in df.columns:
            df["non_detect"] = df["non_detect"].bfill()
    if 'on' in interp_corrected_data:
        if fill_limit == "limit":
            df["correcte_data"].bfill(inplace = True, limit = fill_limit_number)
        if fill_limit == "no limit":
            df["corrected_data"].bfill(inplace = True)

    #df.loc[df["corrected_data"] != -99, "corrected_data"].round(2)
        

    return df