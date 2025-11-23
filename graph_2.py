import pandas as pd
import datetime as dt
import configparser
import os
import numpy as np


from scipy.signal import find_peaks, find_peaks_cwt
#import dash_core_components as dcc
from dash import dcc
from dash import html
import plotly.io as pio
pio.kaleido.scope.default_format = "svg"
from plotly.subplots import make_subplots
import numpy as np
import datetime as dt
import plotly.graph_objects as go
from sklearn import preprocessing

if not os.path.exists("images"):
    os.mkdir("images")

config = configparser.ConfigParser()
config.read('gdata_config.ini')
color_map = {
    'north_seidel_creek': r'rgb(0, 0, 255)',
    'south_seidel_creek': r'rgb(173, 216, 230)',
    'webster_creek': r'rgb(118, 78, 159)',

    'fisher_creek': r'rgb(221, 204, 119)',#r'rgb(255, 0, 0)',
    'weiss_creek': r'rgb(255, 192, 203)',
    'cherry_trib': r'rgb(255, 237, 111)',
    'judd_creek': r'rgb(237, 110, 90)',#r'rgb(220, 20, 60)',
    'tahlequah_creek': r'rgb(253, 180, 98)',
    'taylor_creek': r'rgb(255, 99, 71)',
    'data': r'rgba(102, 102, 102, 0.4)',
    'corrected_data': r'rgba(29, 105, 150, 0.7)',
    'comparison' : r'rgba(152, 78, 163, 0.7)', # 0  is fully transparent , 1 is opaque
    1: r'#72B7B2',
    2: r'#F8A19F',
    'mean_discharge' : r'#316395', # dark blue
    "min7q_rolling_helper" : r"#2DE9FF",
    'min7q' : r"#00B5F7",
    'water_temperature' : r"#D62728",
    'temperature' : r"#D62728",
    'mean_temperature' : r"#D62728",
    'max_temperature' : r'#AF0038',
    'min_temperature' : r"#FF9DA6",
    'raw_water_temperature' : r"#D62728", # mean water temperature
    'corrected_water_temperature' : r'#AF0038', # max water temperature

    'conductivity' : r"#FECB52",
    'mean_conductivity' : r"#FECB52",
    'max_conductivity' : r'#FEAA16',
    'min_conductivity' : r"#F7E1A0",

    'discharge' : r"rgba(82, 188, 163, 0.6)",
    'mean_discharge' : r"#00B5F7",
    'max_discharge' : r'#2E91E5',
    'min_discharge' : r"rgb(179, 225, 207)",

    "high_pulse" : r"#DC587D",
    "low_pulse" : r"#F7E1A0",
   

    'mean_conductivity' : r'#FEAF16',
    'low_flow_peroid_water_temperature' : r"#F8A19F",
    'low_flow_peroid_box' : r'rgba(99, 110, 250, 0.3)',
    'summer_season_box' : r'rgba(99, 110, 250, 0.3)',

    #"water_year_7q" : r"rgba(204, 204, 204, 0.1)",
    "water_year_7q" : r"rgba(127, 60, 141, 0.9)",
    "min_7d" :  r"rgba(222, 172, 242, 0.9)",
    
    "low_flow_peroid_7q" : r"rgba(204, 204, 204, 0.3)",
    "summer_season_7q" : r"rgba(204, 204, 204, 0.6)",

    "field_observation" : r"rgb(136, 136, 136)",
    "dry_indicator": r'#AB63FA',

    "mean" : r"rgb(255, 165, 0)", # orange
    "gmean" : r"rgba(255, 200, 140, 0.5)", # light orange
    "hmean" : r"rgba(255, 140, 0, 0.5)", # dark orange
   

    # Add more mappings as needed
    #'north_seidel_creek': r'#EF553B',
    #'south_seidel_creek': r'#FFA15A',
    #'webster_creek': r'#EECA3B',
    #'cherry_trib': r'#636EFA',
    #'fisher_creek': r'#AB63FA',
    #'judd_creek': r'#19D3FA',
    #'tahlequah_creek': r'#7E7DCD',
   # 'taylor_creek': r'#00CC96',
    #'weiss_creek': r'#1CFFCE',
    1: r'#72B7B2',
    2: r'#F8A19F',
    # Add more mappings as needed
    }

# site code = site_sql_id
def parameter_graph(df, site_selector_value, site_code, site_name, parameter, comparison_data, primary_min, primary_max, secondary_min, secondary_max, normalize_data, statistics, display_statistics):
    #df = df.sort_values(by='datetime', ascending=True) # order is required for observations to plot
    data_axis = True # uncorrected data
    base_parameter_axis = False # water level/stage for discharge
    derived_parameter_axis = False # stage
    comparison_axis = True
    statistics = pd.read_json(statistics, orient="split")
    if parameter == "FlowLevel" or parameter == "discharge":
        base_parameter = "water_level"
        derived_parameter = "discharge"
        data_axis = False
        base_parameter_axis = False
        derived_parameter_axis = True
        comparison_axis = True
        primary_y_title = "stage (wl feet)" # stage
        secondary_y_title = "discharge (cfs)" # discharge


    elif parameter == "LakeLevel" or parameter == "Piezometer" or parameter == "water_level":
        base_parameter = "water_level"
        derived_parameter = "water_level"
        primary_y_title = "water level (wl feet)"
        secondary_y_title = "water level (wl feet)"
        data_axis = True # uncorrected data
        base_parameter_axis = False # water level/stage for discharge
        derived_parameter_axis = False # stage

    else:
        base_parameter = parameter
        derived_parameter = parameter
        primary_y_title = f"{parameter.replace('_', ' ')} ({config[parameter]['unit']})"
        secondary_y_title = f"{parameter.replace('_', ' ')} ({config[parameter]['unit']})"
        #elif comparison_axis == "primary":
        #    comparison_axis = False
        #elif comparison_axis == "secondary":
        #    comparison_axis = True

    #if primary_min == " ":
    #    primary_min = df.select_dtypes(include='number').columns
    if primary_min == " ":
        primary_min = df[df.select_dtypes(include='number').columns].min()

    if secondary_min == " ":
        secondary_min = df[df.select_dtypes(include='number').columns].min()

    if primary_max == " ":
        primary_max = df[df.select_dtypes(include='number').columns].max()

    if secondary_max == " ":
        secondary_max = df[df.select_dtypes(include='number').columns].max()
    
    try:
        from data_cleaning import reformat_data
        df = reformat_data(df)
        comparison_data = reformat_data(comparison_data)
    except:
        pass

    ### filter out dry data
    dry = df.copy()
    dry.loc[dry['non_detect'] != "1", 'corrected_data'] = np.nan
    df.loc[df['non_detect'] == "1", 'corrected_data'] = np.nan

    # get site number?
     # replace _ with space
    #subplot_titles = [value.replace("_", " ") for value in df.index.unique()]
    subplot_titles = parameter
    number_of_rows = 1
    number_of_columns = 1
    title_font_size = 40 # plot title
    annotation_font_size = 40 # subplot titels are hardcoded as annotations
    show_subplot_titles = False # supblot titles are hardcoded as annotations
    font_size = 20 # axis lables, numbers, offset information
    show_chart_title = True
    chart_title = f"{site_name.replace('_', ' ')} {(derived_parameter.replace('_', ' '))} {dt.datetime.strftime(df['datetime'].min(), '%Y-%m-%d')} to {dt.datetime.strftime(df['datetime'].max(), '%Y-%m-%d')}"
    title_x = 0.5
    plot_background_color = 'rgba(0,0,0,0)' #'rgba(0,0,0,0)' clearn
    
    subplot_titles = subplot_titles if show_subplot_titles else None

   

   

    figure_autosize = False #True/False
    y_axis_auto_margin  = True #True/False
    horizontal_subplot_spacing = 0.00

    
    font = "Arial"
    #fig_width = 1000
    #fig_height = 100
# Create subplots
    fig = make_subplots(rows=number_of_rows, cols=number_of_columns, subplot_titles=subplot_titles, specs=[[{"secondary_y": True}] * number_of_columns] * number_of_rows, horizontal_spacing = horizontal_subplot_spacing)
    fig.update_layout(title_x=title_x)
    fig.update_layout(plot_bgcolor=plot_background_color)
    fig.update_layout(autosize = figure_autosize)
    
    
    ### legend
    # x is horizontal, y is veertical   The 'y' property is a number and may be specified as: - An int or float in the interval [-2, 3]
    legend_orientation =  "h" #h or v
    legend_x = 0.2
    legend_y = -0.1
    show_legend = True #True/False
    fig.update_layout(legend=dict(orientation=legend_orientation, x=legend_x, y=legend_y, font=dict(
            size=14)), showlegend=show_legend)

   # margin_l = 0
   # margin_r = 0
   # margin_b = 0
    
    #fig.update_layout(margin=dict(l=margin_l, r=margin_r, t=title_font_size, b=margin_b))  # Adjust the margin as neededautosize=False,
    fig.update_layout(font=dict(size=font_size))  # Set the desired text size)

    fig.update_annotations(font_size=annotation_font_size) # subplot titles are hardcoded as an annotation
    
    show_chart_title = True
    if show_chart_title == True:
        fig.update_layout(title_text=f"{chart_title}", title_font=dict(size=title_font_size))
       

    # font 
     # font 
    fig.update_layout(font_family=font, title_font_family=font,) # updates font for whole figure
       
    row_count = 1
    
  
    fig.update_yaxes(range=[primary_min, primary_max], showticklabels=True, ticks="inside", showgrid=False, showline=True, linecolor='black', linewidth=2, title_text = primary_y_title, row=row_count, col=1, secondary_y=False, )
    # fig.update_yaxes(range=[primary_min, primary_max], showticklabels=True, ticks="inside", showgrid=False, showline=True, linecolor='black', linewidth=2, title_text=f"{derived_parameter.replace('_', ' ')} ({config[parameter]['unit']})", row=row_count, col=1, secondary_y=False, )
    # range=[primary_min, primary_max],
    # secondary y axis
    fig.update_yaxes(range=[primary_min, primary_max], showticklabels=True, ticks="inside", showgrid=False, showline=True, linecolor='black', linewidth=2, title_text = secondary_y_title, row=row_count, col=1, secondary_y=True)
    # range=[primary_min, primary_max], 
    tick_format = '%b-%d' # shows month and day (%y year without centuary %Y year with century) 
    # Check if the data spans more than one year
    min_year = df['datetime'].min().year
    max_year = df['datetime'].max().year

    # Set x-axis tick format based on the year range
    if min_year != max_year:
        # Show full date including year if data spans multiple years
        tick_format = '%Y-%m'  # '%y-%m-%d'  
        minor_ticks = pd.date_range(df['datetime'].min(), df['datetime'].max(), freq='5D')  # Minor ticks every 15 days 
    else:
        # Only show month and day if data is within a single year
        tick_format = '%m-%d'
        minor_ticks = pd.date_range(df['datetime'].min(), df['datetime'].max(), freq='1D')  # Minor ticks every 15 days   

        # Define major and minor tick locations
    #major_ticks = pd.date_range(df['datetime'].min(), df['datetime'].max(), freq='MS')  # Major ticks at start of each month
    #minor_ticks = pd.date_range(df['datetime'].min(), df['datetime'].max(), freq='15D')  # Minor ticks every 15 days    
    
    # right now comparison data has a slightly longer ranger then df so use this for the range
    if not df.empty and not comparison_data.empty:
        x_min = min(df["datetime"].min(), comparison_data["datetime"].min())
        x_max = max(df["datetime"].max(), comparison_data["datetime"].max())
    elif df.empty and not comparison_data.empty:
        x_min = comparison_data["datetime"].min()
        x_max = comparison_data["datetime"].max()
    else:   
        x_min = df["datetime"].min()
        x_max = df["datetime"].max()
    
    
    fig.update_xaxes(range=[x_min, x_max], showticklabels=True, ticks="inside", tickformat=tick_format, showgrid=False, showline=True, linecolor='black', linewidth=2, mirror = True)
    #fig.update_xaxes(range=[df['datetime'].min(),df['datetime'].max()], showticklabels=True, ticks="inside", tickformat=tick_format, showgrid=False, showline=True, linecolor='black', linewidth=2, mirror = True)
    
    
    # major ticks
    #fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray', major_ticks="outside",  tick0=major_ticks[0],  )# First minor tick
    # minor ticks
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray', minor_ticks="outside",  tick0=minor_ticks[0],  )# First minor tick

    if 'offset' not in df.columns:
            df[f'offset'] =df[f'corrected_data']-df[f'data']
     
    if ("q_observation" in df.columns or "discharge_observation" in df.columns):
            if "q_observation" in df.columns:
                obs = "q_observation"
            if "discharge_observation" in df.columns:
                obs = "discharge_observation" 
          
    if normalize_data == True:
        exclude_cols = ["estimate", "datetime", "comment"]

        for col in df.columns:
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
                d_min = df[col].min(skipna=True)
                d_max = df[col].max(skipna=True)
                df[col] = ((df[col] - d_min) / (d_max - d_min)).round(2)
    else:
        pass

    # comparison graph graph it first so it goes behind data
    if not comparison_data.empty:
        comparison_sites = comparison_data[["site", "parameter"]].drop_duplicates()
        for index, row in comparison_sites.iterrows():
            dfc = comparison_data.loc[(comparison_data["site"] == row["site"]) & (comparison_data["parameter"] == row["parameter"])].copy() 
            dfc.loc[dfc["corrected_data"] == -99, "corrected_data"] = np.nan # remove dry data
            if normalize_data == True:
                    d_min = dfc["corrected_data"].min(skipna=True)
                    d_max =  dfc["corrected_data"].max(skipna=True)
                    dfc["corrected_data"] = ((dfc["corrected_data"] - d_min) / (d_max - d_min)).round(2)
            if row["site"] == site_name and row["parameter"] == parameter: # graph existing data
               
                fig.add_trace(go.Scatter(
                    x=dfc.loc[:, "datetime"],
                    y=dfc.loc[:, f"corrected_data"],
                    line=dict(color=color_map.get(f'comparison', 'black'), width = 2),name=f"existing data",showlegend=True, hovertemplate='<b>%{meta}</b> <br><b>date:</b> %{x|%Y-%m-%d %H:%M}<br><b>value:</b> %{y}<extra></extra>',
                    meta=f"existing data",), row=row_count, col=1, secondary_y=derived_parameter_axis),
          
            else: # graph comparison data
                fig.add_trace(go.Scatter(
                    x=dfc.loc[:, "datetime"],
                    y=dfc.loc[:, f"corrected_data"],
                    line=dict(width = 2), name=f"comparison: {row}",showlegend=False, hovertemplate='<b>%{meta}</b> <br><b>date:</b> %{x|%Y-%m-%d %H:%M}<br><b>value:</b> %{y}<extra></extra>',
                    meta=f"comparison: {row}"),row=row_count, col=1, secondary_y=derived_parameter_axis),

            
    if f"data" in df.columns and not "discharge" in df.columns: # dont show raw data if discharge it gets too messy
            fig.add_trace(go.Scatter(
                x=df.loc[:, "datetime"],
                y=df.loc[:, f"data"],
                line=dict(color=color_map.get(f"data", 'black'), width = 1),
                name=f"raw {base_parameter.replace('_', ' ')}",showlegend=True, hovertemplate='<b>%{meta}</b> <br><b>date:</b> %{x|%Y-%m-%d %H:%M}<br><b>value:</b> %{y}<extra></extra>',
                meta=f"raw {base_parameter.replace('_', ' ')}",),row=row_count, col=1, secondary_y=data_axis),
    
    # graph corrected data          
    if f"corrected_data" in df.columns and parameter != "discharge": # if it is waterlevel only       
        fig.add_trace(go.Scatter(
            x=df.loc[:, "datetime"],
            y=df.loc[:, f"corrected_data"],
            line=dict(color=color_map.get(f"corrected_data", 'black'), width = 2),name=f"corrected {base_parameter.replace('_', ' ')}",showlegend=True, hovertemplate='<b>%{meta}</b> <br><b>date:</b> %{x|%Y-%m-%d %H:%M}<br><b>value:</b> %{y}<extra></extra>',
            meta=f"corrected {base_parameter.replace('_', ' ')}",), row=row_count, col=1, secondary_y=derived_parameter_axis),
    
        # graph dry data
        if not dry.empty:
            # find min
            dry_min = df["corrected_data"].min()
            if dry_min < 0+20: # in elevation not relative feet
                dry_min = 0
            dry.loc[dry['non_detect'] == "1", 'corrected_data'] = dry_min
            fig.add_trace(go.Scatter(
                    x=dry.loc[:, "datetime"],
                    y=dry.loc[:, "corrected_data"],
                    line=dict(color='#6B7B8C', width=2),name=f"dry / non-detect",showlegend=True, hovertemplate='<b>%{meta}</b> <br><b>date:</b> %{x|%Y-%m-%d %H:%M}<br><b>value:</b> %{y}<extra></extra>',
                    meta=f"dry / non-detect",), row=row_count, col=1, secondary_y=derived_parameter_axis),

    if f"corrected_data" in df.columns and parameter == "discharge": # if it is waterlevel only       
                fig.add_trace(go.Scatter(
                    x=df.loc[:, "datetime"],
                    y=df.loc[:, f"corrected_data"],
                    line=dict(color=color_map.get(f"corrected_data", 'black'), width = 2),name=f"corrected {base_parameter.replace('_', ' ')}",showlegend=True, hovertemplate='<b>%{meta}</b> <br><b>date:</b> %{x|%Y-%m-%d %H:%M}<br><b>value:</b> %{y}<extra></extra>',
                    meta=f"corrected {base_parameter.replace('_', ' ')}",), row=row_count, col=1, secondary_y=base_parameter_axis),
            
    # special graph -discharge
    if f"discharge" in df.columns:
        fig.update_yaxes(title_text="discharge cfs", row=row_count, col=1, showticklabels=True, secondary_y=derived_parameter_axis, )
                #fig.update_yaxes(showgrid=False, showticklabels=False, row=row_count, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(
            x=df.loc[:, "datetime"],
            y=df.loc[:, "discharge"],
            line=dict(color=color_map.get(f"discharge", 'black'), width = 2), name=f"discharge",showlegend=True, hovertemplate='<b>%{meta}</b> <br><b>date:</b> %{x|%Y-%m-%d %H:%M}<br><b>value:</b> %{y}<extra></extra>',
            meta=f"discharge",),row=row_count, col=1, secondary_y=derived_parameter_axis),
    
        if "percentile_05" in display_statistics:
            y_values = [statistics["percentile_05"]] * len(df["datetime"])
            fig.add_trace(go.Scatter(
                x=df.loc[:, "datetime"],
                y=y_values,
                line=dict(color='rgba(255, 100, 0, 0.6)', width=2, dash='dash'),
                name=f"precentile_05", showlegend=True,
                hovertemplate='<b>%{meta}</b> <br><b>date:</b> %{x|%Y-%m-%d %H:%M}<br><b>value:</b> %{y}<extra></extra>', 
                meta=f"precentile_05",), row=row_count, col=1, secondary_y=derived_parameter_axis)

        if "percentile_05_q" in display_statistics:
            y_values = [statistics["percentile_05_q"]] * len(df["datetime"])
            fig.add_trace(go.Scatter(
                x=df.loc[:, "datetime"],
                y = y_values,
                 # Light orange (using RGBA for transparenc
                line=dict(color='rgba(255,165,0,0.6)', dash='dash'),
                name=f"precentile_05_q", showlegend=True, hovertemplate='<b>%{meta}</b> <br><b>date:</b> %{x|%Y-%m-%d %H:%M}<br><b>value:</b> %{y}<extra></extra>',
                meta=f"precentile_05_q",), row=row_count, col=1, secondary_y=derived_parameter_axis)

    # dry indicator
    #if f"dry_indicator" in df.columns and data_axis != "none":
    #    
    #    df.loc[df['dry_indicator'] == "dry indicator", "dry_indicator"] = df['data'].min()
    #    df.loc[df['dry_indicator'] == " ", "dry_indicator"] =  np.nan # graph dry data is data min for visualization
    #    fig.add_trace(go.Scatter(
    #            x=df.loc[:, "datetime"], # 1 is graph if dry 0 is graph if not dry
    #            y=df.loc[:, f"dry_indicator"],
    #            line=dict(color=color_map.get(f"dry_indicator", 'black'), width = 2), name=f"dry indicator",showlegend=True, hovertemplate='<b>%{meta}</b> <br><b>date:</b> %{x|%Y-%m-%d %H:%M}<br><b>value:</b> %{y}<extra></extra>',
    #            meta=f"dry indicator",), row=row_count, col=1, secondary_y=data_axis),
        
    # average
    if f"mean" in df.columns and data_axis != "none":
        average_axis = base_parameter_axis
        fig.add_trace(go.Scatter(
                x=df.loc[:, "datetime"],
                y=df.loc[:, f"mean"],
                line=dict(color=color_map.get(f"mean", 'black'), width = 2),
                name=f"mean",showlegend=True, hovertemplate='<b>%{meta}</b> <br><b>date:</b> %{x|%Y-%m-%d %H:%M}<br><b>value:</b> %{y}<extra></extra>',
                meta=f"mean",), row=row_count, col=1, secondary_y=average_axis),   
     # average
    if f"interpolated_data" in df.columns:
        estimate_axis = base_parameter_axis       
        fig.add_trace(go.Scatter(
            x=df.loc[:, "datetime"],
            y=df.loc[:, "interpolated_data"],
            line=dict( width = 2),
            name=f"interpolated data",showlegend=True, hovertemplate='<b>%{meta}</b> <br><b>date:</b> %{x|%Y-%m-%d %H:%M}<br><b>value:</b> %{y}<extra></extra>',
            meta=f"interpolated data",), row=row_count, col=1, secondary_y=estimate_axis),
    
    # display field observation points    
    #if "field_observations" in df.columns or "observations" in df.columns or "observation" in df.columns or "observation_stage" in df.columns:
    if any(col in df.columns for col in ["field_observations", "observations", "observation", "observation_stage", "parameter_observation"]):
        if "field_observations" in df.columns:
            obs = "field_observations"
        if "observations" in df.columns:
            obs = "observations"
        if "observation" in df.columns:
            obs = "observation"
        if "observation_stage" in df.columns:
            obs = "observation_stage"
        if "parameter_observation" in df.columns:
            obs = "parameter_observation"
        
        ### plot field observations    
        if "corrected_data" in df.columns and parameter != "discharge":
            fig.add_trace(go.Scatter(
                x=df.loc[df[f'{obs}'].notna(), f'datetime'],
                y=df.loc[df[f'{obs}'].notna(), f"{obs}"],
                mode='markers', marker=dict(color=color_map.get(f"field_observation", 'black'), size=12, opacity=.9), 
                name=f"observation", showlegend=True, hovertemplate='<b>%{meta}</b> <br><b>date:</b> %{x|%Y-%m-%d %H:%M}<br><b>value:</b> %{y}<br><b>offset:</b> %{customdata}<extra></extra>',
                meta=f"observation", customdata=round(df.loc[df[f'{obs}'].notna(), f'offset'], 2).to_numpy(),),
                row=row_count, col=1, secondary_y=derived_parameter_axis)
        if "corrected_data" in df.columns and parameter == "discharge":
            fig.add_trace(go.Scatter(
                x=df.loc[df[f'{obs}'].notna(), f'datetime'],
                #y=df.loc[df[f'{obs}'].notna(), f'corrected_data'],
                y=df.loc[df[f'{obs}'].notna(), f"{obs}"],
                mode='markers', 
                marker=dict(color=color_map.get(f"field_observation", 'black'), size=12, opacity=.9), 
                name=f"observation",
                showlegend=True, 
                hovertemplate='<b>%{meta}</b> <br><b>date:</b> %{x|%Y-%m-%d %H:%M}<br><b>value:</b> %{y}<br><b>offset:</b> %{customdata}<extra></extra>',
                meta=f"observation",
                customdata=round(df.loc[df[f'{obs}'].notna(), f'offset'], 2).to_numpy(),  # Pass the offset data as customdata
                ),
                row=row_count, col=1, secondary_y=base_parameter_axis)
        
        row_count = 1
        annotation_x = 0.00 #0.05 # allows offset for when year is displatyed on axis (left and right 0 is left justified 0.5 is in the middle)
        annotation_y = -.07 # up and down (started with .9)
       

        df_min = df.loc[df['datetime'] == df['datetime'].min()]
        df_max = df.loc[df['datetime'] == df['datetime'].max()]
        annotation_axis = base_parameter_axis
        if f"data" in df.columns and "corrected_data" in df.columns:
            if not df.empty:
                df_clean = df.dropna(subset=['corrected_data'])  # Drop rows where "corrected_data" is NaN
                if not df_clean.empty:
            #df_clean = df.dropna(subset=['corrected_data'])  # Drop rows where "corrected_data" is NaN
                    df_min = df_clean.loc[df_clean['datetime'] == df_clean['datetime'].min()]
                    df_min = df_clean.loc[df_clean['datetime'] == df_clean['datetime'].min()]
                    
            fig.add_annotation(
                text=(f"<span style='line-height:0.9;'>"
                    f"{df_min['datetime'].iloc[0].strftime('%Y-%m-%d %H:%M')}<br>"
                    f"obs: {df_min['corrected_data'].iloc[0]} | inst: {round(df_min['data'].iloc[0], 2)}<br>"
                    f"offset: {round((df_min['corrected_data'].iloc[0] - df_min['data'].iloc[0]), 2)}"
                    f"</span>"),
                xref="x domain", yref="y domain", x=annotation_x, y=annotation_y-.07,
                showarrow=False, row=row_count, col=1, secondary_y=annotation_axis,
                font=dict(size=14))
          
            fig.add_annotation(
                text=(f"<span style='line-height:0.7;'>"
                    f"{df_max['datetime'].iloc[0].strftime('%Y-%m-%d %H:%M')}<br>"
                    f"obs: {df_max['corrected_data'].iloc[0]} | inst: {round(df_max['data'].iloc[0], 2)}<br>"
                    f"offset: {round((df_max['corrected_data'].iloc[-1] - df_max['data'].iloc[0]), 2)}"
                    f"</span>"),
                xref="x domain", yref="y domain", x=annotation_x+1, y=annotation_y-.07,
                showarrow=False, row=row_count, col=1, secondary_y=annotation_axis,
                font=dict(size=14))
              
            fig.add_annotation(text=f"<span style='line-height:0.9;'>session shift: {round(((df_max['corrected_data'].iloc[0] - df_max['data'].iloc[0]) - (df_min['corrected_data'].iloc[0] - df_min['data'].iloc[0])),2)}",
                            xref="x domain", yref="y domain",
                            x=.5, y=annotation_y-.02, showarrow=False, row=row_count, col=1, secondary_y=annotation_axis,)
    
    # discharge observation
    if ("q_observation" in df.columns or "discharge_observation" in df.columns):
            if "q_observation" in df.columns:
                obs = "q_observation"
            if "discharge_observation" in df.columns:
                obs = "discharge_observation"
            # graph field observation
            if "discharge" in df.columns:
                fig.add_trace(go.Scatter(
                x=df.loc[df[f'{obs}'].notna(),f'datetime'],
                y=df.loc[df[f'{obs}'].notna(),f'{obs}'],
                mode='markers', marker=dict(color=color_map.get(f"q_observation", 'black'), size=12, opacity=.9), 
                name=f"{derived_parameter} observation",showlegend=True, hovertemplate='<b>%{meta}</b> <br><b>date:</b> %{x|%Y-%m-%d %H:%M}<br><b>value:</b> %{y}<extra></extra>',meta=f"{obs}",),
                row=row_count, col=1, secondary_y = derived_parameter_axis)
           
            row_count = 1
            annotation_x = 0.00 # allows offset for when year is displatyed on axis (obs is .05)
            annotation_y = 1 # up and down? (obs is -.07)
            
            if "rating_number" in df.columns:
                    fig.add_annotation(text=f"rating: {df_min['rating_number'].iloc[0]}",
                        xref="x domain", yref="y domain",
                        x=annotation_x, y=annotation_y, showarrow=False, row=row_count, col=1, secondary_y=False,)
                    
            if f"discharge" in df.columns:
                    fig.add_annotation(text=f"obs: {df_min['discharge'].iloc[0]}",
                        xref="x domain", yref="y domain",
                        x=annotation_x, y=annotation_y-.04, showarrow=False, row=row_count, col=1, secondary_y=False,)
                
            if "q_offset" in df.columns:
                    fig.add_annotation(text=f"offset: {df_min[f'q_offset'].iloc[0]} ({df_min[f'precent_q_change'].iloc[0]}%)",
                        xref="x domain", yref="y domain",
                        x=annotation_x, y=annotation_y-.08, showarrow=False, row=row_count, col=1, secondary_y=False,)
              
            """fig.add_annotation(text=f"offset: {round(obs_df[f'offset'].iloc[0], 2)}",
                        xref="x domain", yref="y domain",
                        x=annotation_x, y=annotation_y-.075, showarrow=False, row=row_count, col=1, secondary_y=False,)"""
                
           # if p_obs_df.shape[0] > 1: #  if there are more then one observation graph first and last
            if "rating_number" in df.columns:
                        fig.add_annotation(text=f"rating: {df_max['rating_number'].iloc[0]}",
                            xref="x domain", yref="y domain",
                        x=annotation_x+1, y=annotation_y, showarrow=False, row=row_count, col=1, secondary_y=False,)

            if f"discharge" in df.columns:
                        fig.add_annotation(text=f"obs: {df_max['discharge'].iloc[0]}",
                            xref="x domain", yref="y domain",
                            x=annotation_x+1, y=annotation_y-.04, showarrow=False, row=row_count, col=1, secondary_y=False,)
                    
            if "q_offset" in df.columns:
                        fig.add_annotation(text=f"offset: {df_max[f'q_offset'].iloc[0]} ({df_max[f'precent_q_change'].iloc[0]}%)",
                            xref="x domain", yref="y domain",
                            x=annotation_x+1, y=annotation_y-.08, showarrow=False, row=row_count, col=1, secondary_y=False,)
                    
            """fig.add_annotation(text=f"offset: {round(obs_df[f'offset'].iloc[-1], 2)}",
                            xref="x domain", yref="y domain",
                            x=annotation_x+.95, y=annotation_y-.075, showarrow=False, row=row_count, col=1, secondary_y=False,)"""
                        
                        # shift
                    #if observation_axis != "none":
            try: # if offset is nan it wont graph
                        fig.add_annotation(text=f"rating shift: {round((df_max[f'q_offset'].iloc[0] - df_min[f'q_offset'].iloc[0]),2)}",
                                    xref="x domain", yref="y domain",
                                    x=annotation_x+.5, y=annotation_y, showarrow=False, row=row_count, col=1, secondary_y=False,)
            except:
                pass
    row_count = row_count+1
    
    return fig

def cache_graph_export(df, site_selector_value, site_code, site_name, parameter, comparison_data, primary_min, primary_max, secondary_min, secondary_max, normalize_data, statistics, display_statistics):
    fig = parameter_graph(df, site_selector_value, site_code, site_name, parameter, comparison_data, primary_min, primary_max, secondary_min, secondary_max, normalize_data, statistics, display_statistics)
   
    #fig.update_layout(autosize=False) # larer fig height will go off page
    fig.update_layout(autosize=True, height = 800, width = 1800) # larer fig height will go off page
   
    #return html.Div(dcc.Graph(
    #    figure=fig, id = "graph",
    #    config={'responsive': True},)  # Make the graph responsive
    return fig

#def save_fig(df, site_code, site_name, parameter, comparison_site, comparison_parameter, data_axis, corrected_data_axis, derived_data_axis, observation_axis, comparison_axis):

def plot_for_save(df, site_selector_value, site_sql_id, site, parameter, comparison_data, primary_min, primary_max, secondary_min, secondary_max, normalize_data, statistics, display_statistics):
   
    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    matplotlib.use("Agg")  # Use non-GUI backend suitable for scripts/servers

    def create_hydrograph(df, site, parameter, statistics):
        """
        Create a hydrograph plot with dual y-axes for hydrological data.
        
        Parameters:
        -----------
        df : DataFrame
            Input data with columns: datetime, corrected_data, data, etc.
        site : str
            Site name
        parameter : str
            Parameter type (water_temperature, discharge, etc.)
        statistics : dict
            Statistics dict with min/max values
        """
        statistics = pd.read_json(statistics, orient="split")
        dfp = df.copy()
        end_date = dfp["datetime"].max().date()
        
        # Initialize figure with landscape dimensions
        fig, ax1 = plt.subplots(figsize=(11, 8.5))
        ax2 = ax1.twinx()
        
        # Prepare dry/non-detect data
        dry = dfp.copy()
        dry.loc[dry['non_detect'] != "1", 'corrected_data'] = np.nan
        dfp.loc[dfp['non_detect'] == "1", 'corrected_data'] = np.nan
        
        # Plot based on parameter type
        if parameter == "discharge":
            _plot_discharge_axes(ax1, ax2, dfp, dry)
        else:
            _plot_non_discharge_axes(ax1, ax2, dfp, dry, parameter)
        
        # Add observation points
        _add_observations(ax1, ax2, dfp, parameter)
        
        # Configure axes
        _configure_date_axis(ax1, dfp)
        _set_axis_limits(ax1, ax2, dfp, statistics, parameter)
        
        # Add title and styling
        plt.title(f"{site.replace('_', ' ')} {parameter.replace('_', ' ')} "
                f"{dfp['datetime'].min().strftime('%Y-%m-%d')} to "
                f"{dfp['datetime'].max().strftime('%Y-%m-%d')}")
        
        _style_plot(ax1, ax2)
        
        # Add legend
        _add_legend(ax1, ax2)
        
        # Add metadata text
        _add_metadata_text(fig, ax1, dfp, parameter)
        
        # Save and export
        export_path = (f"W:/STS/hydro/GAUGE/Temp/Ian's Temp/"
                    f"{site}_{parameter.replace('_', ' ')}_"
                    f"{dfp['datetime'].min().strftime('%Y-%m-%d')} to "
                    f"{dfp['datetime'].max().strftime('%Y-%m-%d')}.pdf")
        plt.savefig(export_path, dpi=300, format='pdf')
        plt.close()
        print(f"Graph saved to {export_path}")
        
        return export_path


    def _plot_discharge_axes(ax1, ax2, dfp, dry):
        """Plot discharge parameter - stage on ax1, discharge on ax2."""
        # ax1: Plot stage (corrected_data)
        ax1.set_ylabel('water level feet')
        
        if 'corrected_data' in dfp.columns:
            ax1.plot(dfp['datetime'], dfp['corrected_data'], 
                    label='stage', color='blue', linewidth=2)
            
            # Add dry/non-detect indicator
            if not dry.empty:
                dry_min = dfp["corrected_data"].min()
                if dry_min < 20:
                    dry_min = 0
                dry.loc[dry["non_detect"] == "1", "corrected_data"] = dry_min
                ax1.plot(dry["datetime"], dry["corrected_data"], 
                        label="dry / non-detect", color="#567494", linewidth=2)
        
        # ax2: Plot discharge
        if 'discharge' in dfp.columns:
            ax2.plot(dfp['datetime'], dfp['discharge'], 
                    label='discharge', color='lightseagreen', linewidth=2)
            ax2.set_ylabel('discharge (CFS)')


    def _plot_non_discharge_axes(ax1, ax2, dfp, dry, parameter):
        """Plot non-discharge parameters - corrected data on ax1, raw data on ax2."""
        # ax1: Plot corrected data
        if parameter == "water_temperature":
            ax1.set_ylabel('water temperature (deg. C)')
            color = 'firebrick'
            label = parameter
        else:
            ax1.set_ylabel('water level feet')
            color = 'blue'
            label = 'corrected data'
        
        if 'corrected_data' in dfp.columns:
            ax1.plot(dfp['datetime'], dfp['corrected_data'], 
                    label=label, color=color, linewidth=2)
            
            # Add dry/non-detect indicator for non-temperature parameters
            if parameter != "water_temperature" and not dry.empty:
                dry_min = dfp["corrected_data"].min()
                if dry_min < 20:
                    dry_min = 0
                dry.loc[dry["non_detect"] == "1", "corrected_data"] = dry_min
                ax1.plot(dry["datetime"], dry["corrected_data"], 
                        label="dry / non-detect", color="#567494", linewidth=2)
        
        # ax2: Plot raw data
        if 'data' in dfp.columns:
            ax2.plot(dfp['datetime'], dfp['data'], 
                    label='raw data', color='darkgrey', linewidth=2)
            ax2.set_ylabel('raw data')

        # non discharge ax1 = corrected_data, ax2 = data
    def _add_observations(ax1, ax2, dfp, parameter):
        """Add observation scatter points."""
        if 'observation_stage' in dfp.columns:
            ax1.scatter(dfp['datetime'], dfp['observation_stage'], 
                    label='observation stage', color="#444546", s=30)
        
        if 'parameter_observation' in dfp.columns:
            ax1.scatter(dfp['datetime'], dfp['parameter_observation'], 
                    label=f"{parameter} observation", color="#444546", s=30)
        
        if 'q_observation' in dfp.columns:
            ax2.scatter(dfp['datetime'], dfp['q_observation'], 
                    label='discharge observation', color="#444546", s=30)


    def _configure_date_axis(ax1, dfp):
        """Configure x-axis date formatting based on time span."""
        time_span = dfp['datetime'].max() - dfp['datetime'].min()
        
        if time_span.days < 30:  # Less than a month
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, time_span.days // 5)))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        elif time_span.days <= 365:  # Less than a year
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        else:  # More than a year
            ax1.xaxis.set_major_locator(mdates.YearLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


    def _set_axis_limits(ax1, ax2, dfp, statistics, parameter):
        """Set y-axis limits for both axes."""
        min_corrected = statistics.get('min_corrected_data', dfp["corrected_data"].min())
        max_corrected = statistics.get('max_corrected_data', dfp["corrected_data"].max())
        
        # Handle invalid values
        if min_corrected == -99 or pd.isna(min_corrected):
            min_corrected = dfp["corrected_data"].min()
        if max_corrected == -99 or pd.isna(max_corrected):
            max_corrected = dfp["corrected_data"].max() + 1
        
        # Set minimum to 0 if less than 20
        if min_corrected < 20:
            min_corrected = 0
        corrected_diff = abs(min_corrected-max_corrected)
        
        ax1.set_ylim([min_corrected, max_corrected])
        print(f"min corrected data: {min_corrected}, max corrected data: {max_corrected}")
        
        # Set secondary axis limits based on parameter
        if parameter == "discharge" and 'discharge' in dfp.columns:
            # For discharge plots: ax2 shows discharge
            min_derived = statistics.get('min_derived_parameter', 0)
            max_derived = statistics.get('max_derived_parameter', dfp["discharge"].max())
            
            if pd.isna(min_derived) or min_derived == -99:
                min_derived = 0
            if pd.isna(max_derived) or max_derived == -99:
                max_derived = dfp["discharge"].max()
            
            ax2.set_ylim([min_derived, max_derived + 1])
        else:
            # For non-discharge plots: ax2 shows raw data
            #diff = abs(dfp["corrected_data"].max() - dfp["corrected_data"].min())
            mean = dfp['data'].mean()
            ax2.set_ylim([dfp['data'].mean()-(corrected_diff/2), dfp['data'].mean()+(corrected_diff/2)])
            #ax2.set_ylim([dfp['data'].min(), dfp['data'].min() + corrected_diff])


    def _style_plot(ax1, ax2):
        """Apply styling to plot."""
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        
        ax1.grid(False)
        ax2.grid(False)


    def _add_legend(ax1, ax2):
        """Add combined legend for both axes."""
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.legend(handles1 + handles2, labels1 + labels2, 
                loc='upper center', ncol=len(labels1 + labels2), 
                bbox_to_anchor=(0.5, -0.03))


    def _add_metadata_text(fig, ax1, dfp, parameter):
        """Add metadata text to figure."""
        first_idx = dfp['datetime'].idxmin()
        last_idx = dfp['datetime'].idxmax()
        
        first_corrected = dfp.loc[first_idx, 'corrected_data']
        first_data = dfp.loc[first_idx, 'data']
        last_corrected = dfp.loc[last_idx, 'corrected_data']
        last_data = dfp.loc[last_idx, 'data']
        
        first_offset = round(first_corrected - first_data, 2)
        last_offset = round(last_corrected - last_data, 2)
        change = round(last_offset - first_offset, 2)
        
        # Left side - first log
        fig.text(0.01, 0.07, f"first log: {dfp['datetime'].min().strftime('%Y-%m-%d %H:%M')}", fontsize=9)
        fig.text(0.01, 0.04, f"value {first_corrected} | instrument: {first_data}", fontsize=9)
        fig.text(0.01, 0.01, f"offset {first_offset}", fontsize=9)
        
        # Right side - last log
        fig.text(0.80, 0.07, f"last log: {dfp['datetime'].max().strftime('%Y-%m-%d %H:%M')}", fontsize=9)
        fig.text(0.80, 0.04, f"value: {last_corrected} | instrument: {last_data}", fontsize=9)
        fig.text(0.80, 0.01, f"offset: {last_offset}", fontsize=9)
        
        # Center - change
        fig.text(0.5, 0.01, f"change: {change}", fontsize=9)
        
        # Add discharge-specific metadata
        if parameter == "discharge" and "rating_number" in dfp.columns:
            q_obs_df = dfp.dropna(subset=["q_observation"]).copy()
            if not q_obs_df.empty:
                _add_discharge_metadata(fig, ax1, q_obs_df)


    def _add_discharge_metadata(fig, ax1, q_obs_df):
        """Add discharge-specific metadata to top of plot."""
        first_idx = q_obs_df['datetime'].idxmin()
        last_idx = q_obs_df['datetime'].idxmax()
        
        # Left side
        
        fig.text(0.01, 0.99, f"rating {q_obs_df.loc[first_idx, 'rating_number']}", 
                transform=ax1.transAxes, fontsize=9, verticalalignment='top')
        fig.text(0.01, 0.96, f"obs {q_obs_df.loc[first_idx, 'q_observation']}", 
                transform=ax1.transAxes, fontsize=9, verticalalignment='top')
        fig.text(0.01, 0.93, f"obs {q_obs_df.loc[first_idx, 'q_offset']} "
                f"({q_obs_df.loc[first_idx, 'precent_q_change']} %)", 
                transform=ax1.transAxes, fontsize=9, verticalalignment='top')
        
        # Right side
        fig.text(0.85, 0.99, f"rating {q_obs_df.loc[last_idx, 'rating_number']}", 
                transform=ax1.transAxes, fontsize=9, verticalalignment='top')
        fig.text(0.85, 0.96, f"obs {q_obs_df.loc[last_idx, 'q_observation']}", 
                transform=ax1.transAxes, fontsize=9, verticalalignment='top')
        fig.text(0.85, 0.93, f"obs {q_obs_df.loc[last_idx, 'q_offset']} "
                f"({q_obs_df.loc[last_idx, 'precent_q_change']} %)", 
                transform=ax1.transAxes, fontsize=9, verticalalignment='top')
    create_hydrograph(df, site, parameter, statistics)