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
    
    if parameter == "FlowLevel" or parameter == "discharge":
        base_parameter = "water_level"
        derived_parameter = "discharge"
        data_axis = False
        base_parameter_axis = False
        derived_parameter_axis = True
        comparison_axis = True

    elif parameter == "LakeLevel" or parameter == "Piezometer" or parameter == "water_level":
        base_parameter = "water_level"
        derived_parameter = "water_level"

    else:
        base_parameter = parameter
        derived_parameter = parameter
  
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
    #fig.update_yaxes(title_font_family=font, secondary = False)
    #fig.update_yaxes(title_font_family=font, Secondary = True)
       
    row_count = 1

       
        ### Water temperature
        #for i in df.index.unique():
       
        #fig.update_yaxes(range=[0,1], row=row_count, col=1, secondary_y=True)
    # primary y axis
    
    fig.update_yaxes(range=[primary_min, primary_max], showticklabels=True, ticks="inside", showgrid=False, showline=True, linecolor='black', linewidth=2, title_text=f"{derived_parameter.replace('_', ' ')} ({config[parameter]['unit']})", row=row_count, col=1, secondary_y=False, )
    # range=[primary_min, primary_max],
    # secondary y axis
    fig.update_yaxes(range=[primary_min, primary_max], showticklabels=True, ticks="inside", showgrid=False, showline=True, linecolor='black', linewidth=2, row=row_count, col=1, secondary_y=True)
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
   
    # Define major and minor tick locations


    #if normalize_data == False:
    # get observation df before normalization
   # if any(col in df.columns for col in ["field_observations", "observations", "observation", "observation_stage"]):
   #     if "field_observations" in df.columns:
   #         obs = "field_observations"
   #     if "observations" in df.columns:
   #         obs = "observations"
   #     if "observation" in df.columns:
   ##         obs = "observation"
    #    if "observation_stage" in df.columns:
   #         obs = "observation_stage"
   #     if "parameter_observation" in df.columns:
   #         obs = "parameter_observation"#

    if 'offset' not in df.columns:
            df[f'offset'] =df[f'corrected_data']-df[f'data']
     
       

    if ("q_observation" in df.columns or "discharge_observation" in df.columns):
            if "q_observation" in df.columns:
                obs = "q_observation"
            if "discharge_observation" in df.columns:
                obs = "discharge_observation"
            # graph field observation
          
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
                    
    if f"corrected_data" in df.columns and parameter != "discharge": # if it is waterlevel only       
                fig.add_trace(go.Scatter(
                    x=df.loc[:, "datetime"],
                    y=df.loc[:, f"corrected_data"],
                    line=dict(color=color_map.get(f"corrected_data", 'black'), width = 2),name=f"corrected {base_parameter.replace('_', ' ')}",showlegend=True, hovertemplate='<b>%{meta}</b> <br><b>date:</b> %{x|%Y-%m-%d %H:%M}<br><b>value:</b> %{y}<extra></extra>',
                    meta=f"corrected {base_parameter.replace('_', ' ')}",), row=row_count, col=1, secondary_y=derived_parameter_axis),
    
    if f"corrected_data" in df.columns and parameter == "discharge": # if it is waterlevel only       
                fig.add_trace(go.Scatter(
                    x=df.loc[:, "datetime"],
                    y=df.loc[:, f"corrected_data"],
                    line=dict(color=color_map.get(f"corrected_data", 'black'), width = 2),name=f"corrected {base_parameter.replace('_', ' ')}",showlegend=True, hovertemplate='<b>%{meta}</b> <br><b>date:</b> %{x|%Y-%m-%d %H:%M}<br><b>value:</b> %{y}<extra></extra>',
                    meta=f"corrected {base_parameter.replace('_', ' ')}",), row=row_count, col=1, secondary_y=base_parameter_axis),
            
    # special graph
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
    if f"dry_indicator" in df.columns and data_axis != "none":
        
        df.loc[df['dry_indicator'] == "dry indicator", "dry_indicator"] = df['data'].min()
        df.loc[df['dry_indicator'] == " ", "dry_indicator"] =  np.nan # graph dry data is data min for visualization
        fig.add_trace(go.Scatter(
                x=df.loc[:, "datetime"], # 1 is graph if dry 0 is graph if not dry
                y=df.loc[:, f"dry_indicator"],
                line=dict(color=color_map.get(f"dry_indicator", 'black'), width = 2), name=f"dry indicator",showlegend=True, hovertemplate='<b>%{meta}</b> <br><b>date:</b> %{x|%Y-%m-%d %H:%M}<br><b>value:</b> %{y}<extra></extra>',
                meta=f"dry indicator",), row=row_count, col=1, secondary_y=data_axis),
        
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
                y=df.loc[df[f'{obs}'].notna(),f'{derived_parameter}'],
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
def save_fig(df, site_selector_value, site_sql_id, site, parameter, comparison_data, primary_min, primary_max, secondary_min, secondary_max, normalize_data, statistics, display_statistics):
   
    end_date = df["datetime"].max().date()
   
    file_path = r"W:\STS\hydro\GAUGE\Temp\Ian's Temp\{0}_{1}_{2}.pdf".format(site, parameter, end_date)
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    matplotlib.use("Agg")  # Use non-GUI backend suitable for scripts/servers
    print("export")
   
    


    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Create secondary y-axis
    ax2 = ax1.twinx()

    # Plot 'corrected_data' and 'observation_stage' on the primary y-axis
    # Plot 'data' on the secondary y-axis
    # render order, first to grap is on the bottom

    # colors https://matplotlib.org/stable/gallery/color/named_colors.html
    if parameter == "water_temperature":
        ax1.set_ylabel('water temperature (deg. C)')
        if 'corrected_data' in df.columns:
            ax1.plot(df['datetime'], df['corrected_data'], label='corrected data', color='firebrick', linewidth=2)

    else:
        ax1.set_ylabel('water level feet')  
        if 'corrected_data' in df.columns:
            ax1.plot(df['datetime'], df['corrected_data'], label='corrected data', color='blue', linewidth=2)

    if 'observation_stage' in df.columns:
        ax1.scatter(df['datetime'], df['observation_stage'], label='observation stage', color='lightgrey', s=30)
    if 'parameter_observation' in df.columns:
        ax1.scatter(df['datetime'], df['parameter_observation'], label=f"""{parameter} observation""", color='lightgrey', s=30)
    #  secondary axis, if not discharge plot data
    if 'data' in df.columns and parameter != "discharge":
        ax2.plot(df['datetime'], df['data'], label='raw data', color='lightgrey', linewidth=1) 
    if 'discharge' in df.columns: 
        ax2.plot(df['datetime'], df['discharge'], label='raw data', color='lightseagreen', linewidth=2) 
        ax2.scatter(df['datetime'], df['q_observation'], label=f"""{parameter} observation""", color='lightgrey', s=30)
         # Label secondary axis
        ax2.set_ylabel('discharge (CFS)')
        
    # Date formatting for x-axis
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

 
    ax1.set_xlabel('date')


    # Title
    plt.title(f"{site.replace('_', ' ')} {parameter.replace('_', ' ')} {df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}")

    # Border around entire plot
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    # Hide grid lines
    ax1.grid(False)
    ax2.grid(False)

    # Legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if parameter == "discharge":
        plt.legend(handles1 + handles2, labels1 + labels2, loc='lower left')
    else:
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

  
    # Save the figure
    plt.tight_layout()

   

    # Add text below x-axis
    #plt.text(0.01, 0.01,  f"Start: {start_str}", ha='left', va='top', fontsize=9)
    # plt.text(0.99, -0.05, f"End: {end_str}", ha='right', va='top', fontsize=9)
    # stage 
    fig.text(0.01, 0.07,  f" ", fontsize=9)
    fig.text(0.01, 0.05,  f"first log: {df['datetime'].min().strftime('%Y-%m-%d %H:%M')}", fontsize=9)
    fig.text(0.01, 0.03,  f"value {df.loc[df['datetime'] == df['datetime'].min(), 'corrected_data'].dropna().iloc[0]} | instrument: {df.loc[df['datetime'] == df['datetime'].min(), 'data'].dropna().iloc[0]}", fontsize=9)
    fig.text(0.01, 0.01,  f"offset {(df.loc[df['datetime'] == df['datetime'].min(), 'corrected_data'].dropna().iloc[0] - df.loc[df['datetime'] == df['datetime'].min(), 'data'].dropna().iloc[0]).round(2)}", fontsize=9)
    fig.text(.80, 0.07, f""" """, fontsize=9)
    fig.text(.80, 0.05, f"""last log: {df["datetime"].max().strftime('%Y-%m-%d %H:%M')}""", fontsize=9)
    fig.text(0.80, 0.03,  f"value: {df.loc[df['datetime'] == df['datetime'].max(), 'corrected_data'].dropna().iloc[0]} | instrument: {df.loc[df['datetime'] == df['datetime'].max(), 'data'].dropna().iloc[0]}", fontsize=9)
    fig.text(0.80, 0.01,  f"offset: {(df.loc[df['datetime'] == df['datetime'].max(), 'corrected_data'].dropna().iloc[0] - df.loc[df['datetime'] == df['datetime'].max(), 'data'].dropna().iloc[0]).round(2)}", fontsize=9)

    fig.text(0.50, 0.01,  f"changet: {((df.loc[df['datetime'] == df['datetime'].min(), 'corrected_data'].dropna().iloc[0] - df.loc[df['datetime'] == df['datetime'].min(), 'data'].dropna().iloc[0]).round(2) - (df.loc[df['datetime'] == df['datetime'].max(), 'corrected_data'].dropna().iloc[0] - df.loc[df['datetime'] == df['datetime'].max(), 'data'].dropna().iloc[0]).round(2)).round(2)}", fontsize=9)
    
    if parameter == "discharge":
        if "rating_number" in df.columns:
            q_obs_df = df.dropna(subset = ["q_observation"]).copy()
            # discharge plot at top of graph
            fig.text(0.01, 0.99, f"""rating {q_obs_df.loc[q_obs_df['datetime'] == q_obs_df['datetime'].min(), 'rating_number'].iloc[0]}""", transform=ax1.transAxes, fontsize=9, verticalalignment='top',),
            fig.text(0.01, 0.96, f"""obs {q_obs_df.loc[q_obs_df['datetime'] == q_obs_df['datetime'].min(), 'q_observation'].iloc[0]}""", transform=ax1.transAxes, fontsize=9, verticalalignment='top',),
            
            fig.text(0.80, 0.99, f"""rating {q_obs_df.loc[q_obs_df['datetime'] == q_obs_df['datetime'].max(), 'rating_number'].iloc[0]}""", transform=ax1.transAxes, fontsize=9, verticalalignment='top',),
            fig.text(0.80, 0.96, f"""obs {q_obs_df.loc[q_obs_df['datetime'] == q_obs_df['datetime'].max(), 'q_observation'].iloc[0]}""", transform=ax1.transAxes, fontsize=9, verticalalignment='top',),
        
        #fig.text(0.80, -.1,  f"""rating {df.loc[df['datetime'] == df['datetime'].max(), 'rating_number'].dropna().iloc[0]}""")
    
    export_path = f"W:/STS/hydro/GAUGE/Temp/Ian's Temp/{site}_{parameter.replace('_', ' ')}_{df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}.pdf"
    plt.savefig(export_path, dpi=300, format='pdf')
    plt.close()
    print(f"Graph saved to {export_path}")
    # Use plotly.io.write_image to export the figure as a PDF
    #pio.write_image(fig, file_path, format='pdf')
    #fig.write_image(file_path, format="pdf", engine="kaleido")

def format_cache_data(df_raw, parameter):
    '''takes a raw df from cache, and does some pre-processing and adds settings'''
    '''returns df to cache, which sends df back to this program'''
    '''as this program is used in multiple parts of cache and is still in dev,
        this is a good workaround from having to copy and paste the dev code'''
    end_time = df_raw.tail(1)
    end_time['datetime'] = pd.to_datetime(
    end_time['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce', infer_datetime_format=True)
    end_time['datetime'] = end_time['datetime'].map(
        lambda x: dt.datetime.strftime(x, '%Y_%m_%d'))
    end_time = end_time.iloc[0, 0]

    df_raw['datetime'] = pd.to_datetime(
        df_raw['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce', infer_datetime_format=True)
    df_raw['datetime'] = df_raw['datetime'].map(
        lambda x: dt.datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))


    if parameter == "water_level" or parameter == "LakeLevel":
        observation = "observation_stage"
       # df_raw = df_raw[["datetime", "data", "corrected_data"]]
    elif parameter == "groundwater_level" or parameter == "Piezometer":
        observation = "observation_stage"
    elif parameter == 'water_temperature':
        observation = "parameter_observation"
    elif parameter == 'Conductivity' or 'conductivty':
        observation = "parameter_observation"
        #parameter = "Conductivity"

    elif parameter == "discharge" or parameter == "FlowLevel":
        #parameter = "discharge"
        df_raw = df_raw
        observation = "q_observation"
    return df_raw, parameter, observation, end_time