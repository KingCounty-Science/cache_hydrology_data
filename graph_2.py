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
    'corrected_data': r'rgba(29, 105, 150, 0.6)',
    'comparison' : r'rgba(152, 78, 163, 0.6)',
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
def parameter_graph(df, site_code, site_name, parameter, comparison_site, comparison_parameter, data_axis, corrected_data_axis, derived_data_axis, observation_axis, comparison_axis, primary_min, primary_max, secondary_min, secondary_max):
    df = df.sort_values(by='datetime', ascending=False)
    if parameter == "FlowLevel":
        base_parameter = "water_level"
        derived_parameter = "discharge"
    else:
        base_parameter = parameter
        derived_parameter = parameter

    if data_axis == "primary":
        data_axis = False # secondary-y=Faluse
    elif data_axis == "secondary":
        data_axis = True

    elif corrected_data_axis == "primary":
        corrected_data_axis = False
    elif corrected_data_axis == "secondary":
        corrected_data_axis = True

    elif derived_data_axis == "primary":
        derived_data_axis == False
    elif derived_data_axis == "secondary":
        derived_data_axis = True

    #elif observation_axis == "primary":
    #    observation_axis = False
    #elif observation_axis == "secondary":
    #    observation_axis = True

    elif comparison_axis == "primary":
        comparison_axis = False
    elif comparison_axis == "secondary":
        comparison_axis = True

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
    except:
        pass
    # get site number?
     # replace _ with space
    #subplot_titles = [value.replace("_", " ") for value in df.index.unique()]
    subplot_titles = parameter
    number_of_rows = 1
    number_of_columns = 1
    title_font_size = 45 # plot title
    annotation_font_size = 45 # subplot titels are hardcoded as annotations
    show_subplot_titles = False # supblot titles are hardcoded as annotations
    font_size = 27 # axis lables, numbers, offset information
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
    fig.update_layout(legend=dict(orientation=legend_orientation, x=legend_x, y=legend_y), showlegend=show_legend)

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


    fig.update_xaxes(range=[df['datetime'].min(),df['datetime'].max()], showticklabels=True, ticks="inside", tickformat='%b-%d', showgrid=False, showline=True, linecolor='black', linewidth=2, mirror = True)

   
        
    if f"data" in df.columns and data_axis != "none":
        fig.add_trace(go.Scatter(
                    x=df.loc[:, "datetime"],
                    y=df.loc[:, f"data"],
                    line=dict(color=color_map.get(f"data", 'black'), width = 1),
                    name=f"raw {base_parameter.replace('_', ' ')}",showlegend=True,),row=row_count, col=1, secondary_y=data_axis),
            
    if f"corrected_data" in df.columns and corrected_data_axis != 'none':        
        fig.add_trace(go.Scatter(
                    x=df.loc[:, "datetime"],
                    y=df.loc[:, f"corrected_data"],
                    line=dict(color=color_map.get(f"corrected_data", 'black'), width = 2),
                    name=f"corrected {base_parameter.replace('_', ' ')}",showlegend=True,),row=row_count, col=1, secondary_y=corrected_data_axis),
           
    # special graph
    if f"{derived_parameter}" in df.columns and derived_data_axis != "none":
            fig.update_yaxes(title_text=f"{derived_parameter.replace('_', ' ')} ({config[derived_parameter]['unit']})", row=row_count, col=1, showticklabels=True, secondary_y=True, )
            #fig.update_yaxes(showgrid=False, showticklabels=False, row=row_count, col=1, secondary_y=True)
            fig.add_trace(go.Scatter(
                x=df.loc[:, "datetime"],
                y=df.loc[:, f"{derived_parameter}"],
                line=dict(color=color_map.get(f"{derived_parameter}", 'black'), width = 5),
                name=f"{derived_parameter.replace('_', ' ')}",showlegend=True,),row=row_count, col=1, secondary_y=derived_data_axis),

    # comparison graph    
    if "comparison" in df.columns and comparison_axis != "none":
        #df['comparison'] = df['comparison'].astype(float)
        df['comparison'] = pd.to_numeric(df['comparison'], errors='coerce')
        fig.add_trace(go.Scatter(
                x=df.loc[:, "datetime"],
                y=df.loc[:, f"comparison"],
                line=dict(color=color_map.get(f"comparison", 'black'), width = 5),
                name=f"comparison ({comparison_site} {comparison_parameter})",showlegend=True,),row=row_count, col=1, secondary_y=comparison_axis),
    # dry indicator
    if f"dry_indicator" in df.columns and data_axis != "none":
        
        df.loc[df['dry_indicator'] == "dry indicator", "dry_indicator"] = df['data'].min()
        df.loc[df['dry_indicator'] == " ", "dry_indicator"] =  np.nan # graph dry data is data min for visualization
        fig.add_trace(go.Scatter(
                x=df.loc[:, "datetime"], # 1 is graph if dry 0 is graph if not dry
                y=df.loc[:, f"dry_indicator"],
                line=dict(color=color_map.get(f"dry_indicator", 'black'), width = 5),
                name=f"dry indicator",showlegend=True,),row=row_count, col=1, secondary_y=data_axis),
       
        print(df)
    def annotations(obs):
            row_count = 1
            annotation_x = 0.00 #0.05 # allows offset for when year is displatyed on axis
            annotation_y = -.085
            
              # annotation 
            obs_df = df.dropna(subset=[f"{obs}"]).copy() # this solves the Try using .loc[row_indexer,col_indexer] = value instead as obs_df is a slice
            obs_df = obs_df.sort_values(by='datetime', ascending=True)
            if 'offset' not in obs_df:
                 obs_df[f'offset'] = obs_df[f'observation_stage']-obs_df[f'data']
            
            if obs_df.shape[0] > 0:
                # first observation
                fig.add_annotation(text=f"{obs_df['datetime'].iloc[0].strftime('%Y-%m-%d %H:%M')}",
                        xref="x domain", yref="y domain",
                        x=annotation_x, y=annotation_y, showarrow=False, row=row_count, col=1, secondary_y=False,)
                fig.add_annotation(text=f"obs: {obs_df[f'{obs}'].iloc[0]} | inst: {round(obs_df[f'data'].iloc[0], 2)}",
                        xref="x domain", yref="y domain",
                        x=annotation_x, y=annotation_y-.03, showarrow=False, row=row_count, col=1, secondary_y=False,)
                fig.add_annotation(text=f"offset: {round(obs_df[f'offset'].iloc[0], 2)}",
                        xref="x domain", yref="y domain",
                        x=annotation_x, y=annotation_y-.06, showarrow=False, row=row_count, col=1, secondary_y=False,)
              
                #fig.add_annotation(text=f"offset: {round(obs_df[f'offset'].iloc[0], 2)}",
                #        xref="x domain", yref="y domain",
                #        x=annotation_x, y=annotation_y-.09, showarrow=False, row=row_count, col=1, secondary_y=False,)
                # last observation
                
                fig.add_annotation(text=f"{obs_df['datetime'].iloc[-1].strftime('%Y-%m-%d %H:%M')}",
                        xref="x domain", yref="y domain",
                        x=annotation_x+1, y=annotation_y, showarrow=False, row=row_count, col=1, secondary_y=False,) #0.95
                fig.add_annotation(text=f"obs: {obs_df[f'{obs}'].iloc[-1]} | inst: {round(obs_df[f'data'].iloc[-1], 2)}",
                        xref="x domain", yref="y domain",
                        x=annotation_x+1, y=annotation_y-.03, showarrow=False, row=row_count, col=1, secondary_y=False,)
                fig.add_annotation(text=f"offset: {round(obs_df[f'offset'].iloc[-1], 2)}",
                        xref="x domain", yref="y domain",
                        x=annotation_x+1, y=annotation_y-.06, showarrow=False, row=row_count, col=1, secondary_y=False,)
                
                #fig.add_annotation(text=f"offset: {round(obs_df[f'offset'].iloc[-1], 2)}",
                #        xref="x domain", yref="y domain",
                #        x=annotation_x+1, y=annotation_y-.09, showarrow=False, row=row_count, col=1, secondary_y=False,)
                    
                    # shift
                #if observation_axis != "none":
                fig.add_annotation(text=f"session shift: {round((obs_df[f'offset'].iloc[-1] - obs_df[f'offset'].iloc[0]),2)}",
                            xref="x domain", yref="y domain",
                            x=.5, y=legend_y+.02, showarrow=False, row=row_count, col=1, secondary_y=False,)
    # display field observation points    
    if "field_observations" in df.columns or "observations" in df.columns or "observation" in df.columns or "observation_stage" in df.columns and observation_axis != "none":
            if "field_observations" in df.columns:
                obs = "field_observations"
            if "observations" in df.columns:
                obs = "observations"
            if "observation" in df.columns:
                obs = "observation"
            if "observation_stage" in df.columns:
                obs = "observation_stage"
            fig.add_trace(go.Scatter(
                x=df['datetime'],
                y=df[f'{obs}'],
            mode='markers',
            marker=dict(
                color=color_map.get(f"field_observation", 'black'), size=12, opacity=.9),
            text='', name=f"{obs.replace('_', ' ')}", showlegend=True), row=row_count, col=1, secondary_y=corrected_data_axis)
            # corrected_data_axis
            annotations(obs)
    
    ### Discharge annotation
    def q_annotations(obs):
            row_count = 1
            annotation_x = 0.05 # allows offset for when year is displatyed on axis (obs is .05)
            annotation_y = 1 # up and down? (obs is -.07)
            
              # annotation 
            obs_df = df.dropna(subset=[f"{obs}"]).copy() # this solves the Try using .loc[row_indexer,col_indexer] = value instead as obs_df is a slice
            obs_df['rating_number'] = obs_df['rating_number'].replace('NONE', '')
            obs_df['q_offset'] = obs_df['q_offset'].replace(np.nan, '')
            obs_df = obs_df.sort_values(by='datetime', ascending=True)
            if obs_df.shape[0] > 0:
                # first observation
                #fig.add_annotation(text=f"{obs_df['datetime'].iloc[0].strftime('%Y-%m-%d %H:%M')}",
                #        xref="x domain", yref="y domain",
                #        x=annotation_x, y=annotation_y, showarrow=False, row=row_count, col=1, secondary_y=False,)

                # rating
                # qm: {obs_df['observation_number'].iloc[0]}
                fig.add_annotation(text=f"rating: {obs_df['rating_number'].iloc[0]}",
                        xref="x domain", yref="y domain",
                        x=annotation_x, y=annotation_y, showarrow=False, row=row_count, col=1, secondary_y=False,)
                
                fig.add_annotation(text=f"obs: {obs_df[f'{obs}'].iloc[0]}",
                        xref="x domain", yref="y domain",
                        x=annotation_x, y=annotation_y-.03, showarrow=False, row=row_count, col=1, secondary_y=False,)
                
                fig.add_annotation(text=f"offset: {obs_df[f'q_offset'].iloc[0]} ({obs_df[f'precent_q_change'].iloc[0]}%)",
                        xref="x domain", yref="y domain",
                        x=annotation_x, y=annotation_y-.06, showarrow=False, row=row_count, col=1, secondary_y=False,)
              
                """fig.add_annotation(text=f"offset: {round(obs_df[f'offset'].iloc[0], 2)}",
                        xref="x domain", yref="y domain",
                        x=annotation_x, y=annotation_y-.075, showarrow=False, row=row_count, col=1, secondary_y=False,)"""
                # last observation
                
                #fig.add_annotation(text=f"{obs_df['datetime'].iloc[-1].strftime('%Y-%m-%d %H:%M')}",
                #        xref="x domain", yref="y domain",
                #       x=annotation_x+.95, y=annotation_y, showarrow=False, row=row_count, col=1, secondary_y=False,)

                # {obs_df['observation_number'].iloc[-1]} 
                fig.add_annotation(text=f"rating: {obs_df['rating_number'].iloc[-1]}",
                        xref="x domain", yref="y domain",
                       x=annotation_x+.95, y=annotation_y, showarrow=False, row=row_count, col=1, secondary_y=False,)
                
                fig.add_annotation(text=f"obs: {obs_df[f'{obs}'].iloc[-1]}",
                        xref="x domain", yref="y domain",
                        x=annotation_x+.95, y=annotation_y-.03, showarrow=False, row=row_count, col=1, secondary_y=False,)
                fig.add_annotation(text=f"offset: {obs_df[f'q_offset'].iloc[-1]} ({obs_df[f'precent_q_change'].iloc[-1]}%)",
                        xref="x domain", yref="y domain",
                        x=annotation_x+.95, y=annotation_y-.06, showarrow=False, row=row_count, col=1, secondary_y=False,)
                
                """fig.add_annotation(text=f"offset: {round(obs_df[f'offset'].iloc[-1], 2)}",
                        xref="x domain", yref="y domain",
                        x=annotation_x+.95, y=annotation_y-.075, showarrow=False, row=row_count, col=1, secondary_y=False,)"""
                    
                    # shift
                #if observation_axis != "none":
                try: # if offset is nan it wont graph
                    fig.add_annotation(text=f"rating shift: {round((obs_df[f'q_offset'].iloc[-1] - obs_df[f'q_offset'].iloc[0]),2)}",
                                xref="x domain", yref="y domain",
                                x=annotation_x+.5, y=annotation_y, showarrow=False, row=row_count, col=1, secondary_y=False,)
                except:
                    pass
    # display field observation points    
    if "q_observation" in df.columns or "discharge_observation" in df.columns and observation_axis != "none":
            if "q_observation" in df.columns:
                obs = "q_observation"
            if "discharge_observation" in df.columns:
                obs = "discharge_observation"
           
            fig.add_trace(go.Scatter(
                x=df['datetime'],
                y=df[f'{obs}'],
            mode='markers',
            marker=dict(
                color=color_map.get(f"q_observation", 'black'), size=12, opacity=.9),
            text='', name=f"{obs.replace('_', ' ')}", showlegend=True), row=row_count, col=1, secondary_y=derived_data_axis)
            # corrected_data_axis
            q_annotations(obs)
    row_count = row_count+1
    
    return fig

def cache_graph_export(df, site_code, site_name, parameter, comparison_site, comparison_parameter, data_axis, corrected_data_axis, derived_data_axis, observation_axis, comparison_axis, primary_min, primary_max, secondary_min, secondary_max):
    fig = parameter_graph(df, site_code, site_name, parameter, comparison_site, comparison_parameter, data_axis, corrected_data_axis, derived_data_axis, observation_axis, comparison_axis, primary_min, primary_max, secondary_min, secondary_max)
   
    
   
    fig.update_layout(autosize=True, height = 1000)
    return html.Div(dcc.Graph(figure = fig), style = {'width': '100%', 'height': '100%'})

#def save_fig(df, site_code, site_name, parameter, comparison_site, comparison_parameter, data_axis, corrected_data_axis, derived_data_axis, observation_axis, comparison_axis):
def save_fig(df, site, site_sql_id, parameter, comparison_site, comparison_parameter, rating, data_axis, corrected_data_axis, derived_data_axis, observation_axis, comparison_axis, end_date, primary_min, primary_max, secondary_min, secondary_max):
    # end date
    fig = parameter_graph(df, site_sql_id, site, parameter, comparison_site, comparison_parameter, data_axis, corrected_data_axis, derived_data_axis, observation_axis, comparison_axis, primary_min, primary_max, secondary_min, secondary_max)
    #start_date = df.head(1).iloc[0, df.columns.get_loc("datetime")].date().strftime("%Y_%m_%d")
   # end_date = df.tail(1).iloc[0, df.columns.get_loc("datetime")].date().strftime("%Y_%m_%d")
    #end_date = df['datetime'].max().date()

    # end_date = dt.datetime.strftime(df['datetime'].max(), '%Y-%m-%d')
    paper_width = 2300
    paper_height = 1300
    fig.update_layout(autosize=True, width=paper_width, height = paper_height)
    #fig.update_layout(autosize=True, width=paper_width, height = paper_height)
    file_path = r"W:\STS\hydro\GAUGE\Temp\Ian's Temp\{0}_{1}_{2}.pdf".format(site, parameter, end_date)

    # Use plotly.io.write_image to export the figure as a PDF
    pio.write_image(fig, file_path, format='pdf')
   
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