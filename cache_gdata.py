# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:29:39 2021
updated Friday Feb 14 2025

@author: IHiggins
"""
import base64
import io

import json
import pyodbc
import configparser
import dash
from dash import callback_context
from dash import Dash, html, Input, Output, callback, ctx
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
from dotenv import load_dotenv

load_dotenv()
socrata_api_id = os.getenv("socrata_api_id")
socrata_api_secret = os.getenv("socrata_api_secret")

## fix copy of slice error with df.loc[df.A > 5, 'A'] = 1000

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets, long_callback_manager=long_callback_manager)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#Driver = 'SQL Server'
#Server = 'KCITSQLPRNRPX01'
#Database = 'gData'
#Trusted_Connection = 'yes'

import configparser
config = configparser.ConfigParser()
config.read('gdata_config.ini')

comparison_sites = configparser.ConfigParser()
comparison_sites.read('gdata_config.ini')


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



Discharge_Table = 'tblDischargeGauging'
Discharge_Table_Raw = 'D_Value'
Discharge_DateTime = 'D_TimeDate'

Site_Table = 'tblGaugeLLID'


# Query SQL every sql query will need its own CONN
# INITIAL AVAILABLE BAROMOTERS



# Barometer Association Table
Barometer_Association_Table = 'tblBaroLoggerAssociation'

# get available sites
from import_socrata_data import site_metadata

site_list, barometer_list = site_metadata()


comparison_list = pd.read_csv('external_comaprison_sites.csv', skipinitialspace=True)


desired_order = ["site", "parameter"] # observation and observation_stage are kinda redundent at some point and should be clarified
existing_columns = [col for col in desired_order if col in comparison_list.columns] 
# Filter out columns that exist in the DataFrame   
comparison_list = comparison_list[existing_columns].copy()
comparison_list["parameter"] = comparison_list["parameter"].str.split(", ") # split parameter colum into list
comparison_list = comparison_list.explode("parameter", ignore_index=True) # create a row for each site/parameter column
comparison_list = comparison_list.drop_duplicates() # remove duplicates incase a parameter is repeated
comparison_list = [f"{site} {parameter}" for site, parameter in zip(comparison_list["site"], comparison_list["parameter"])] # convert to list

#comparison_list = comparison_list["site"].values.tolist()
comparison_list = site_list + comparison_list





app.layout = html.Div([
    # dcc.Location(id='url', refresh=False),
    # Select a Site
    # Site = site name site_sql_id is site number
    # select site, paramter and data source
    html.Div([
        # site selector
        html.Div([
            html.Label('Select Site and Parameter', style={'marginBottom': '5px', 'fontWeight': 'bold', 'fontSize': '16px'}),
            # select site
            dcc.Dropdown(
                id='site_selector',
                options=[{'label': i, 'value': i} for i in site_list],
                style={'width': '100%', 'fontSize': '14px'}
            )
            ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
        # select date range
        html.Div(id='Select_Data_Source_Output', style={'width': '15%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'fontSize': '14px'}),
        # import or queryt
        html.Div([
            html.Label('Select Import or Query', style={'marginBottom': '5px', 'fontWeight': 'bold', 'fontSize': '16px'}),
            daq.ToggleSwitch(id='Select_Data_Source', value=False)
        ], style={'width': '15%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
    
    ], style={'backgroundColor': '#ccccff', 'border': '2px solid #5e006a', 'padding': '10px', 'display': 'flex', 'alignItems': 'flex-start'}),
    # storage
    html.Div(id='site', style={'display': 'none'}),
    html.Div(id='site_sql_id', style={'display': 'none'}),
    html.Div(id='parameter', style={'display': 'none'}),

    # data level need to move this
    html.P(id="data_label", children=["data column axis"]), 
    dcc.RadioItems(id='data_axis', options=['data', 'corrected_data'], value='data', inline=True),
    #html.Div(dcc.RangeSlider(0, 20, 1, value=[5, 15], id='select_datetime_by_obs'), id = "select_datetime_by_obs_dev", style={'display': 'block'}),
    
    # date time selector
# date time selector with accordion
html.Details([
    html.Summary("Date/Time Selector", style={
        'font-weight': 'bold',
        'font-size': '16px',
        'padding': '10px',
        'background-color': '#e3a66a',
        'border-radius': '5px',
        'cursor': 'pointer',
        'list-style': 'none',
        'user-select': 'none'
    }),
    html.Div([
        # Left section - Date range inputs (35%)
        html.Div([
            html.Label("Select date range", style={'margin-bottom': '15px', 'font-weight': 'bold', 'text-align': 'center', 'display': 'block', 'font-size': '16px'}),
            html.Div([
                html.Div([
                    html.Label('Start Datetime:', style={'text-align': 'center', 'display': 'block', 'margin-bottom': '5px', 'font-size': '15px'}),
                    dbc.Input(id='startDate', type='datetime-local', style={'width': '100%', 'font-size': '15px', 'padding': '6px'})
                ], style={'flex': '1'}),
                html.Div([
                    html.Label('End Datetime:', style={'text-align': 'center', 'display': 'block', 'margin-bottom': '5px', 'font-size': '15px'}),
                    dbc.Input(id='endDate', type='datetime-local', style={'width': '100%', 'font-size': '15px', 'padding': '6px'})
                ], style={'flex': '1'})
            ], style={'display': 'flex', 'gap': '10px'})
        ], style={'flex': '0 0 35%', 'padding-right': '20px', 'background-color': '#eed8a4'}),
        
        # Right section - Observation dropdowns (65%)
        html.Div([
            html.Label("Select by observation", style={'margin-bottom': '15px', 'font-weight': 'bold', 'text-align': 'center', 'display': 'block', 'font-size': '16px'}),
            html.Div([
                html.Div([
                    html.Label('First observation:', style={'text-align': 'center', 'display': 'block', 'margin-bottom': '5px', 'font-size': '15px'}),
                    dcc.Dropdown(id='select_datetime_by_obs_a', options=[{"label": "", "value": ""}], value='', style={'font-size': '15px'})
                ], style={'flex': '1'}),
                html.Div([
                    html.Label('Last observation:', style={'text-align': 'center', 'display': 'block', 'margin-bottom': '5px', 'font-size': '15px'}),
                    dcc.Dropdown(id='select_datetime_by_obs_b', options=[{"label": "", "value": ""}], value='', style={'font-size': '15px'})
                ], style={'flex': '1'})
            ], style={'display': 'flex', 'gap': '10px'})
        ], style={'flex': '0 0 60%', 'padding-left': '20px', 'background-color': '#eed8a4'})
        
    ], style={
        'display': 'flex',
        'padding': '15px',
        'align-items': 'flex-start',
        'backgroundColor': '#eed8a4'
    })
], id='select_datetime_by_obs_dev', 
   open=True,  # Set to False if you want it collapsed by default
   style={
       'border': '2px solid #e3a66a', 
       'border-radius': '5px', 
       'margin': '10px 0'
   }),

    dcc.Store(id='query_start_date'),  
    dcc.Store(id='query_end_date'),    
    dcc.Store(id="statistics"),
    
    

    # Barometric Correction Radio Button
    # dynamic visability https://stackoverflow.com/questions/50213761/changing-visibility-of-a-dash-component-by-updating-other-component
  
                      



    html.Div(id='New_Callback'),
    ### File structure
    # Import file structures
    # Wrap the entire structure in a div with an ID
html.Details([
    html.Summary("File Upload & Configuration", style={
        'font-weight': 'bold',
        'font-size': '16px',
        'padding': '10px',
        'background-color': '#439473',
        'border-radius': '5px',
        'cursor': 'pointer',
        'list-style': 'none',
        'user-select': 'none',
        'color': 'white'
    }),
    html.Div([
        html.Div([
             # Middle: Upload component (15%)
            html.Div([
                dcc.Upload(
                    id='datatable-upload',
                    children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                    style={
                        'width': '100%', 
                        'height': '60px', 
                        'lineHeight': '60px',
                        'borderWidth': '1px', 
                        'borderStyle': 'dashed',
                        'borderRadius': '5px', 
                        'textAlign': 'center'
                    })
            ], style={'width': '15%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '15px'}),
            # Left side: Radio items and numeric inputs (30%)
            html.Div([
                html.Label('Select File Structure', style={'marginBottom': '10px', 'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='File_Structure',
                    options=[
                        {'label': 'onset_U20', 'value': 'onset_U20'},
                        {'label': 'onset_U24', 'value': 'onset_U24'},
                        {'label': 'aqua4plus_ct2x', 'value': 'aqua4plus_ct2x'},
                        {'label': 'csv', 'value': 'csv'}
                    ], 
                    value='onset_U20', 
                    labelStyle={'display': 'inline-block', 'margin-right': '15px'}
                ),
                html.Div([
                    html.Div(daq.NumericInput(id='HEADER_ROWS', label='HEADER ROWS', labelPosition='top', value=1), 
                             style={'width': '25%', 'display': 'inline-block', 'padding': '5px'}),
                    html.Div(daq.NumericInput(id='FOOTER_ROWS', label='FOOTER ROWS', labelPosition='top', value=0), 
                             style={'width': '25%', 'display': 'inline-block', 'padding': '5px'}),
                    html.Div(daq.NumericInput(id='TIMESTAMP_COLUMN', label='TIMESTAMP_COLUMN', labelPosition='top', value=1), 
                             style={'width': '25%', 'display': 'inline-block', 'padding': '5px'}),
                    html.Div(daq.NumericInput(id='DATA_COLUMN', label='DATA_COLUMN', labelPosition='top', value=1), 
                             style={'width': '25%', 'display': 'inline-block', 'padding': '5px'}),
                ], style={'marginTop': '10px'})
            ], style={
                'width': '30%', 
                'display': 'inline-block', 
                'verticalAlign': 'top', 
                'padding': '15px',
                'border': '2px solid #cee5d0',
                'backgroundColor': "#B2E6D1"
            }),
            
           
            
            # Barometric correction radio buttons (20%)
            html.Div([
        html.Label('Barometric Correction', style={'marginBottom': '5px', 'fontWeight': 'bold', 'display': 'block'}),
        dcc.RadioItems(
            id='Barometer_Button',
            options=[
                {'label': 'Barometric Correction', 'value': 'Baro'}, 
                {'label': 'No Correction', 'value': 'No_Baro'}
            ], 
            value='Baro')
    ], style={
        'width': '20%',
        'display': 'inline-block', 
        'verticalAlign': 'top',
        'padding': '15px'
    }),

            # Dropdown and Button side by side (20% + remaining space)
           html.Div([
        html.Label('   ', style={'marginBottom': '5px', 'fontWeight': 'bold', 'display': 'block'}),
        dcc.Dropdown(
            id='available_barometers', 
            options=[{'label': i, 'value': i} for i in barometer_list], 
            style={'display': 'none', 'width': '100%'}
        )
    ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '15px'}),

    # Button (10%)
    html.Div([
        html.Button('Delete Association', id='Delete_Association', n_clicks=0, 
                   style={'display': 'none', 'padding': '10px 20px', 'fontSize': '14px'})
    ], style={'width': '10%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '15px'}),
            
        ], style={
            'display': 'flex', 
            'alignItems': 'flex-start', 
            'gap': '10px',
            'padding': '10px',
            'backgroundColor': '#cee5d0'
        })
    ])
], id='File_Structure_Container',
   open=True,  # Set to False if you want it collapsed by default
   style={
       'border': '2px solid #439473',
       'borderRadius': '5px',
       'margin': '10px 0'
   }),

    # date time picker not native to dash see https://community.plotly.com/t/dash-timepicker/6541/10
    
   
    

    # page_action='none',
   
    
    html.Div(id='output-container-date-picker-range'),

  
    html.Div(id='graph_output', style={'width': '50vw', }), # Use viewport width to scale with screen size
    html.Div(dcc.Graph(id='graph',figure=go.Figure(),config={'responsive': True, 'modeBarButtonsToAdd': ['select2d', 'lasso2d']}),),    
    

    html.Div(id="graph_where"),
    # it is hard to make this dcc dropdown take up enough room inclosing it in html.Div([]) restricts it to a small window

    # Add this button to your layout
    # store selected data
  

    # In your layout
    #dcc.Store(id='selected-data-store'),
    #html.Button('Clear Selection', id='clear-selection-button', n_clicks=0),
    html.Div(dcc.Dropdown(id='Ratings', value='NONE'), style={'display': 'block'}),

    html.Div(id="display"),
    dcc.Store(id='observations', storage_type='memory', data = pd.DataFrame().to_json(orient="split")), # set initial observation to blank), # store observations
    dcc.Store(id='import_data', storage_type='memory', data = pd.DataFrame().to_json(orient="split")),
    #dcc.Store(id='import_data', storage_type='memory'),
    dcc.Store(id='comparison_data', storage_type='memory', data = pd.DataFrame().to_json(orient="split")),

    # ### data dables
    html.Div([
       html.Details([
    html.Summary("Data Table", style={
        'font-weight': 'bold',
        'font-size': '16px',
        'padding': '10px',
        'background-color': '#984EA3',
        'border-radius': '5px',
        'cursor': 'pointer',
        'list-style': 'none',
        'user-select': 'none',
        'color': 'white'
    }),
    # data table div
    html.Div([
        # data data table
        html.Div(dash_table.DataTable(
            id="corrected_data_datatable", 
            editable=True, 
            sort_action="native", 
            sort_mode="multi", 
            fixed_rows={'headers': True}, 
            row_deletable=False,
            page_action='none', 
            style_table={'height': 'calc(100vh - 250px)', 'overflowY': 'auto'}, 
            virtualization=True, 
            fill_width=False, 
            filter_action='native',
            style_data={
                'width': '200px', 
                'maxWidth': '200px',
                'fontSize': '14px',
                'fontFamily': 'Arial, sans-serif'
            },
            style_header={
                'textAlign': 'center',
                'fontSize': '15px',
                'fontWeight': 'bold'},
            style_cell={'fontSize': '14px'},
            # data table style
            style_data_conditional=[
                {'if': {'state': 'selected'},'backgroundColor': '#FFDDC1',  'color': 'black'},
                {'if': {'column_id': 'c_discharge',},'backgroundColor': r'rgba(152, 78, 163, 0.3)', },
                {'if': {'column_id': 'c_stage',},'backgroundColor': r'rgba(152, 78, 163, 0.3)', },
                {'if': {'column_id': 'c_water_level',},'backgroundColor': r'rgba(152, 78, 163, 0.3)',},
                {'if': {'column_id': 'c_water_temperature',},'backgroundColor': r'rgba(152, 78, 163, 0.3)',},
                {'if': {'column_id': 'datetime'}, 'width': '120px', 'maxWidth': '120px', 'minWidth': '120px', 'textAlign': 'center'},
                {'if': {'column_id': 'data'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
                {'if': {'column_id': 'corrected_data'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
                {'if': {'column_id': 'discharge'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
                {'if': {'column_id': 'observation_stage'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
                {'if': {'column_id': 'c_water_level'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
                {'if': {'column_id': 'offset'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
                {'if': {'column_id': 'estimate'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
                {'if': {'column_id': 'warning'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
                {'if': {'column_id': 'non_detect'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
                {'if': {'column_id': 'comments'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
                {'if': {'column_id': 'c_stage'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
                {'if': {'column_id': 'c_discharge'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
                {'if': {'column_id': 'observation_number'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
                {'if': {'column_id': 'q_observation'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
                {'if': {'column_id': 'q_offset'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
                {'if': {'column_id': 'precent_q_change'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
                {'if': {'column_id': 'rating_number'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
            ],
        ))
    ], style={'padding': '10px'})
], 
open=True,
style={
    'border': '2px solid #984EA3',
    'borderRadius': '5px',
    'margin': '10px 0',
    'flex': '60%', 
    'width': '60%'
}),
        
        html.Details([
    html.Summary("Field Observations", style={
        'font-weight': 'bold',
        'font-size': '16px',
        'padding': '10px',
        'background-color': '#4A90E2',
        'border-radius': '5px',
        'cursor': 'pointer',
        'list-style': 'none',
        'user-select': 'none',
        'color': 'white'
    }),
    html.Div([
        dash_table.DataTable(
            id='observations_datatable', 
            columns=[], 
            data=[], 
            editable=True, 
            fixed_rows={'headers': True}, 
            row_deletable=True,
            row_selectable='multi',
            style_table={
                'height': 'calc(100vh - 350px)',
                'overflowY': 'auto',
            },
            style_data={
                'fontSize': '14px',
                'fontFamily': 'Arial, sans-serif'
            },
            style_header={
                'textAlign': 'center',
                'fontSize': '15px',
                'fontWeight': 'bold'
            },
            style_cell={
                'fontSize': '14px'
            },
            style_data_conditional=[
                {'if': {'column_id': 'datetime'}, 'width': '120px', 'maxWidth': '120px', 'minWidth': '120px', 'textAlign': 'center'},
                {'if': {'column_id': 'observation_stage'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
                {'if': {'column_id': 'comments'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
                {'if': {'column_id': 'q_observation'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
                {'if': {'column_id': 'observation_number'}, 'width': '90px', 'maxWidth': '90px', 'minWidth': '90px', 'textAlign': 'center'},
            ],
            virtualization=True,
            
        ),
        
        html.Div([
            html.Button('add row', id='add_row_button_a', n_clicks=0, 
                       style={'fontSize': '14px', 'padding': '8px 12px', 'marginRight': '10px'}),
            html.Button('refresh', id='refresh_button_a', n_clicks=0,
                       style={'fontSize': '14px', 'padding': '8px 12px', 'marginRight': '10px'}),
            html.Label("Show observation range", style={'fontSize': '14px', 'marginRight': '10px'}),
            daq.ToggleSwitch(id='show_inrange_observations', value=True)
        ], style={'marginTop': '15px', 'display': 'flex', 'alignItems': 'center', 'gap': '10px'})
        
    ], style={'padding': '10px'})
], 
open=True,
style={
    'border': '2px solid #4A90E2',
    'borderRadius': '5px',
    'margin': '10px 0',
    'width': '35%',
    'display': 'inline-block',
    'verticalAlign': 'top'
}),
# settings and controls div
    html.Details([
        html.Summary("Settings & Controls", style={
            'font-weight': 'bold',
            'font-size': '16px',
            'padding': '8px 10px',
            'background-color': '#FF8C42',
            'border-radius': '5px',
            'cursor': 'pointer',
            'list-style': 'none',
            'user-select': 'none',
            'color': 'white'
        }),
    html.Div([
        html.Div([
            dcc.Dropdown(id='comparison_sites',options=[{'label': i, 'value': i} for i in comparison_list], multi=True),]),
            #dcc.Dropdown(id='comparison_parameter', value='0'),
            dcc.Checklist(id="checklist", options=['comparison_site'],value=['comparison_site'],inline=True),
            html.Div([
                        html.Label("primer hours before/after"),
                        daq.NumericInput(id = "primer_hours", label='',labelPosition='',value=2,),]),
            # realtime update info
            html.Div([daq.ToggleSwitch(id='realtime_update'), html.Button(id="run_job", children="Run Job!"), html.Div(id='realtime_update_info'),], style={'display': 'flex', 'flex-direction': 'row'}), #dynamic default so sql query doesnt automatically correct for obs
            html.Div([daq.ToggleSwitch(id='apply_discharge_offset', value=False), html.Div(id='offset_label') ], id='offset_container', style={'display': 'none', 'marginLeft': '1rem'}),
            html.P(id="paragraph_id", children=["Button not clicked"]),
        # interpolation and graphing
        html.Div([
            
            html.Button("data managment", id="open-modal-button"),
                dbc.Modal([dbc.ModalHeader("data managment"),
                dbc.ModalBody([
                    html.Label('for import sites 1. Resampel to interval, 2. Run expansion, 3. Resample to 15 min, 4. Fill'),
                    html.Div([html.Label('expand to observations: select start and/or end observation to fill to, will need to run filler'), html.Button('run expansion', id='to_observations_button'),], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center','gap': '10px'}),
                    html.Div(id="data_interval", children="data_interval")]),
                            dcc.RangeSlider(id='interval', min=0, max=5, step=None, marks={0: '1 min', 1: '5 min', 2: '15 min', 3: '30 min', 4: 'hourly', 5: 'daily'}, value=[2]),
                            html.Button('resample', id='resample_button'),                
                    # interpolation
                    html.Div([
                        html.Button('calculate_average', id='calculate_average_button'), 
                        html.Button('interpolate', id='interpolate_button'), 
                        html.Button('accept interpolation', id='accept_interpolation_button'), 
                    ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center','gap': '10px'}),
                    # fill values
                    html.Div([html.Label("basic forward and backward fill"),], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'margin-top': '20px'}),
                    html.Div([
                        html.Div([html.Label('limit consecutive na to fill'), dcc.Dropdown(id='fill_limit',options=[{'label': 'no limit', 'value': 'no limit'},{'label': 'limit', 'value': 'limit'}], value='no_limit')], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center','gap': '10px'}),
                        html.Div([html.Label('enter number on consecutive na to fill'), daq.NumericInput(id='fill_limit_number', value=4,)], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center','gap': '10px'}),
                        #html.Div(daq.NumericInput(id='HEADER_ROWS', label='HEADER ROWS', labelPosition='top', value=1,), style={'width': '10%', 'display': 'inline-block'}),
                        html.Div([html.Label('limit area'), dcc.Dropdown(id='fill_limit_area',options=[{'label': 'inside (na surrounded by data)', 'value': 'inside'},{'label': 'outside (na outside data)', 'value': 'outside'}], value='outside')], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center','gap': '10px'}),
                        html.Button('basic forward fill', id='basic_forward_fill'), 
                        html.Button('basic backward fill', id='basic_backward_fill'), 
                    ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center','gap': '10px'}),
                    html.Div([
                        html.Div([html.Label('select interoplation method'), dcc.Dropdown(id='set_method',options=[{'label': 'pad', 'value': 'pad'},{'label': 'linear', 'value': 'linear'},], value='linear')], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center','gap': '10px'}),
                        html.Div([html.Label('limit consecutive na to fill'), dcc.Dropdown(id='set_limit',options=[{'label': 'no limit', 'value': 'no limit'},{'label': 'limit', 'value': 'limit'}], value='limit')], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center','gap': '10px'}),
                        html.Div([html.Label('enter number on consecutive na to fill'), daq.NumericInput(id='limit_number', value=4,)], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center','gap': '10px'}),
                        #html.Div(daq.NumericInput(id='HEADER_ROWS', label='HEADER ROWS', labelPosition='top', value=1,), style={'width': '10%', 'display': 'inline-block'}),
                        
                        html.Div([html.Label('limit direction'), dcc.Dropdown(id='limit_direction',options=[{'label': 'forward', 'value': 'forward'},{'label': 'backward', 'value': 'backward'},{'label': 'both', 'value': 'both'}], value='both')], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center','gap': '10px'}),
                        html.Div([html.Label('limit area'), dcc.Dropdown(id='limit_area',options=[{'label': 'inside (na surrounded by data)', 'value': 'inside'},{'label': 'outside (na outside data)', 'value': 'outside'}], value='inside')], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center','gap': '10px'}),
                        html.Button('fill missing data', id='fill_missing_data'), 


                    ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center','gap': '10px'}),

                    html.Div([
                        html.Label('run interpolation on data column', style={'margin-right': '10px'}),
                        dcc.Checklist(
                            id='interp-data', 
                            options=[{'label': 'Interpolate data', 'value': 'on'}], 
                            value=[],  # Empty = off, ['on'] = on
                            style={'display': 'inline-block'}
                            ),
                        html.Label('run interpolation on corrected data column', style={'margin-right': '10px'}),
                        dcc.Checklist(
                            id='interp-corrected-data', 
                            options=[{'label': 'Interpolate Corrected Data', 'value': 'on'}], 
                            value=[],  # Empty = off, ['on'] = on
                            style={'display': 'inline-block'}
                            )
                        ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center','gap': '10px'}),
                    html.Div([html.Label("Interpolation Functions"),], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'margin-top': '20px'}),
                    # dry indicator
                    html.Div([
                        html.Div([
                            html.Label("Apply dry/non-detect indicator"),
                            html.Div([
                                # Left column - text inputs
                                html.Div([
                                    html.Div([
                                        html.Label("start dry indicator range"),
                                        dcc.Input(
                                            id='start-dry-indicator-range',
                                            type='text',
                                            placeholder='YYYY-MM-DD HH:MM:SS',
                                            value='2024-01-01 00:00:00',
                                            style={'width': '200px'}
                                        )
                                    ], style={'margin': '10px'}),
                                    html.Div([
                                        html.Label("end dry indicator range"),
                                        dcc.Input(
                                            id='end-dry-indicator-range',
                                            type='text',
                                            placeholder='YYYY-MM-DD HH:MM:SS',
                                            value='2024-01-01 23:59:59',
                                            style={'width': '200px'}
                                        )
                                    ], style={'margin': '10px'}),
                                ], style={'flex': '1', 'padding-right': '20px'}),
                                # Right column - buttons
                                html.Div([
                                    html.Button('set dry indicator range', id='set-dry-indicator', 
                                            style={'margin': '5px', 'display': 'block'}),
                                    html.Button('clear all dry indicators', id='clear-dry-indicator',
                                            style={'margin': '5px', 'display': 'block'}),
                                ], style={'flex': '1', 'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center'}),
                            ], style={'display': 'flex', 'align-items': 'flex-start'}),
                        ], style={'border': '1px solid #ccc','padding': '15px','margin': '10px','border-radius': '4px' }),
                        # dry threshold
                        html.Div([
                                html.Label("apply dry threshold to raw data"),
                                dcc.Input(
                                    id='dry-threshold-raw-input',
                                    type='number',
                                    step='any',
                                    value=0,
                                    style={'width': '200px'}
                                ),
                                html.Button('apply dry threshold raw', id='apply-dry-threshold-raw'),
                            ], style={
                                'margin': '10px',
                                'border': '1px solid #ccc',  # thin border
                                'padding': '10px',           # add some padding inside the border
                                'border-radius': '4px'       # optional: rounded corners
                            }),
                        html.Div([
                                html.Label("apply dry threshold to corrected data"),
                                dcc.Input(
                                    id='dry-threshold-corrected-data-input',
                                    type='number',
                                    step='any',
                                    value=0,
                                    style={'width': '200px'}
                                ),
                                html.Button('apply dry threshold corrected data', id='apply-dry-threshold-corrected-data'),  
                            ], style={
                                'margin': '10px',
                                'border': '1px solid #ccc',  # thin border
                                'padding': '10px',           # add some padding inside the border
                                'border-radius': '4px'       # optional: rounded corners
                            }),
                            ], style={'display': 'flex'}),
                    dbc.ModalFooter(dbc.Button("close", id="close-modal-button", className="ml-auto")),], id="modal", size="xl",),
            # graphing options
            html.Button("graphing options", id="open-graphing-options-button"),
                dbc.Modal([
                    dbc.ModalHeader("select data axis"),
                    dbc.ModalBody([
                        html.Div([
                            html.Label("Realtime Updating Graph"),
                            daq.ToggleSwitch(id='graph_realtime_update', value=True),], style={'display': 'flex'}),
                            dcc.Checklist(options=['percentile_05', 'percentile_05_q'], id = "display_statistics", value = []), #value=['Montreal']
                            html.Div([
                            html.Label("normalize data"),
                            daq.ToggleSwitch(id='normalize_data', value=False),], style={'display': 'flex'}),
                                daq.NumericInput(id = "primary_min", label='primary min',labelPosition='bottom',value=" ",),
                                daq.NumericInput(id = "primary_max", label='primary max',labelPosition='bottom',value=" ",),
                                daq.NumericInput(id = "secondary_min", label='secondary min',labelPosition='bottom',value=" ",),
                                daq.NumericInput(id = "secondary_max", label='secondary max',labelPosition='bottom',value=" "),
                        ]),
                        dbc.ModalFooter(dbc.Button("close", id="close-graphing-options-button", className="ml-auto")),], id="graphing-options", size="xl",),   



                        
        ], style={'display': 'flex', 'flex-direction': 'row'}),
    ], style={'padding': '10px'})
], 
open=True,  # Start collapsed since it's settings
style={
    'border': '2px solid #FF8C42',
    'borderRadius': '5px',
    'margin': '5px 0',
    'flex': '20%', 
    'width': '20%'
}),
    ], style={'display': 'flex'}),   
   
## export   
   html.Div([
    html.H3('Upload/Export Data', style={'textAlign': 'center', 'marginBottom': '20px'}),
    html.Div([
        html.Button('upload_data', id='upload_data_button', n_clicks=0, style={'padding': '10px 20px', 'fontSize': '16px'}),
        html.Div(id='upload_data_children', style={'width': '5%', 'display': 'inline-block'}),

        html.Button('export_data', id='export_data_button', n_clicks=0, style={'padding': '10px 20px', 'fontSize': '16px', 'marginLeft': '10px'}),
        html.Div(id='export_data_children', style={'width': '5%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label('session toggle (when true upload by session)', style={'marginRight': '10px', 'fontSize': '18px'}),
            dcc.Checklist(
                id='session-toggle',
                options=[{'label': '', 'value': 'on'}],
                value=['on'],
                style={'display': 'inline-block'}
            )
        ], style={'display': 'inline-block', 'marginLeft': '20px'})
    ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'center', 'alignItems': 'center'})
], style={'backgroundColor':	'#d4be9f', 'padding': '20px', 'width': '100%'})

])




# Select file source
@app.callback(
    Output('Select_Data_Source_Output', 'children'),
    Input('Select_Data_Source', 'value'))
def update_output(Select_Data_Source):
    if Select_Data_Source is False: # File Import
        return 'File Import'
    if Select_Data_Source is True: # Database Query
        return 'Database Query'
    
# Select data level
@app.callback(
    Output('data_axis', 'value'),
    Input('Select_Data_Source', 'value'))
def update_output(Select_Data_Source):
    if Select_Data_Source is False: # File Import
        return 'data'
    if Select_Data_Source is True: # Database Query
        return 'corrected_data'

# display file upload
@app.callback(
    Output('datatable-upload', component_property='style'),
    Input('Select_Data_Source', 'value'))
def display_upload(Select_Data_Source):
    if Select_Data_Source is False:
        return {'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed',
                'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'}
        # return {'display': 'block'}
    if Select_Data_Source is True:
        return {'display': 'none'}

# display file structure
@app.callback(
    Output('File_Structure_Container', component_property='style'),
    Input('Select_Data_Source', 'value'))
def display_file_structure(Select_Data_Source):
    if Select_Data_Source is False:
        return {'display': 'block'}
        # return {'display': 'block'}
    if Select_Data_Source is True:   
        return {'display': 'none'}

# run pause
@app.callback(
    Output('realtime_update', 'value'),
    Input('Select_Data_Source', 'value'),
    prevent_initial_call=True)
def run_job(select_data_source):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if "Select_Data_Source" in changed_id:
        # I think True should be sql query but that does not seem to work, however I think the true and false are confusing
        if select_data_source == False:  # if file upload automatically correct
            return True  # ealtime Update off
        if select_data_source == True: # if sql pause correction so uploaded data can be shown

            return False
    else:
        dash.no_update




# ONSET
@app.callback(
    Output('HEADER_ROWS', 'value'),
    Output('FOOTER_ROWS', 'value'),
    Output('TIMESTAMP_COLUMN', 'value'),
    Output('DATA_COLUMN', 'value'),
    Input('File_Structure', 'value'),
    Input('parameter', 'children'))
def special_csv(file_structure, parameter):
    if parameter == "water_temperature":
        return list(config["file_structure_water_temperature"][file_structure].split(","))
    elif parameter == 'Conductivity' or parameter == 'conductivity':
        return list(config["file_structure_conducitvity"][file_structure].split(","))
    else:
        if file_structure == 'onset_U20':
            #return 2, 3, 1, 2
            return list(config["file_structure_discharge"][file_structure].split(","))
           # return config["file_structure_discharge"][file_structure]
        if file_structure == 'onset_U24':
            return 2, 3, 1, 4
        if file_structure == 'csv':
            return 1, 0, 0, 1

# CSV cutting one callback
@app.callback(
    Output('HEADER_ROWS', component_property='style'),
    Output('FOOTER_ROWS', component_property='style'),
    Output('TIMESTAMP_COLUMN', component_property='style'),
    Output('DATA_COLUMN', component_property='style'),
    Input('File_Structure', 'value'))
def display_csv_trimmer(File_Structure):
    if File_Structure == 'csv':
        return {'display': 'inline-block'}, {'display': 'inline-block'}, {'display': 'inline-block'}, {'display': 'inline-block'}
        # return {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}


# barometer selection
@app.callback(
          Output(component_id='Barometer_Button', component_property='style'),
          Output(component_id='available_barometers', component_property='style'),
          Output(component_id = 'Delete_Association',  component_property='style'),
          Input('Select_Data_Source', 'value'),
          Input('parameter', 'children'),
          Input('Barometer_Button', 'value'),
          )



def barometer_selection(data_source, parameter, barometer_button):
    if parameter:  # if a parameter has been selected
        if data_source is False: # data file upload
            if parameter == "LakeLevel" or parameter == "Piezometer" or parameter == "discharge" or parameter == "lake_level" or parameter == "water_level" or parameter == "groundwater_level":
                return {'display': 'block'}, {'display': 'block'}, {'display': 'block'},
            else:
                return {'display': 'none', 'border': 'none'}, {'display': 'none'}, {'display': 'none'},
        elif barometer_button == "No_Baro":
            return {'display': 'none', 'border': 'none'}, {'display': 'none'}, {'display': 'none'},
        
        else: # if data source is true aka sql query
            return {'display': 'none', 'border': 'none'}, {'display': 'none'}, {'display': 'none'}
    elif data_source is True:
            return {'display': 'none', 'border': 'none'}, {'display': 'none'}, {'display': 'none'},
    else:
        return dash.no_update 

# data interval
# pop up window    
@app.callback(
    Output("modal", "is_open"),
    [Input("open-modal-button", "n_clicks"), Input("close-modal-button", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output(component_id='data_interval', component_property='children'),
    Input(component_id='interval', component_property='value'),)
def data_interval(interval):
    labels = {0: '1', 1: '5', 2: '15', 3: '30', 4: '60', 5: '1440'}
    data_interval = labels.get(interval[0], "Unknown")
    return data_interval

### graphing options
# data interval
# pop up window    
@app.callback(
    Output("graphing-options", "is_open"),
    [Input("open-graphing-options-button", "n_clicks"), Input("close-graphing-options-button", "n_clicks")],
    [State("graphing-options", "is_open")],
)
def toggle_graphing_options(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# Get SQL Number from G_ID: site=name Site_Code site_sql_id = sql number G_ID
@app.callback(
    Output('site','children'),
    Output('site_sql_id','children'),
    Output('parameter','children'),
    Input('site_selector','value'))
def get_sql_number_from_gid(site_selector_value):
    if site_selector_value:
        site, parameter = site_selector_value.split(" ", 1) # takes the site_list value "71a water_level" and splits site and paramter
        from import_data import get_site_sql_id
        site_sql_id = get_site_sql_id(site)
        return site, site_sql_id, parameter
    else:
        return dash.no_update



# companion parameters
@app.callback(
    #Output(component_id='comparison_parameter', component_property='options'),
    Output('comparison_data', component_property='data'),
    Input(component_id='comparison_sites', component_property='value'),
    Input('query_start_date', 'data'),
    Input('query_end_date', 'data'),
    Input('primer_hours', 'value')
    )
def update_companion_parameters(comparison_sites, start_date, end_date, primer_hours):
    from import_data import get_site_sql_id, sql_import, usgs_data_import 
    if comparison_sites and start_date and end_date:
            start_date =  (pd.to_datetime(start_date).to_pydatetime()) - timedelta(hours=primer_hours)
            end_date =  (pd.to_datetime(end_date).to_pydatetime()) + timedelta(hours=primer_hours)
            # convert string to list
            #comparison_sites = comparison_sites.split()
            comparison_data = pd.DataFrame(columns=["datetime", "site", "parameter", "corrected_data"])
            for site_item in comparison_sites:
                if site_item.startswith("USGS"): # usgs sites
                    # Split off the last word as parameter, rest as site
                    usgs_site, usgs_parameter = site_item.rsplit(" ", 1)
                    usgs_site_codes = pd.read_csv("external_comaprison_sites.csv", skipinitialspace=True)
                    usgs_site_code = usgs_site_codes.loc[usgs_site_codes["site"] == usgs_site, "site_sql_id"].item()
                    df = usgs_data_import(usgs_site, usgs_parameter, start_date, end_date)
                else: # kc sites
                    site, parameter = site_item.split(" ", 1)
                    site_sql_id = get_site_sql_id(site)
                    df = sql_import(parameter, site_sql_id, start_date, end_date)
                    if parameter == "discharge":
                        
                        df["site"] = site
                        df_q = df.copy()
                        df_q["parameter"] = "discharge"
                        df_q = df_q[["site", "datetime", "parameter", "discharge"]]
                        df_q.rename(columns={'discharge': 'corrected_data'}, inplace=True)
                        ## add stage
                        #df["parameter"] = "stage"
                        #df = df[["site", "datetime", "parameter", "corrected_data"]]
                        
                        #df = pd.concat([df, df_q])
                        df = df_q
                    else:   
                        df["site"] = site
                        df["parameter"] = parameter
                    desired_order = ["site", "parameter", "datetime", "corrected_data"]
                    existing_columns = [col for col in desired_order if col in df.columns] 
                            # Filter out columns that exist in the DataFrame   
                    df = df[existing_columns].copy()
                comparison_data = pd.concat([comparison_data, df], ignore_index=True)
            
            return comparison_data.to_json(orient="split")
    else:
         return dash.no_update
    

# existing_data
### im not sure what this does
@app.callback(
    #Output(component_id='comparison_parameter', component_property='options'),
    Output(component_id='comparison_sites', component_property='value'),
    Input(component_id='site_selector', component_property='value'),
    State('query_start_date', 'data'),
    State('query_end_date', 'data'),
    )
def add_existing_data(site_selector_value, start_date, end_date):

    #'comparison_sites' when a site is selected/changed this clears comparison sites and adds the current site to the comparison for comparison
 
    if site_selector_value:
            #return None # sets the site to be a comparison site to see existing data
            return None
            # return None # returns nothing
    else:
         return dash.no_update


# Ratings
@app.callback(
    # Output('Ratings', 'value'),
    Output('Ratings', 'options'),
    Output('Ratings', 'style'),
    Input('parameter', 'children'),
    Input('site_sql_id', 'children'))
def Select_Ratings(parameter, site_sql_id):
   #print("select ratings")
    if parameter == "FlowLevel" or parameter == "discharge":
        with sql_engine.begin() as conn:
            ratings = pd.read_sql_query(f"select RatingNumber as rating_number from tblFlowRatings WHERE G_ID = '{site_sql_id}' GROUP BY RatingNumber ORDER BY RatingNumber DESC;", conn)
        ratings = ratings['rating_number'].values.tolist()
    
        return [{'label': i, 'value': i} for i in ratings], {'display': 'block'}
    else:
        return [{'label': "NONE", 'value': "NONE"}], {'display': 'none'}
    
# rating corrections 

# Show/hide the container based on parameter value
@app.callback(
    Output('offset_container', 'style'),
    Input('parameter', 'children'))

def toggle_offset_visibility(parameter):
    if parameter == 'discharge':
        return {'display': 'flex', 'alignItems': 'center', 'gap': '0.5rem'}
    else:
        return {'display': 'none'}


# Update the label text based on toggle value
@app.callback(
    Output('offset_label', 'children'),
    Input('apply_discharge_offset', 'value')
)
def update_label(toggle_value):
    if toggle_value:
        return "Apply individual discharge offsets"
    else:
        return "Do not apply individual discharge offsets"

# all observations
@app.callback(
    Output('observations', 'data'),
    Input('site', 'children'),
    Input("site_sql_id", "children"),
    Input('parameter', 'children'),
)

def all_observations(site, site_sql_id, parameter):
    from import_data import get_observations_join
    if site and parameter:
        try:
            observations = get_observations_join(parameter, site_sql_id, "*", "*") # convert start/end date from utc to pdt
        
            if parameter == "FlowLevel" or parameter == "discharge":
                        
                        observations.rename(columns={"parameter_observation": "q_observation"}, inplace=True)
            # at the moment the date range is updated based on observations so you dont want to trigger observations to update if its changed
            return  observations.to_json(orient="split")
        except:
            return pd.DataFrame().to_json(orient="split")
    else:
        #return pd.DataFrame().to_json(orient="split")
        return dash.no_update
    
@app.callback(
    Output('statistics', 'data'),
    Input("site_sql_id", "children"),
    Input('parameter', 'children'),
)  
def get_statistics(site_sql_id, parameter):     
    if not site_sql_id or not parameter:
        return None
    
    statistics = sql_statistics(parameter, site_sql_id)
    stats_df = pd.DataFrame({
        'datetime': [
            statistics["first_datetime"], 
            statistics["last_datetime"], 
            datetime.now().replace(second=0, microsecond=0)
        ],
        'observation_number': [
            'first record', 
            'last record', 
            "today (just todays date)"
        ]
    })
    stats_df['datetime'] = pd.to_datetime(stats_df['datetime']).dt.tz_localize(None)

    # Return as JSON string in the correct format
    return stats_df.to_json(orient='split', date_format='iso')


### Date range
@app.callback(
    Output('select_datetime_by_obs_a', 'options'),
    Output('select_datetime_by_obs_b', 'options'),
    Output('query_start_date', 'data'),
    Output('query_end_date', 'data'),
    Input("observations", "data"),
    Input('startDate', 'value'),
    Input('endDate', 'value'),
    State('site', 'children'),
    Input("site_sql_id", "children"),
    Input('parameter', 'children'),
    State('Select_Data_Source', 'value'),
    Input('select_datetime_by_obs_a', 'value'),
    Input('select_datetime_by_obs_b', 'value'),
    State('data_interval', 'children'),
    Input('statistics', 'data'),
)
def daterange(observations, startDate, endDate, site, site_sql_id, parameter, 
              data_source, obs_a, obs_b, data_interval, stats_json):
    if not site_sql_id or not parameter or not stats_json:
        return dash.no_update
    elif parameter == "barometer": # barometer is funny for some reason so we are bypassing the stats df for now
        if startDate is not None and endDate is not None: # if a start and end date then send those
            
            query_start_date = startDate
            query_end_date = endDate
            
            return dash.no_update, dash.no_update, query_start_date, query_end_date
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    else:
        # create obs for dropdowns 
        query_start_date = ""
        query_end_date = ""
        
        # Load statistics dataframe
        stats_df = pd.read_json(stats_json, orient="split")
   
        stats_df['datetime'] = pd.to_datetime(stats_df['datetime']).dt.tz_localize(None)
        
        # Read observations
        obs = pd.read_json(observations, orient="split")
        obs['datetime'] = pd.to_datetime(obs['datetime'])  # Ensure datetime column is datetime type
        # Combine observations with statistics
        obs = pd.concat([obs, stats_df], ignore_index=True)
        obs = obs.sort_values(by='datetime', ascending=False)
        
        # Generate observation options (shared logic for both a and b)
        obs_options = [
            f"date: {row['datetime'].strftime('%Y-%m-%d %H:%M')} observation: {row['observation_number']}"
            for _, row in obs.iterrows()
            if pd.notna(row['datetime'])
        ]
        
        # Extract unique years
        obs_years = obs['datetime'].dropna().dt.year.unique().tolist()
        
        # Ensure current year and next year are included
        current_year = datetime.now().year
        if current_year not in obs_years:
            obs_years.append(current_year)
        
        future_year = (datetime.now() + timedelta(days=91)).year
        if future_year not in obs_years:
            obs_years.append(future_year)
        
        # Add year options
        year_options = [f"water year: {year}" for year in sorted(obs_years)]
        full_options = year_options + obs_options
        
        if startDate is not None:
            query_start_date = startDate
        else:
            # Process obs_a (start date)
            if obs_a:
                if str(obs_a).lower().startswith("water year"):
                    year = int(str(obs_a).lower().replace("water year:", "").strip())
                    query_start_date = datetime(year-1, 10, 1, 0, 0).strftime('%Y-%m-%d %H:%M')
                else:
                    obs_a_str = obs_a.split("date: ")[1].split(" observation:")[0].strip()
                    obs_a_dt = datetime.strptime(obs_a_str, '%Y-%m-%d %H:%M')
                    query_start_date = pd.to_datetime(obs_a_dt).to_pydatetime()
                    
                    from import_data import sql_get_closest_datetime
                    try:
                        result_a, result_b, result_c = sql_get_closest_datetime(parameter, site_sql_id, query_start_date)
                        query_start_date = result_a
                    except:
                        pass
        if endDate is not None:
            query_end_date = endDate
        else:
            # Process obs_b (end date)
            if obs_b:
                if str(obs_b).lower().startswith("water year"):
                    year = int(str(obs_b).lower().replace("water year:", "").strip())
                    query_end_date = datetime(year, 10, 1, 0, 0).strftime('%Y-%m-%d %H:%M')
                else:
                    obs_b_str = obs_b.split("date: ")[1].split(" observation:")[0].strip()
                    obs_b_dt = datetime.strptime(obs_b_str, '%Y-%m-%d %H:%M')
                    query_end_date = pd.to_datetime(obs_b_dt).to_pydatetime()
                    
                    try:
                        result_a, result_b, result_c = sql_get_closest_datetime(parameter, site_sql_id, query_end_date)
                        
                        if result_a == result_b and result_b == result_c:
                            query_end_date = pd.Timestamp(query_end_date).floor(f'{data_interval}T') + pd.Timedelta(f"{data_interval}T")
                        elif result_a != result_b and result_b != result_c:
                            query_end_date = result_a
                        elif result_a == result_b and result_a < result_c:
                            query_end_date = result_c
                        else:
                            query_end_date = result_b
                    except:
                        pass

    
    
    return (
        full_options,  # obs_a_options
        full_options,  # obs_b_options
        query_start_date, 
        query_end_date
    )
# Pick Range, query existing data in SQL Database
@app.callback(
    # Output('output-container-date-picker-range', 'children'),
    Output('import_data', 'data'),
    
    Input('query_start_date', 'data'), # Using as Input
    Input('query_end_date', 'data'), # Using as Input
    Input('site', 'children'),
    Input("site_sql_id", "children"),
    Input('parameter', 'children'),
    Input('datatable-upload', 'contents'),
    Input('datatable-upload', 'filename'),
    State('HEADER_ROWS', 'value'),
    State('FOOTER_ROWS', 'value'),
    State('TIMESTAMP_COLUMN', 'value'),
    Input('DATA_COLUMN', 'value'),
    Input('Select_Data_Source', 'value'),
    Input('Barometer_Button', 'value'),
    Input('available_barometers', 'value'),
    # Input(component_id='Available_Barometers', component_property='value'),
    # State('Barometer_Button', 'value')
)
def update_daterange(query_start_date, query_end_date, site, site_sql_id, parameter, contents, filename, header_rows, footer_rows, timestamp_column, data_column, data_source, barometer_button, available_barometers):
    # Call and process incoming data
    # if there is no csv file (northing in contents) query data from sql server
    # contents = holder of imported data, when data is being imported contents = True
    # data_source is if we are quering the server vs importing data, select_data_source False is file import True is SQL query
    # program corrects off the "data" column but other values are pulled
    # if contents is None:  # nothin in datatable upload
    ### pretty good at not sending back empty things, if no updates returns blank df, clears import data
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    from import_data import sql_import
    
    # no matter what these need to be exist
    if site and parameter:
        # sql query
        if data_source == True and query_start_date != '' and query_end_date != '' and ("query_start_date" in changed_id or "query_end_date" in changed_id or "query_start_date" in changed_id or "query_end_date" in changed_id): #and 'select_range' in changed_id:  # query sql server
            df = sql_import(parameter, site_sql_id, query_start_date, query_end_date) #convert start/end date from utc to pdt
            
            return  df.to_json(orient="split")

        elif data_source == False and contents is not None:  # if there is a file #if data_source == False:  # file upload and run get field observations after data import for better error control
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            # Assume that the user uploaded a CSV file
            # this assumes the file name ends in 'csv' or 'xls'
            def dateparse (time):    
                return pd.to_datetime( time, format='%Y%m%d %H:M:S', errors='ignore')
            # Assume that the user uploaded an excel file
            if 'xls' in filename or 'xlsx' in filename:
                df = pd.read_excel(decoded, usecols=[int(timestamp_column), int(
                        data_column)], skiprows=int(header_rows), skipfooter=int(footer_rows), names=['datetime', 'data'], parse_dates=[0], date_parser=dateparse, engine='python')
            else: # if its a a csv or they didnt specift
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), usecols=[int(
                        timestamp_column), int(data_column)], skiprows=int(header_rows), skipfooter=int(footer_rows), names=['datetime', 'data'], parse_dates=[0], date_parser=dateparse, engine='python')
                
            # adjust baro button for parameteres
            if parameter == "water_temperature":
                barometer_button = "no_baro"
            else:
                barometer_button = barometer_button
            ### calculate barometer
            if barometer_button == "Baro": #and data_source is False:  # data source false is file import
                from import_data import get_site_sql_id
                barometer_site = available_barometers
                barometer_sql_id = get_site_sql_id(barometer_site)
                # THIS IS DUMB, ITS A PLACHHOULDER there needs to be a formula to convert wl feet
                if df['data'].mean() < 50: # if under 30 assumed to be psi
                    df['data'] = round((df['data']*68.9476), 3) # convert to millibar
                elif df["data"].mean() > 50 and df["data"].mean() < 850: # assume kpa
                    df["data"] = round(df["data"] * 10, 3)
                elif df['data'].mean() > 850: # assumed to be millibar
                    df['data'] = df['data'] # millibar

                barometer_query = sql_import("barometer", barometer_sql_id, df['datetime'].min(), df['datetime'].max()) # fx converts to PST and out of PST
                barometer_query = barometer_query.rename(columns = {"corrected_data": "barometer_data"})
                # resample barometer to 5 minutes
                barometer_query = barometer_query.set_index("datetime").resample('5T').interpolate(method='linear').reset_index(level=None, drop=False)
                 
                df = pd.merge(df,barometer_query[['datetime', "barometer_data"]],on=['datetime'])
                  
                df['data'] = ((df['data']-df["barometer_data"]) * 0.0335).round(3) # convert to water level feat
                output = df.drop(['barometer_data'], axis=1)
                
                return  output.to_json(orient="split")#, df['datetime'].min() + timedelta(hours=(7)), df['datetime'].max() + timedelta(hours=(7))
            elif barometer_button != "Baro":
                if parameter == "water_temperature" and df["data"].mean() > 30:  # f
                    df["data"] = round((df["data"] - 32) * (5/9), 2)
                return df.to_json(orient="split")
                #return dash.no_update
            else:
                return pd.DataFrame().to_json(orient="split")
        else:
            
            return pd.DataFrame().to_json(orient="split")
            #return dash.no_update
    else:
         return pd.DataFrame().to_json(orient="split")


# show, amd add observations in observations datatable
@app.callback(
    Output('observations_datatable', 'data'),
    Output('observations_datatable', 'columns'),  
    Input("observations", "data"), 
    Input('add_row_button_a', 'n_clicks'),
    Input('refresh_button_a', 'n_clicks'),
    Input('show_inrange_observations', 'value'),
    State('parameter', 'children'),
    State("site_sql_id", "children"),
    State('observations_datatable', 'data'),
    State('observations_datatable', 'columns'),
    Input('query_start_date', 'data'),  # obs a
    Input('query_end_date', 'data'), # obs b),
    )
def observations_dattable(observations, add_row_button_a, refresh_button_a, inrange_observations, parameter, site_sql_id, rows, columns, query_start_date, query_end_date):

    observations = pd.read_json(observations, orient="split")
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0] 
    #ctx = callback_context
    #if ctx.triggered: 
    if not observations.empty and parameter != "0":
    #if parameter != "0": # get obs
        if "add_row_button_a.n_clicks" in changed_id:
            rows.append({c['id']: '' for c in columns})  # Add a new blank row
            return rows, columns#, [{"name": i, "id": i} for i in observations.columns]
        else:
            if "refresh_button_a.n-clicks" in changed_id:
                from import_data import get_observations_join
                observations = get_observations_join(parameter, site_sql_id, "*", "*") # convert start/end date from utc to pdt     
            if parameter == "FlowLevel" or parameter == "discharge" and "parameter_observation" in observations.columns:
                                observations.rename(columns={"parameter_observation": "q_observation"}, inplace=True)
            if (parameter == "Conductivity" or parameter == "conductivity" or parameter == "water_temperature" )and "observation_stage" in observations.columns:
                                    observations.drop(columns=["observation_stage"], inplace=True)
                
            
                    
            #observations["datetime"] = observations["datetime"].dt.strftime('%Y-%m-%d %H:%M') # allows dash graph to display datetime without the "Td"

            if inrange_observations is True and query_start_date and query_end_date:
               
                query_start_date =  pd.to_datetime(query_start_date).to_pydatetime()
                query_end_date =  pd.to_datetime(query_end_date).to_pydatetime()
                observations = observations.loc[(observations["datetime"] >= query_start_date) & (observations["datetime"] <= query_end_date)]
            observations["datetime"] = observations["datetime"].dt.strftime('%Y-%m-%d %H:%M') # allows dash graph to display datetime without the "Td"
            """def safe_format_datetime(value):
                try:
                    return value.strftime('%Y-%m-%d %H:%M')
                except AttributeError:  # This occurs if the value is not a valid datetime
                    #return value  # or you could return the original value, e.g., `value`
                    return "2000-01-01 12:00" # proper error handling will have cascading errors 
            observations["datetime"] = observations["datetime"].apply(safe_format_datetime)"""

            return observations.to_dict('records'), [{"name": i, "id": i} for i in observations.columns]
       

    else:
        return dash.no_update


@app.callback(
    Output("corrected_data_datatable", "data"),
    Output("corrected_data_datatable", "columns"),
    Input('import_data', 'data'),
    Input('observations_datatable', 'data'),
    Input('comparison_data', 'data'),
    State('parameter', 'children'),
    State('site', 'children'),
    State("site_sql_id", "children"),
    State('corrected_data_datatable', 'data'),
    Input("realtime_update", "value"),
    Input("run_job", "n_clicks"),
    State('Ratings', 'value'),
    Input('basic_forward_fill', 'n_clicks'),
    Input('basic_backward_fill', 'n_clicks'),
    State("fill_limit", 'value'),
    State("fill_limit_number", 'value'),
    State("fill_limit_area", "value"),
    State("set_method", 'value'),
    State('set_limit', 'value'),
    State('limit_number', 'value'),
    State('limit_direction', 'value'),
    State('limit_area', 'value'),
    Input('fill_missing_data', 'n_clicks'),
    State('data_interval', 'children'),
    Input('resample_button', 'n_clicks'),
    State('query_start_date', 'data'),
    State('query_end_date', 'data'),
    Input("to_observations_button", "n_clicks"),
    State('Select_Data_Source', 'value'),
    Input("data_axis", "value"),
    Input('apply_discharge_offset', 'value'),
    Input('set-dry-indicator', 'n_clicks'),
    Input('apply-dry-threshold-raw', 'n_clicks'),
    Input('apply-dry-threshold-corrected-data', 'n_clicks'),
    Input('clear-dry-indicator', 'n_clicks'),
    State('start-dry-indicator-range', 'value'),
    State('end-dry-indicator-range', 'value'),
    State('dry-threshold-raw-input', 'value'),
    State('dry-threshold-corrected-data-input', 'value'),
    State('interp-data', "value"),
    State('interp-corrected-data', "value"),
)
def correct_data(import_data, obs_rows, comparison_data, parameter, site, site_sql_id, rows, realtime_update, run_job, rating_number, basic_forward_fill, basic_backward_fill, fill_limit, fill_limit_number, fill_area, method, set_limit, limit_number, limit_direction, limit_area, fill_missing_data, data_interval, resample, query_start_date, query_end_date, to_observations_button, data_source, data_level, apply_discharge_offset, set_dry_indicator, apply_dry_threshold_raw, apply_dry_threshold_corrected_data, clear_dry_indicator, start_dry_indicator_range, end_dry_indicator_range, dry_threshold_raw_input, dry_threshold_corrected_data_input, interp_data, interp_corrected_data):
    from data_cleaning import reformat_data, initial_column_managment, column_managment, fill_timeseries
    from discharge import discharge_calculation
    from dash import ctx
    
    try:
        observation = config[parameter]["observation_class"]
    except KeyError:
        observation = ""
    

    changed_id = ctx.triggered_id # just looks at "update_button"
    #changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0] # requires "update_button.n_clicks"
    
   
    
    try:
        observation = config[parameter]["observation_class"]
    except KeyError:
        observation = ""
    # Get the triggered property from Dash callback context
    #changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0] 
   
   
    

    if not pd.read_json(import_data, orient="split").empty and not "datetime" in pd.DataFrame(rows).columns: # and import_data is not None:
          
            data = pd.read_json(import_data, orient="split")
    elif "datetime" in pd.DataFrame(rows).columns:
          
            data = pd.DataFrame(rows)
    else:
         
            return dash.no_update

    

    data = reformat_data(data)
    
    data = initial_column_managment(data)
    
    data = fill_timeseries(data, data_interval)
    
    # format observations if present
    if obs_rows:
        observations = pd.DataFrame(obs_rows)
       
        observations = reformat_data(observations)
 
        observations = observations.dropna(subset=['datetime'])
        #print("data_interval", data_interval)
        data = pd.merge_asof(data.sort_values('datetime'), observations.sort_values('datetime'), on='datetime', tolerance=pd.Timedelta(f"{int(data_interval)/2}m"), direction="nearest")
       
    data = reformat_data(data)
    
    # add existing data
    if comparison_data:
        try:
            comparison_data = pd.read_json(comparison_data, orient="split")
            #if not comparison_data.empty:
            #    comparison_data = reformat_data(comparison_data)
            #    c_data = comparison_data.loc[(comparison_data["site"] == site) & (comparison_data["parameter"] == parameter)]
            #    c_data = c_data[["datetime", "corrected_data"]]
            #    if not c_data.empty:
            #        c_data = c_data.rename(columns={"datetime": "datetime", "corrected_data": f"c_{parameter}"})
            #    if parameter == "discharge":
            #        wl_data = comparison_data.loc[(comparison_data["site"] == site) & (comparison_data["parameter"] == "stage")]
            #        wl_data = wl_data[["datetime", "corrected_data"]]
            #        if not wl_data.empty:
            #            wl_data = wl_data.rename(columns={"datetime": "datetime", "corrected_data": f"c_stage"})
            #            c_data = wl_data.merge(c_data, left_on="datetime", right_on="datetime", how="outer")
            #    data = data.merge(c_data, left_on="datetime", right_on="datetime", how="outer")
        except:
            pass
    
    if "c_discharge" in data.columns and comparison_data is None:
        data.drop("c_discharge", axis=1, inplace=True)
  
    # Realtime update or run job logic
    if realtime_update is True or changed_id == 'run_job':
        if changed_id == "to_observations_button":
            from data_cleaning import to_observations
            data = to_observations(data, query_start_date, query_end_date)
        if changed_id == "basic_interpolate_missing_data":
            from interpolation import basic_interpolation
            data = basic_interpolation(data, method, set_limit, limit_number, limit_direction, limit_area, interp_data, interp_corrected_data)
        if changed_id == "basic_backward_fill":
            from interpolation import run_basic_backward_fill
            data = run_basic_backward_fill(data, fill_limit, fill_limit_number, fill_area, interp_data, interp_corrected_data)
        if changed_id == "basic_forward_fill":
            from interpolation import run_basic_forward_fill
            data = run_basic_forward_fill(data, fill_limit, fill_limit_number, fill_area, interp_data, interp_corrected_data)
        if changed_id == "resample_button" or changed_id == "resample_button.n_clicks":
            from interpolation import resample_data
          
            data = resample_data(data, data_interval)
        
        data = parameter_calculation(data, data_level)
        
        
    # Additional logic for specific parameters
    if (parameter == "discharge" or parameter == "FlowLevel") and rating_number != "NONE":   
        data = discharge_calculation(data, rating_number, site_sql_id, apply_discharge_offset)
    
    if changed_id == "basic_interpolate_missing_data":
        from interpolation import basic_interpolation
        data = basic_interpolation(data, method, set_limit, limit_number, limit_direction, limit_area, interp_data, interp_corrected_data)
    elif changed_id == "basic_backward_fill":
        from interpolation import run_basic_backward_fill
        data = run_basic_backward_fill(data, fill_limit, fill_limit_number, fill_area, interp_data, interp_corrected_data)
    elif changed_id == "basic_forward_fill":
        from interpolation import run_basic_forward_fill
        data = run_basic_forward_fill(data, fill_limit, fill_limit_number, fill_area, interp_data, interp_corrected_data)
    elif changed_id == "resample_button":
        from interpolation import resample_data
        data = resample_data(data, data_interval)
    elif changed_id == 'set-dry-indicator':
        from data_cleaning import set_dry_indicator_function
        data = set_dry_indicator_function(data, start_dry_indicator_range, end_dry_indicator_range)
    elif changed_id == 'apply-dry-threshold-raw':
        from data_cleaning import apply_dry_threshold_raw_function
        data = apply_dry_threshold_raw_function(data, dry_threshold_raw_input)
    elif changed_id == 'apply-dry-threshold-corrected-data':
        from data_cleaning import apply_dry_threshold_corrected_data_function
        data = apply_dry_threshold_corrected_data_function(data, dry_threshold_corrected_data_input)
    elif changed_id == 'clear-dry-indicator':
        from data_cleaning import clear_dry_indicator_function
        data = clear_dry_indicator_function(data, start_dry_indicator_range, end_dry_indicator_range)
        #mask = (data['datetime'] >= start_dry_indicator_range) & (data['datetime'] <= end_dry_indicator_range)
        #data.loc[mask, 'warning'] = 0
        #data.loc[mask, 'corrected_data'] = np.nan

    data = column_managment(data)
   
    # Reformat the datetime for Dash graph display
    data["datetime"] = data["datetime"].dt.strftime('%Y-%m-%d %H:%M')
    # Convert to records
    data_records = data.to_dict('records')
    columns = [{"name": i, "id": i} for i in data.columns]
    
    return data_records, columns
  



#selected_points = selectedData['points']
    
## Get x and y values from selected points
#x_values = [point['x'] for point in selected_points]
#y_values = [point['y'] for point in selected_points]
    
## If you have custom data stored in the trace
#if 'customdata' in selected_points[0]:
#custom_values = [point['customdata'] for point in selected_points]
    
# Example: return as a dataframe
    
#df_selected = pd.DataFrame({
#'x': x_values,
#        'y': y_values
#    })
#    print("selected points")
#    print(selected_points)
#    return selected_points
### clear selected data
#@app.callback(
#    Output('graph', 'selectedData'),
#    Input('clear-selection-button', 'n_clicks'),
#    prevent_initial_call=True
#)
#def clear_selection(n_clicks):
#    return None

@app.callback(
        #Output(component_id='graph_output', component_property='children'),
        Output('graph', "figure"),
        #Output('graph', 'selectedData'),
        #Output('selected-data-store', 'data'),  # or whatever output you need
        #Input('graph', 'selectedData'),
        Input('graph_realtime_update', 'value'),
        Input("corrected_data_datatable", "data"),
        State('site_selector', 'value'), # only needed to graph existing data differenty
        State('site', 'children'),
        State('site_sql_id', 'children'),
        State('parameter', 'children'),
        Input('comparison_data', 'data'),
        State('Ratings', 'value'),
        Input("primary_min", "value"),
        Input("primary_max", "value"),
        Input("secondary_min", "value"),
        Input("secondary_max", "value"),
        Input("normalize_data", "value"),
        State("statistics", "data"),
        State("display_statistics", "value"),
        State('query_start_date', 'data'),  # obs a
        State('query_end_date', 'data'), # obs b),
)
def graph(graph_realtime_update, df, site_selector_value, site, site_sql_id, parameter, comparison_data, rating, primary_min, primary_max, secondary_min, secondary_max, normalize_data, statistics, display_statistics, query_start_date, query_end_date):
    """when its all said and done this is pretty inefficient; it started this way because i was exporting the plotly graph, now i export a seperate matplotlib graph """
    
    from data_cleaning import reformat_data, parameter_calculation
    from graph_2 import cache_graph_export
    df = pd.DataFrame(df)
    comparison_data = pd.read_json(comparison_data, orient = "split")
    #selected_data = selectedData
    if not df.empty and graph_realtime_update is True:
        #print("selected data")
        #print(selectedData)
        df = reformat_data(df) 
        if query_start_date != "" and query_end_date != "": # really we are trying to get rid of comparison data
                df = df.loc[(df["datetime"] >= query_start_date) & (df["datetime"] <= query_end_date)].copy()
        if query_start_date == "" or not query_start_date:
            query_start_date = df['datetime'].min()
        if query_end_date == "" or not query_end_date:
            query_end_date = df['datetime'].max()
        #fig = html.Div(dcc.Graph(figure = go.Figure()), style = {'width': '100%', 'display': 'inline-block'})
        fig = cache_graph_export(df, site_sql_id, site_selector_value, site, parameter, comparison_data, primary_min, primary_max, secondary_min, secondary_max, normalize_data, statistics, display_statistics)
        #fig.update_layout(dragmode='lasso', hovermode='closest',) # allows for selection
        
        return fig #selected_points
    else:
            return dash.no_update

@app.callback(
        #Output('Corrected_Data', 'filter_query'),
        #Output('Corrected_Data', 'selected_rows'),
        Output('corrected_data_datatable', 'active_cell'),
        Input('graph', 'relayoutData'),
        Input('graph', 'clickData')
)
def update_graph_range(graph_range, graph_point):
    # filter query 
    #if graph_range and 'xaxis.range[0]' in graph_range:
    #    graph_range_start_date = graph_range['xaxis.range[0]']
    #    graph_range_end_date = graph_range['xaxis.range[1]']
    #    filter_query = r"{datetime} <= "+graph_range_start_date+r" && {datetime} >= "+graph_range_end_date
    #    print("filter query: ",filter_query)
       
    #else:
    #    filter_query = ''
    
    if graph_point:
        
        point_index = graph_point['points'][0]['pointIndex']
        point_datetime = graph_point['points'][0]['x']
        active_cell = {'row': point_index, 'column': 0}
    
        #return "", [point_index], active_cell
        return active_cell
    else:
        return dash.no_update

    

@app.callback(
    dash.dependencies.Output('upload_data_children', 'children'),
    [dash.dependencies.Input('upload_data_button', 'n_clicks'),
    dash.dependencies.Input('export_data_button', 'n_clicks')],
    State('corrected_data_datatable', 'data'),
    State('site_selector', 'value'),
    State('site', 'children'),
    State('site_sql_id', 'children'),
    State('parameter', 'children'),
    State('comparison_data', 'data'),
    State('Ratings', 'value'),
    State("primary_min", "value"),
    State("primary_max", "value"),
    State("secondary_min", "value"),
    State("secondary_max", "value"),
    State("normalize_data", "value"),
    State("statistics", "data"),
    State("display_statistics", "value"),
    State('query_start_date', 'data'),
    State('query_end_date', 'data'),
    State('session-toggle', 'value')
)
def run_data_action(upload_clicks, export_clicks, df, site_selector_value, site, site_sql_id, parameter,
                    comparison_data, rating, primary_min, primary_max, secondary_min, secondary_max,
                    normalize_data, statistics, display_statistics, query_start_date, query_end_date, session_toggle):
    
    from data_cleaning import reformat_data
    from graph_2 import plot_for_save
    import pandas as pd
    import dash
    from workup_notes import workup_notes_main

    triggered = dash.callback_context.triggered
    trigger_id = triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id in ["upload_data_button", "export_data_button"]:
        df = pd.DataFrame(df)
        if "existing_data" in df.columns:
            df = df[df["existing_data"].isnull()].copy()
            df = df.dropna(subset=["data"])
        if df.empty or df.shape[1] < 1:  # check if df is valid
            return dash.no_update
        else:
          
            df = reformat_data(df)
            # Apply date filter
            if query_start_date and query_end_date:
                    df = df[(df["datetime"] >= query_start_date) & (df["datetime"] <= query_end_date)].copy()
            
            start_date = df["datetime"].min()
            end_date = df["datetime"].max()

            # Standardize parameter naming
            param_map = {
                    "Conductivity": "Conductivity",
                    "conductivity": "Conductivity",
                    "water_level": "water_level",
                    "LakeLevel": "water_level",
                    "groundwater_level": "groundwater_level",
                    "Piezometer": "groundwater_level",
                    "discharge": "discharge",
                    "FlowLevel": "discharge"
            }
            parameter = param_map.get(parameter, parameter)
            comparison_data = pd.read_json(comparison_data, orient="split")
                
            normalize_data = False  # for upload plot standard data
                
            # Fix session toggle check - handle Dash checklist format
            session_enabled = session_toggle and 'on' in session_toggle if isinstance(session_toggle, list) else bool(session_toggle)
                
            if session_enabled:  # export graph and workup notes by session
                # define obs
                if 'q_observation' in df.columns:
                        obs = 'q_observation'
                elif "field_observations" in df.columns:
                        obs = "field_observations"
                elif "observations" in df.columns:
                        obs = "observations"
                elif "observation" in df.columns:
                        obs = "observation"
                elif "observation_stage" in df.columns:
                        obs = "observation_stage"
                elif "observation_stage" not in df.columns and "parameter_observation" in df.columns:
                        obs = "parameter_observation"
                else:
                        obs = None

                if obs and obs in df.columns:
                    obs_dates = df[df[obs].notna()]['datetime'].sort_values()

                    # Create graphs for each period between observations
                    for i in range(len(obs_dates) - 1):
                        start_date = obs_dates.iloc[i]
                        end_date = obs_dates.iloc[i + 1]
                                
                        # Filter data for this period
                        period_df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].copy()
                                
                        if len(period_df) > 0:  # session upload
                            plot_for_save(period_df, site_selector_value, site_sql_id, site, parameter, comparison_data, 
                                            primary_min, primary_max, secondary_min, secondary_max, normalize_data, statistics, display_statistics)
                            workup_notes_main(period_df, parameter, site_sql_id, site)
                            desired_order = ["datetime", "data", "corrected_data", "discharge", "estimate", "warning", "non_detect"]
                            df_export = period_df[[col for col in desired_order if col in period_df.columns]].copy()
                            export_path = f"W:/STS/hydro/GAUGE/Temp/Ian's Temp/{site}_{parameter}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
                            df_export.to_csv(export_path, index=False)
                else:
                    # Fallback if no observation column found
                    session_enabled = False
                        
            if not session_enabled:  # no session upload
                plot_for_save(df, site_selector_value, site_sql_id, site, parameter, comparison_data, 
                                primary_min, primary_max, secondary_min, secondary_max, normalize_data, statistics, display_statistics)
                
                desired_order = ["datetime", "data", "corrected_data", "discharge", "estimate", "warning", "non_detect"]
                df_export = df[[col for col in desired_order if col in df.columns]].copy()
                export_path = f"W:/STS/hydro/GAUGE/Temp/Ian's Temp/{site}_{parameter}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv"
                df_export.to_csv(export_path, index=False)
                
            # sql upload
            if trigger_id == "upload_data_button":
                from sql_upload import full_upload
                workup_notes_main(df, parameter, site_sql_id, site)  # Fixed: use df instead of undefined period_df    
                # Use the same df_export logic as above
                desired_order = ["datetime", "data", "corrected_data", "discharge", "estimate", "warning", "non_detect"]
                df_final = df[[col for col in desired_order if col in df.columns]].copy()

                if parameter not in ["water_level", "groundwater_level"] and "non_detect" in df_final.columns:  # drop non_detect if there isnt a non detect column in db yet
                    df_final = df_final.drop(columns=["non_detect"])
                    
                full_upload(df_final, parameter, site_sql_id, 7)
                print("Upload complete")
                return "  uploaded"
            else:
                return "exported"

    
    else:
        return dash.no_update




# You could also return a 404 "URL not found" page here
if __name__ == '__main__':
    app.run_server(port="8050",host='127.0.0.1',debug=True)
    # Instead of 0.0.0.0, bind to VPN IP
    #app.run_server(
    #    host='10.4.12.49',  # Your VM's VPN IP
    #    port=8050
    #)
    
    #app.run_server(host='0.0.0.0',port='8050')  
    # ethernet adapter ipv4 address 10.219.226.110
    #app.run_server(host='10.219.226.110',port='8050')  
    # ethernet adapter default gateway 10.219.224.1 # doesnt work
    #app.run_server(host='10.219.224.1',port='8050') # doesnt work
    # wireless lan adapter ipv4 address 192.168.0.193
   # app.run_server(host='192.168.0.193',port='8050') # launches but cannot access
    # default gateway  192.168.0.1
    #app.run_server(host='192.168.0.1',port='8050') # doesnt work
    # this computers actual ip address will be someting in the 192 range.
    # launch app with 0.0.0.0 and access it remotely with 192.x.x.x. ipaddress and port