# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:29:39 2021

@author: IHiggins
"""
import base64
import io
import json
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
   
    dcc.Dropdown(id='site_selector', options=[{'label': i, 'value': i} for i in site_list], style={'display': 'block'}), 
    html.Div(id='site', style={'display': 'none'}),
    html.Div(id='site_sql_id', style={'display': 'none'}),
    html.Div(id='parameter', style={'display': 'none'}),

    # Select a Parameter - get options from callback
    #html.Div(dcc.Dropdown(id='Parameter', value='0'), style={'display': 'block'}),
   
    html.Div(daq.ToggleSwitch(id='Select_Data_Source', value=False),),   # toggle between SQL query and file upload
    html.Div(id='Select_Data_Source_Output'),
    html.P(id="data_label", children=["data column axis"]), 
    dcc.RadioItems(id='data_axis', options=['data', 'corrected_data'], value='data', inline=True),
    #html.Div(dcc.RangeSlider(0, 20, 1, value=[5, 15], id='select_datetime_by_obs'), id = "select_datetime_by_obs_dev", style={'display': 'block'}),
    #html.Div(dcc.Dropdown(id='select_datetime_by_obs_a', options=[""], value='0'), dcc.Dropdown(id='select_datetime_by_obs_b', options=[""], value='0'), id = "select_datetime_by_obs_dev", style={'display': 'block'}),  
    html.Div([
    html.Label("select date range"),
    html.Div(dash_datetimepicker.DashDatetimepicker(id='select_range', startDate='', endDate=''), style={'display': 'flex', 'flex-direction': 'row'}),
    html.Label(" or select first observation  "),
    html.Div(dcc.Dropdown(id='select_datetime_by_obs_a', options=[{"label": "", "value": ""}], value=''), style={'width': '25%', 'box-sizing': 'border-box', 'padding-left': '10px', 'display': 'inline-block'}),
    html.Label("  and select second observation  "),
    html.Div(dcc.Dropdown(id='select_datetime_by_obs_b', options=[{"label": "", "value": ""}], value=''), style={'width': '25%', 'box-sizing': 'border-box', 'padding-left': '10px', 'display': 'inline-block'}),
    dcc.Store(id='query_start_date'),  
    dcc.Store(id='query_end_date'),    
        ], id='select_datetime_by_obs_dev', style={'display': 'none'}),
    dcc.Store(id="statistics"),
    
    

    # Barometric Correction Radio Button
    # dynamic visability https://stackoverflow.com/questions/50213761/changing-visibility-of-a-dash-component-by-updating-other-component
    html.Div(
        # Create element to hide/show, in this case an 'Input Component'
        # dcc.store(id='Barometer_Data'),
        dcc.RadioItems(id='Barometer_Button',
                       options=[
                           {'label': 'Preform Barometric Correction', 'value': 'Baro'},
                           {'label': 'Do Not Preform Barometric Correction',
                               'value': 'No_Baro'}
                       ], value='No_Baro'), style={'display': 'block'}  # <-- This is the line that will be changed by the dropdown callback
        ),

    html.Div(
        dcc.Dropdown(
            id='available_barometers', options=[{'label': i, 'value': i} for i in barometer_list], style={'display': 'none'}),
        ),
    html.Button('Delete Association', id='Delete_Association', n_clicks=0),
    html.Div(id='New_Callback'),
    # Import file structures
    html.Div(
        # Create element to hide/show, in this case an 'Input Component'
        dcc.RadioItems(id='File_Structure',
                       options=[
                        {'label': 'onset_U20', 'value': 'onset_U20'},
                        {'label': 'onset_U24', 'value': 'onset_U24'},
                        {'label': 'aqua4plus_ct2x', 'value': 'aqua4plus_ct2x'},
                        {'label': 'csv', 'value': 'csv'}],
                       value='onset_U20'), style={'display': 'block'}  # <-- This is the line that will be changed by the dropdown callback
        ),

    # CSV Trimming
    html.Div([
        html.Div(daq.NumericInput(id='HEADER_ROWS', label='HEADER ROWS', labelPosition='top', value=1,), style={'width': '10%', 'display': 'inline-block'}),
        html.Div(daq.NumericInput(id='FOOTER_ROWS', label='FOOTER ROWS', labelPosition='top',value=0,), style={'width': '10%', 'display': 'inline-block'}),
        html.Div(daq.NumericInput(id='TIMESTAMP_COLUMN',label='TIMESTAMP_COLUMN', labelPosition='top', value=0,), style={'width': '10%', 'display': 'inline-block'}),
        html.Div(daq.NumericInput(id= 'DATA_COLUMN', label='DATA_COLUMN', labelPosition='top',value=1,), style={'width': '10%', 'display': 'inline-block'}),
    ]),

    dcc.Upload(
        id='datatable-upload',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
        }),
    # date time picker not native to dash see https://community.plotly.com/t/dash-timepicker/6541/10
    
   
    

    # page_action='none',
   
    
    html.Div(id='output-container-date-picker-range'),

  
    html.Div(id='graph_output', style={'width': '50vw', }), # Use viewport width to scale with screen size
    html.Div(dcc.Graph(id='graph',figure=go.Figure(),config={'responsive': True}),),

    html.Div(id="graph_where"),

    html.Div(dcc.Dropdown(id='Ratings', value='NONE'), style={'display': 'block'}),

    html.Div(id="display"),
    dcc.Store(id='observations', storage_type='memory', data = pd.DataFrame().to_json(orient="split")), # set initial observation to blank), # store observations
    dcc.Store(id='import_data', storage_type='memory', data = pd.DataFrame().to_json(orient="split")),
    #dcc.Store(id='import_data', storage_type='memory'),
    dcc.Store(id='comparison_data', storage_type='memory', data = pd.DataFrame().to_json(orient="split")),

    # ### data dables
    html.Div([
        html.Div([
            html.Div(dash_table.DataTable(
                id="corrected_data_datatable", editable=True, sort_action="native", sort_mode="multi", fixed_rows={'headers': True}, row_deletable=False,
               
                page_action='none', style_table={'height': '300px', 'overflowY': 'auto'}, virtualization=True, fill_width=False, filter_action='native',
                style_data={'width': '200px', 'maxWidth': '200px', 'minWidth': '100px'},
                style_data_conditional=[
            # Highlight selected cell
            {
                'if': {'state': 'selected'}, # 'active'
                'backgroundColor': '#FFDDC1',  # Color for active cell
                'color': 'black'
            },
            {
            'if': {
                'column_id': 'c_discharge',
            },
            'backgroundColor': r'rgba(152, 78, 163, 0.3)',  # 0.3 is more transparent then 0.7
            },
            {
            'if': {
                'column_id': 'c_stage',
            },
            'backgroundColor': r'rgba(152, 78, 163, 0.3)',  # 0.3 is more transparent then 0.7
            },
            {
            'if': {
                'column_id': 'c_water_level',
            },
            'backgroundColor': r'rgba(152, 78, 163, 0.3)',  # 0.3 is more transparent then 0.7
            },
            
            #{
            #'if': {
            #    'column_id': 'corrected_data',

                # since using .format, escape { with {{
            #    'filter_query': '{{Pressure}} = {}'.format(df['Pressure'].max())
            #},
            #'backgroundColor': '#85144b',
            #'color': 'white'
        #},

        ],),),
        #], style={'width': '80%', 'display': 'inline-block'}),
        ], style={'flex': '60%', 'width': '60%'}),
        
        
        html.Div([
                dash_table.DataTable(
                    id='observations_datatable', columns=[], data=[], editable=True, row_deletable=True,
                    row_selectable='multi',  # Optionally enable row selection
                    style_table={
                        'height': '300px',  # Fix the height of the table container
                        'overflowY': 'auto',  # Add scrolling for overflow content
                    },
                    style_cell={
                        'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',  # Example column width adjustments
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                    },virtualization=True,),  # Improves performance for large datasets
        
            
            html.Button('add row', id='add_row_button_a', n_clicks=0),
            html.Button('refresh', id='refresh_button_a', n_clicks=0),
            html.Label("how observation range"),
            daq.ToggleSwitch(id='show_inrange_observations', value = True)


        ], style={
            'width': '25%',  # Limit width to 25% of the screen
            'height': '100%',  # Ensure it takes up available height
            'overflow': 'auto',  # Enable scrolling if content overflows
            'display': 'inline-block'  # Prevent div from taking up the entire row
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
                html.P(id="paragraph_id", children=["Button not clicked"]),
            # interpolation and graphing
            html.Div([
                
                html.Button("data managment", id="open-modal-button"),
                    dbc.Modal([dbc.ModalHeader("data managment"),
                    dbc.ModalBody([

                        html.Label('for import sites 1. Resampel to interval, 2. Run expansion, 3. Resample to 15 min, 4. Fill'),
                        html.Div([html.Label('expand to observations: select start and/or end observation to fill to, will need to run filler'), html.Button('run expansion', id='to_observations_button'),], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center','gap': '10px'}),
                            


                        html.Div(id="data_interval", children="data_interval")]),
                         
                                dcc.RangeSlider(id='interval', min=0, max=4, step=None, marks={0: '1 min', 1: '5 min', 2: '15 min', 3: 'hourly', 4: 'daily'}, value=[2]),
                                html.Button('resample', id='resample_button'),                
                       
                        html.Div([
                            html.Button('calculate_average', id='calculate_average_button'), 
                            html.Button('interpolate', id='interpolate_button'), 
                            html.Button('accept interpolation', id='accept_interpolation_button'), 
                        ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center','gap': '10px'}),

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
                       
                        html.Div([html.Label("Interpolation Functions"),], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'center', 'margin-top': '20px'}),
                        
                        html.Div([
                            html.Label("Show Dry Indicator"),
                            daq.ToggleSwitch(id='dry_indicator_button', value=False),
                            html.Button('set dry warning', id='set_dry_indicator_warning_button'), 
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
                           
                            #html.P(id="corrected_data_label", children=["corrected data column axis"]),
                            #dcc.RadioItems(id='corrected_data_axis', options=['primary', 'secondary', 'none'], value='secondary', inline=True),
                            #html.P(id="derived_data_label", children=["derived data column axis"]),
                            #dcc.RadioItems(id='derived_data_axis', options=['primary', 'secondary', 'none'], value='secondary', inline=True),
                            #html.P(id="observation_label", children=["observation column axis"]),
                            #dcc.Checklist(id='observation_axis', options=['show', 'none'], value='secondary', inline=True),
                            #html.P(id="comparison_label", children=["comparison column axis"]),
                            #dcc.RadioItems(id='comparison_axis', options=['primary', 'secondary', 'none'], value='primary', inline=True),
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



        #], style={'width': '20%', 'display': 'inline-block'}),
        ], style={'flex': '20%', 'width': '20%'}),
    ], style={'display': 'flex'}),   
   
   
    # html.Br(),
    html.Div([  # big block
        html.Button('upload_data', id='upload_data_button', n_clicks=0),
        html.Div(id='upload_data_children', style={'width': '5%', 'display': 'inline-block'}),

        html.Button('export_data', id='export_data_button', n_clicks=0),
        html.Div(id='export_data_children', style={'width': '5%', 'display': 'inline-block'}),
        
    ],style={'display': 'flex', 'flex-direction': 'row'}),

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
    Output('File_Structure', component_property='style'),
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

# Show or hide barometric selector; if a file is uploaded to application display the barometric pressure question
@app.callback(
    Output(component_id='Barometer_Button', component_property='style'),
    # esentially an inital pop up vs a delayed popup
    Input('Select_Data_Source', 'value'),
    # Input('datatable-upload', 'contents'),
    Input('parameter', 'children'))
def show_hide_baro(Select_Data_Source, parameter):
    if Select_Data_Source is False and parameter:
        if parameter == "LakeLevel" or parameter == "Piezometer" or parameter == "FlowLevel" or parameter == "discharge" or parameter == "lake_level" or parameter == "water_level" or parameter == "groundwater_level":
       
            return {'display': 'block'}
        else:
            return {'display': 'none'} # default is no display so dont update
    #elif Select_Data_Source is True:
    #    return {'display': 'none'}
    else:
        return dash.no_update # default is no display so dont update

# Show or hide barometric search
@app.callback(
    Output(component_id='available_barometers', component_property='style'),
    Input('Barometer_Button', 'value'),
    Input('Select_Data_Source', 'value'),
    Input(component_id='Barometer_Button', component_property='style'),)
def display_barometer_search(Barometer_Button, Select_Data_Source, style):

    if style == {'display': 'none'}: #or Barometer_Button == 'No_Baro':
        print("hide baro search")
    elif style != {'display': 'none'}:
        return {'display': 'inline-block'}

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
    labels = {0: '1', 1: '5', 2: '15', 3: '60', 4: '1440'}
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


# Barometer Search

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
    from import_data import get_site_sql_id, sql_import 
    if comparison_sites and start_date and end_date:
            start_date =  (pd.to_datetime(start_date).to_pydatetime()) - timedelta(hours=primer_hours)
            end_date =  (pd.to_datetime(end_date).to_pydatetime()) + timedelta(hours=primer_hours)
            # convert string to list
            #comparison_sites = comparison_sites.split()
            comparison_data = pd.DataFrame(columns=["datetime", "psite", "parameter", "corrected_data"])
            for site_item in comparison_sites:
                # usgs sites
                if site_item.startswith("USGS"): 
                    #if site_name.startswith("USGS"): # would have to change this if non usgs external sites are added...
                    #parameters = pd.read_csv("external_comaprison_sites.csv", skipinitialspace=True)
                    #parameters = parameters["parameter"].loc[parameters["site"] == site_name].item()
                    #parameters = [param.strip() for param in parameters.split(',')]
                    print("usgs")  
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

                        df["parameter"] = "stage"
                        df = df[["site", "datetime", "parameter", "corrected_data"]]
                        
                        df = pd.concat([df, df_q])
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
            return [site_selector_value] # sets the site to be a comparison site to see existing data
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
           
### Date range
@app.callback(
    Output("select_datetime_by_obs_dev", "style"),
    Output('select_datetime_by_obs_a', 'options'),  # obs a
    Output('select_datetime_by_obs_b', 'options'),  # obs b
    Output('query_start_date', 'data'),
    Output('query_end_date', 'data'),
    Output('statistics', 'data'),
    Input("observations", "data"),
    Input('select_range', 'startDate'),  # startDate is a dash parameter
    Input('select_range', 'endDate'),
    Input('site', 'children'), # may not neven need a site trigger but dont need site, siteid, and paramter
    State("site_sql_id", "children"),
    State('parameter', 'children'),
    Input('Select_Data_Source', 'value'),
    Input('select_datetime_by_obs_a', 'value'),  # obs a
    Input('select_datetime_by_obs_b', 'value'),  # obs b
    Input('data_interval', 'children'),
    #Input('import_data', 'data'),
    )



def daterange(observations, startDate, endDate, site, site_sql_id, parameter, data_source, obs_a, obs_b, data_interval):
    """ignores date range for now"""
    query_start_date = ""
    query_end_date = ""
    # should just have 1 callback that querys obs for everything
    #if data_source == True and site != '0' and parameter != '0' :
    if site and parameter:
        obs = pd.read_json(observations, orient="split")

        statistics =  sql_statistics(parameter, site_sql_id)
     
        stats_df = pd.DataFrame({'datetime': [statistics["first_datetime"], statistics["last_datetime"], datetime.now().replace(second=0, microsecond=0)],'observation_number': ['first record', 'last record', "today (just todays date)"]})
       
        #obs = obs.merge(stats_df, on = ["datetime", "observation_number"], how = "outer")
        obs = pd.concat([obs, stats_df])
        obs = obs.sort_values(by='datetime', ascending=False) 
        
        obs_a_options = [f"date: {row['datetime'].strftime('%Y-%m-%d %H:%M')} observation: {row['observation_number']}" for _, row in obs.iterrows()]
        if obs_a != "": # filter observations so you only see observations greater then first selection
            obs_a = obs_a.split("date: ")[1].split(" observation:")[0].strip()
            obs_a = datetime.strptime(obs_a, '%Y-%m-%d %H:%M')
            obs = obs.loc[obs["datetime"] > obs_a]

            # find closest value in sql
            query_start_date = pd.to_datetime(obs_a).to_pydatetime()
            from import_data import sql_get_closest_datetime
            # get min value
            result_a, result_b, result_c = sql_get_closest_datetime(parameter, site_sql_id, query_start_date)
            print(f"obs a options result a {result_a} result b {result_b} result c {result_c}")
            if result_b != result_c: # obs is before the end of the datafrmae
                 interval = abs(result_c - result_b)
            #if query_start_date == result_a:
            #    query_start_date = result_a
            #else:
            #    query_start_date = min(result_a, result_b)
            query_start_date = result_a
            print("query start date", query_start_date)
        #elif obs_a == "" and data_source == 'File Import' and not pd.read_json(import_data, orient="split").empty: # if your are importind data you can use
        #        import_data = pd.read_json(import_data, orient="split").empty
        #        query_start_date = import_data['datetime'].min()
        #        print("no obs import data obs a", query_start_date)
        obs_b_options = [f"date: {row['datetime'].strftime('%Y-%m-%d %H:%M')} observation: {row['observation_number']}" for _, row in obs.iterrows()]
        if obs_b != "":
            obs_b = obs_b.split("date: ")[1].split(" observation:")[0].strip()
            obs_b = datetime.strptime(obs_b, '%Y-%m-%d %H:%M')
            query_end_date = pd.to_datetime(obs_b).to_pydatetime()
            result_a, result_b, result_c = sql_get_closest_datetime(parameter, site_sql_id, query_end_date)
            print(f"obs b options result a {result_a} result b {result_b} result c {result_c}")
            #print(f"option b maths c-b {abs(result_c - result_b)} a-b {abs(result_a - result_b)}")
            
            # result a is closest, b is below, c is above
            if result_a == result_b and result_b == result_c: # last db log is before this peroid, use query end date
                query_end_date = pd.Timestamp(query_end_date).floor(f'{data_interval}T') + pd.Timedelta(f"{data_interval}T")
                print("first option query end date", query_end_date)
            elif result_a != result_b and result_b != result_c: # this works for some discharge sites
                query_end_date = result_a
                print("second option query end date", query_end_date)
            elif result_a == result_b and result_a < result_c:
                 query_end_date = result_c
                 print("third option query end date", query_end_date)
            else:

                query_end_date = result_b
                print("final option query end date", query_end_date)
        #elif obs_b == "" and data_source == 'File Import' and not pd.read_json(import_data, orient="split").empty: # if your are importind data you can use
        #        import_data = pd.read_json(import_data, orient="split").empty
        #        query_end_date = import_data['datetime'].max()
        #        print("no obs import data obs b", query_end_date)
        return {'display': 'flex', 'flex-direction': 'row'}, obs_a_options, obs_b_options, query_start_date, query_end_date, statistics # obs a is query start date obs b is query end date

    else:
        #min_obs = 0
        #max_obs = 0
        #return {'display': 'none'}, [""], [""], "", "", ""
        return dash.no_update



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
                
            ### calculate barometer
            if barometer_button == "Baro": #and data_source is False:  # data source false is file import
                from import_data import get_site_sql_id
                barometer_site = available_barometers
                barometer_sql_id = get_site_sql_id(barometer_site)
                # THIS IS DUMB, ITS A PLACHHOULDER there needs to be a formula to convert wl feet
                if df['data'].mean() < 50: # if under 30 assumed to be psi
                    df['data'] = round((df['data']*68.9476), 3) # convert to millibar
                     
                if df['data'].mean() > 999: # assumed to be millibar
                    df['data'] = df['data'] # millibar

                barometer_query = sql_import("barometer", barometer_sql_id, df['datetime'].min(), df['datetime'].max()) # fx converts to PST and out of PST
                barometer_query = barometer_query.rename(columns = {"corrected_data": "barometer_data"})
                # resample barometer to 5 minutes
                barometer_query = barometer_query.set_index("datetime").resample('5T').interpolate(method='linear').reset_index(level=None, drop=False)
                 
                df = pd.merge(df,barometer_query[['datetime', "barometer_data"]],on=['datetime'])
                  
                df['data'] = ((df['data']-df["barometer_data"]) * 0.0335).round(3)
                output = df.drop(['barometer_data'], axis=1)
                
                return  output.to_json(orient="split")#, df['datetime'].min() + timedelta(hours=(7)), df['datetime'].max() + timedelta(hours=(7))
            elif barometer_button != "Baro":
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
    
    #Output('observations_datatable', 'columns'),
        
       
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
        #observations = reformat_data(observations)

   
    #print(observations)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0] 
    #ctx = callback_context
    #if ctx.triggered: 
    if not observations.empty and parameter != "0": 
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



### corrected_data
@app.callback(
    Output("corrected_data_datatable", "data"),
    Output("corrected_data_datatable", "columns"),
    Input('import_data', 'data'),
    Input('observations_datatable', 'data'), # I think this can be state as it doesnt need to trigger when it updates itself
    Input('comparison_data', 'data'), # ideally this would be state but the comparison data seems to load after this dev
    #Input('add_row_button_b', 'n_clicks'), # df is usually to big to add rows
    State('parameter', 'children'),
    State('site', 'children'),
    State("site_sql_id", "children"),
    State('corrected_data_datatable', 'data'), # I think this can be state as it doesnt need to trigger when it updates itself
    Input("realtime_update", "value"),
    Input("run_job", "n_clicks"),
    State('Ratings', 'value'),
    Input('basic_forward_fill', 'n_clicks'),
    Input('basic_backward_fill', 'n_clicks'),
    State("fill_limit", 'value'),
    State("fill_limit_number", 'value'),
    State("fill_limit_area", "value"),
    State("set_method", 'value'), # sets method for interpolation
    State('set_limit', 'value'), # fill missing data
    State('limit_number', 'value'), # fill missing data
    State('limit_direction', 'value'), # fill missing data
    State('limit_area', 'value'), # fill missing data
    Input('fill_missing_data', 'n_clicks'), # fill missing data
    State('data_interval', 'children'),
    Input('resample_button', 'n_clicks'),
    State('query_start_date', 'data'),  # obs a
    State('query_end_date', 'data'), # obs b),
    Input("to_observations_button", "n_clicks"),
    State('Select_Data_Source', 'value'),
    Input("data_axis", "value")  # data level data or corrected_data (primary/secondary)
)

  

def correct_data(import_data, obs_rows, comparison_data, parameter, site, site_sql_id, rows, realtime_update, run_job, rating_number, basic_forward_fill, basic_backward_fill, fill_limit, fill_limit_number, fill_area, method, set_limit, limit_number, limit_direction, limit_area, fill_missing_data,data_interval, resample, query_start_date, query_end_date, to_observations_button, data_source, data_level):
    
    
    from data_cleaning import reformat_data, initial_column_managment, column_managment, fill_timeseries
    from discharge import discharge_calculation
    
    try:
        observation = config[parameter]["observation_class"]
    except KeyError:
        observation = ""
    # Get the triggered property from Dash callback context
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0] 
   

    if not pd.read_json(import_data, orient="split").empty and not "datetime" in pd.DataFrame(rows).columns: # and import_data is not None:
            data = pd.read_json(import_data, orient="split")
    elif "datetime" in pd.DataFrame(rows).columns:
            data = pd.DataFrame(rows)
    else:
            return dash.no_update

    data = reformat_data(data)
    data = initial_column_managment(data)
    data = fill_timeseries(data, data_interval)
    
        # format observations if obser
    if obs_rows: #"datetime" in observations.columns:
            #observations = pd.DataFrame(obs_rows)
        observations = pd.DataFrame(obs_rows) # but be warned; if obs returns a blank datatable this may mess up
        observations = reformat_data(observations)
        observations = observations.dropna(subset=['datetime'])
        #if "datetime" in data.columns:
        data = pd.merge_asof(data.sort_values('datetime'), observations.sort_values('datetime'), on='datetime', tolerance=pd.Timedelta(f"{int(data_interval)/2}m"), direction="nearest")
        
    data = reformat_data(data)
    # add existing data, will get deleted at the beginning of each itteration
    #if data_source is False: # File Import  -  only need this when uploading data

    comparison_data = pd.read_json(comparison_data, orient = "split")
    if not comparison_data.empty:
            comparison_data = reformat_data(comparison_data)
            c_data = comparison_data.loc[(comparison_data["site"] == site) & (comparison_data["parameter"] == parameter)]
            c_data = c_data[["datetime", "corrected_data"]] # comparison data only contains the parameter so for discharge the comparison "corrected_data" is "discharge"
            if not c_data.empty:
                c_data = c_data.rename(columns={"datetime": "datetime", "corrected_data": f"c_{parameter}"})
            if parameter == "discharge":
                wl_data = comparison_data.loc[(comparison_data["site"] == site) & (comparison_data["parameter"] == "stage")]
                wl_data = wl_data[["datetime", "corrected_data"]] # comparison data only contains the parameter so for discharge the comparison "corrected_data" is "discharge"
                if not wl_data.empty:
                    wl_data = wl_data.rename(columns={"datetime": "datetime", "corrected_data": f"c_stage"})
                    c_data = wl_data.merge(c_data, left_on="datetime", right_on="datetime", how = "outer")
        # data = pd.merge(comparison_data.sort_values('datetime'), data.sort_values('datetime'), on='datetime', )
        #DataFrame.join(other, on=None, how='left', lsuffix='', rsuffix='', sort=False, validate=None)
           
            #data = c_data.merge(data, left_on="datetime", right_on="datetime", how = "outer") # eg
            data = data.merge(c_data, left_on="datetime", right_on="datetime", how = "outer") # eg
            #data = pd.concat([ data, c_data], axis=0).sort_values(by="datetime").reset_index(drop=True)

          
    # Realtime update or run job logic
    if realtime_update is True or 'run_job' in changed_id:
                from interpolation import basic_interpolation, run_basic_forward_fill, run_basic_backward_fill
                # fill missing data
                changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
                if "to_observations_button" in changed_id:
                    from data_cleaning import to_observations
                    data = to_observations(data, query_start_date, query_end_date)
                if "basic_interpolate_missing_data" in changed_id:
                    
                    data = basic_interpolation(data, method, set_limit, limit_number, limit_direction, limit_area)
                if "basic_backward_fill" in changed_id:
                     data = run_basic_backward_fill(data, fill_limit,fill_limit_number, fill_area)
                if "basic_forward_fill" in changed_id:
                     data = run_basic_forward_fill(data, fill_limit, fill_limit_number, fill_area)
                if "resample_button" in changed_id:
                    from interpolation import resample
                    data = resample(data, data_interval)
                data = parameter_calculation(data, data_level)
                    
                # Additional logic for specific parameters (e.g., discharge)
                if (parameter == "discharge" or parameter == "FlowLevel") and rating_number != "NONE":   
                    data = discharge_calculation(data, rating_number, site_sql_id)

                data = column_managment(data)
                    # Reformat the datetime for Dash graph display
    data["datetime"] = data["datetime"].dt.strftime('%Y-%m-%d %H:%M')
   
                    # Return updated data and columns
    return data.to_dict('records'), [{"name": i, "id": i} for i in data.columns]
    #else:
    #    return dash.no_update
 
  
@app.callback(
        #Output(component_id='graph_output', component_property='children'),
        Output('graph', "figure"),
        Input('graph_realtime_update', 'value'),
        Input("corrected_data_datatable", "data"),
        State('site_selector', 'value'), # only needed to graph existing data differenty
        State('site', 'children'),
        State('site_sql_id', 'children'),
        State('parameter', 'children'),

        Input('comparison_data', 'data'),

        #Input('comparison_site_sql_id', 'children'),
        #Input('comparison_parameter', 'value'),
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
        #df = df.loc[(df["datetime"] >= query_start_date) & (df["datetime"] <= query_end_date)].copy()
)

def graph(graph_realtime_update, df, site_selector_value, site, site_sql_id, parameter, comparison_data, rating, primary_min, primary_max, secondary_min, secondary_max, normalize_data, statistics, display_statistics, query_start_date, query_end_date):
    from data_cleaning import reformat_data, parameter_calculation
    from graph_2 import cache_graph_export
    df = pd.DataFrame(df)
    comparison_data = pd.read_json(comparison_data, orient = "split")
    
    if not df.empty and graph_realtime_update is True:

        df = reformat_data(df) 
        if query_start_date != "" and query_end_date != "": # really we are trying to get rid of comparison data
                df = df.loc[(df["datetime"] >= query_start_date) & (df["datetime"] <= query_end_date)].copy()
        if query_start_date == "" or not query_start_date:
            print(df)
            query_start_date = df['datetime'].min()
            print("graph fill query start date", query_start_date)
        if query_end_date == "" or not query_end_date:
            query_end_date = df['datetime'].max()
            print("graph fill query end date", query_end_date)
        #fig = html.Div(dcc.Graph(figure = go.Figure()), style = {'width': '100%', 'display': 'inline-block'})
        fig = cache_graph_export(df, site_sql_id, site_selector_value, site, parameter, comparison_data, primary_min, primary_max, secondary_min, secondary_max, normalize_data, statistics, display_statistics)
        return fig
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
    [dash.dependencies.Input('upload_data_button', 'n_clicks')],
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
    State('query_start_date', 'data'),  # obs a
    State('query_end_date', 'data'), # obs b),
        
    )

def run_upload_data(n_clicks, df, site_selector_value, site, site_sql_id, parameter, comparison_data, rating, primary_min, primary_max, secondary_min, secondary_max, normalize_data, statistics, display_statistics, query_start_date, query_end_date):
    from data_cleaning import reformat_data 
    from graph_2 import save_fig
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'upload_data_button' not in changed_id:
        return dash.no_update

    elif 'upload_data_button' in changed_id:
        df = pd.DataFrame(df)
        notes_df = df
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if (df.empty or len(df.columns) < 1):
            return dash.no_update
        else:
            # IF THERE is existing eata drop it
            if "existing_data" in df.columns:
                df = df.loc[df.existing_data.isnull()]
            
            #end_date = df["datetime"].max().date().strftime("%Y_%m_%d")
            df = reformat_data(df)
          
            if query_start_date != "" and query_end_date != "": # really we are trying to get rid of comparison data
                df = df.loc[(df["datetime"] >= query_start_date) & (df["datetime"] <= query_end_date)].copy()
            notes_df = df
            end_date = df["datetime"].max().date()
            comparison_data = pd.read_json(comparison_data, orient = "split")
            save_fig(df, site_selector_value, site, site_sql_id, parameter, comparison_data, rating, end_date, primary_min, primary_max, secondary_min, secondary_max, normalize_data, statistics, display_statistics)
            from sql_upload import full_upload

            desired_order = ["datetime", "data", "corrected_data", "discharge", "estimate", "warning"] # observation and observation_stage are kinda redundent at some point and should be clarified
            # Filter out columns that exist in the DataFrame
            existing_columns = [col for col in desired_order if col in df.columns]
            # Reorder the DataFrame columns
            df = df[existing_columns].copy()
            # rename parameters
            if parameter == "Conductivity" or parameter == "conductivity":
                parameter = "Conductivity"
            if parameter == "water_level" or parameter == "LakeLevel":
                parameter = "water_level"
            if parameter == 'groundwater_level' or parameter == "Piezometer":
                parameter = "groundwater_level"
            if parameter == "discharge" or parameter == "FlowLevel":
                parameter = "discharge"
            print("data for full upload")
            print(df)
            full_upload(df, parameter, site_sql_id, 7)
            
            from workup_notes import workup_notes_main

            workup_notes_main(notes_df, parameter, site_sql_id, site)
            print("work up notes")
            result = "  uploaded"

            return result
    else:
        return dash.no_update

@app.callback(
    dash.dependencies.Output('export_data_children', 'children'),
    [dash.dependencies.Input('export_data_button', 'n_clicks')],
    State('corrected_data_datatable', 'data'),
    State('site_selector', 'value'),
    State('site', 'children'),
    State('site_sql_id', 'children'),
    State('parameter', 'children'),
    State('Ratings', 'value'),
    State('comparison_data', 'data'),
    State("primary_min", "value"),
    State("primary_max", "value"),
    State("secondary_min", "value"),
    State("secondary_max", "value"),
    State("normalize_data", "value"),
    State("statistics", "data"),
    State("display_statistics", "value"),
    State('query_start_date', 'data'),  # obs a
    State('query_end_date', 'data'), # obs b),
        
    )
def run_export_data(n_clicks, df, site_selector_value, site, site_sql_id, parameter, comparison_data, rating, primary_min, primary_max, secondary_min, secondary_max, normalize_data, statistics, display_statistics, query_start_date, query_end_date):
    from data_cleaning import reformat_data 
    from graph_2 import save_fig
    ''' uses same function as update graph, this code is becomingly increasingly redundent '''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'export_data_button' not in changed_id:
        return dash.no_update
    
    if 'export_data_button' in changed_id:
        df = pd.DataFrame(df)
        if (df.empty or len(df.columns) < 1):
            return dash.no_update
        else:
             #df, parameter, observation, end_time = format_cache_data(df, site, parameter, fig)
            df = reformat_data(df)  
            if query_start_date != "" and query_end_date != "": # really we are trying to get rid of comparison data
                df = df.loc[(df["datetime"] >= query_start_date) & (df["datetime"] <= query_end_date)].copy()
            
            df_export = df.set_index('datetime').copy()
            #end_date = df["datetime"].max().date().strftime("%Y_%m_%d")
            end_date = df["datetime"].max().date()
          
            df_export.to_csv("W:/STS/hydro/GAUGE/Temp/Ian's Temp/" +
                str(site)+"_"+str(parameter)+"_"+str(end_date)+".csv")
           
            comparison_data = pd.read_json(comparison_data, orient = "split")
            save_fig(df, site, site_selector_value, site_sql_id, parameter, rating, end_date, primary_min, primary_max, secondary_min, secondary_max, normalize_data, statistics, display_statistics)

            result = "  exported"
            return result
            #return result
    else:
        return dash.no_update


# You could also return a 404 "URL not found" page here
if __name__ == '__main__':
    app.run_server(port="8050",host='127.0.0.1',debug=True)
   
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