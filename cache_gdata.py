# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:29:39 2021

@author: IHiggins
"""
import base64
import io
import pyodbc
import configparser
import dash
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


## fix copy of slice error with df.loc[df.A > 5, 'A'] = 1000

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets, long_callback_manager=long_callback_manager)
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#Driver = 'SQL Server'
#Server = 'KCITSQLPRNRPX01'
#Database = 'gData'
#Trusted_Connection = 'yes'

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

# Gives a string of available parameters to the SQL query
Available_Parameters = 'AirTemp, barometer, Conductivity, DO, FlowLevel, Humidity, LakeLevel, pH, Piezometer, Precip, Relative_Humidity, SolarRad, Turbidity, WaterTemp, Water_Quality, Wind'
#filepath = r'C:/Users/ihiggins/cache.pkl'
#pickle.dump(Available_Parameters, open(filepath, 'wb'))
# Available_Parameters = Available_Parameters.tolist()
# Get site list
# Query SQL every sql query will need its own CONN
# INITIAL AVAILABLE BAROMOTERS

with sql_engine.begin() as conn:
        #gage_lookup = pd.read_sql_query('select G_ID, SITE_CODE from tblGaugeLLID;', conn)
        Available_Baros = pd.read_sql_query(
                        f"SELECT {config['site_identification']['site']} "
                        f"FROM {Site_Table} WHERE STATUS = 'Active' AND Barometer = 'True' ORDER BY {config['site_identification']['site']} ASC;", conn)

Available_Baros = Available_Baros[config['site_identification']
                    ["site"]].values.tolist()

# Barometer Association Table
Barometer_Association_Table = 'tblBaroLoggerAssociation'
with sql_engine.begin() as conn:
    Available_Sites = pd.read_sql_query("select SITE_CODE, G_ID from tblGaugeLLID WHERE STATUS = 'Active' ORDER BY SITE_CODE ASC;", conn)
    # site_sql_id = pd.read_sql_query(f"select G_ID as site_sql_id from tblGaugeLLID WHERE SITE_CODE = {site_number};", conn)
# this will need to change when there is more then just flowlevel
vlist = Available_Sites['SITE_CODE'].values.tolist()

comparison_list = pd.read_csv('external_comaprison_sites.csv', skipinitialspace=True)
comparison_list = comparison_list["site"].values.tolist()
comparison_list = vlist + comparison_list


app.layout = html.Div([
    # dcc.Location(id='url', refresh=False),
    # Select a Site
    # Site = site name site_sql_id is site number
   
    dcc.Dropdown(id='site', options=[{'label': i, 'value': i} for i in vlist], value='0', style={'display': 'block'}), 
    html.Div(id='site_sql_id', style={'display': 'none'}),

    # Select a Parameter - get options from callback
    html.Div(dcc.Dropdown(id='Parameter', value='0'), style={'display': 'block'}),
   
    html.Div(daq.ToggleSwitch(id='Select_Data_Source', value=False),),   # toggle between SQL query and file upload
    html.Div(id='Select_Data_Source_Output'),

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
            id='Available_Barometers', options=[{'label': i, 'value': i} for i in Available_Baros], value="0", style={'display': 'none'}),
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
    
    html.Div(dash_datetimepicker.DashDatetimepicker(id='select_range', startDate='', endDate=''),style={'display': 'block'}),
    

    # page_action='none',
    html.Div(
        # Create element to hide/show, in this case an 'Input Component'
        dcc.RadioItems(id='select_data_level',
                       options=[
                           {'label': 'data', 'value': 'data'},
                           {'label': 'corrected_data', 'value': 'corrected_data'}
                       ], value='data'), style={'display': 'block'},  # <-- This is the line that will be changed by the dropdown callback
        ),
    
    html.Div(id='output-container-date-picker-range'),

    # returns a graph
    #html.Div(id='graph_output', style={'width': '100%', 'display': 'inline-block'}),
    html.Div([html.Div(id='graph_output', style={'flex': '100%', 'width': '100%'})], style={'display': 'flex'}),

    #style={'display': 'flex', 'flex-direction': 'row'})
    html.Div(id="graph_where"),

    html.Div(dcc.Dropdown(id='Ratings', value='NONE'), style={'display': 'block'}),

    html.Div(id="display"),
    dcc.Store(id='import_data', storage_type='memory'),
    #dcc.Store(id="barometer_corrected_data", storage_type='memory'),
    # visable data table
    # html.Div(dcc.RadioItems(['all data', 'observations','discharge observations'], 'All Data', id = "filter_options", inline=True)),
    
    html.Div([
        html.Div([
            html.Div(dash_table.DataTable(
                id="Corrected_Data", editable=True, sort_action="native", sort_mode="multi", fixed_rows={'headers': True}, row_deletable=False,
                page_action='none', style_table={'height': '300px', 'overflowY': 'auto'}, virtualization=True, fill_width=False, filter_action='native',
                style_data={'width': '200px', 'maxWidth': '200px', 'minWidth': '100px'},
                style_data_conditional=[{'if': {'column_id': 'comparison'}, 'backgroundColor': 'rgb(222,203,228)', 'color': 'black'}],),),
        #], style={'width': '80%', 'display': 'inline-block'}),
        ], style={'flex': '80%', 'width': '80%'}),
        
        html.Div([
            html.Div([
                dcc.Dropdown(id='comparison_site',options=[{'label': i, 'value': i} for i in comparison_list],value='0'), 
                html.Div(id='comparison_site_sql_id', style={'display': 'none'})]),
                dcc.Dropdown(id='comparison_parameter', value='0'),
                dcc.Checklist(id="checklist", options=['comparison_site'],value=['comparison_site'],inline=True),
                # realtime update info
                html.Div([daq.ToggleSwitch(id='realtime_update'), html.Button(id="run_job", children="Run Job!"), html.Div(id='realtime_update_info'),], style={'display': 'flex', 'flex-direction': 'row'}), #dynamic default so sql query doesnt automatically correct for obs
                html.P(id="paragraph_id", children=["Button not clicked"]),
            # interpolation and graphing
            html.Div([
                
                html.Button("data managment", id="open-modal-button"),
                    dbc.Modal([dbc.ModalHeader("data managment"),
                    dbc.ModalBody([dcc.RangeSlider(id='interval', min=0, max=4, step=None, marks={0: '1 min', 1: '5 min', 2: '15 min', 3: 'hourly', 4: 'daily'}, value=[2]),
                    html.Div(id="data_interval", children="data_interval")]),
                        html.Button('interpolate', id='interpolate_button'), 
                        
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
                            html.P(id="data_label", children=["data column axis"]), 
                            dcc.RadioItems(id='data_axis', options=['primary', 'secondary', 'none'], value='primary', inline=True),
                            html.P(id="corrected_data_label", children=["corrected data column axis"]),
                            dcc.RadioItems(id='corrected_data_axis', options=['primary', 'secondary', 'none'], value='secondary', inline=True),
                            html.P(id="derived_data_label", children=["derived data column axis"]),
                            dcc.RadioItems(id='derived_data_axis', options=['primary', 'secondary', 'none'], value='secondary', inline=True),
                            html.P(id="observation_label", children=["observation column axis"]),
                            dcc.Checklist(id='observation_axis', options=['show', 'none'], value='secondary', inline=True),
                            html.P(id="comparison_label", children=["comparison column axis"]),
                            dcc.RadioItems(id='comparison_axis', options=['primary', 'secondary', 'none'], value='primary', inline=True),
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
   
   #"""app.layout = html.Div([
   # html.Div('Component A', style={'flex': '80%'}),
   # html.Div('Component B', style={'flex': '20%'}),
   # ], style={'display': 'flex'})"""

    # fill_width=False, style_data={'width': '200px','maxWidth': '200px','minWidth': '100px',},
    
   
    # html.Br(),
    html.Div([  # big block
        html.Button('upload_data', id='upload_data_button', n_clicks=0),
        html.Div(id='upload_data_children', style={'width': '5%', 'display': 'inline-block'}),

        html.Button('export_data', id='export_data_button', n_clicks=0),
        html.Div(id='export_data_children', style={'width': '5%', 'display': 'inline-block'}),
        html.Div(dash_table.DataTable(id="Initial_Data_Correction", virtualization=True,),style={"display": "none"},),
    ],style={'display': 'flex', 'flex-direction': 'row'}),

])


# Select file source
@app.callback(
    Output('Select_Data_Source_Output', 'children'),
    Input('Select_Data_Source', 'value'))
def update_output(Select_Data_Source):
    if Select_Data_Source is False:
        return 'File Import'
    if Select_Data_Source is True:
        return 'Database Query'


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
    Input('Select_Data_Source', 'value'))
def run_job(select_data_source):
    # I think True should be sql query but that does not seem to work, however I think the true and false are confusing
    if select_data_source == False:  # if file upload automatically correct
        return True  # ealtime Update off
    if select_data_source == True: # if sql pause correction so uploaded data can be shown

        return False
"""
# global datetime
@app.callback(
    Output('select_range', 'startDate'),
    Output('select_range', 'endDate'),
    Input('select_range', 'startDate'),
    Input('select_range', 'endDate'),)
def update_output(start_date, end_date):
    # Convert the UTC time to PDT (UTC-7)
    if start_date != '' and end_date != '':
        start_date = (pd.to_datetime(start_date).to_pydatetime()) + timedelta(hours=(7))
        start_date = start_date + timedelta(hours=(7))
        end_date = pd.to_datetime(end_date).to_pydatetime()
        end_date = end_date + timedelta(hours=(7))
        return start_date, end_date
    else:
        return '', ''"""


# ONSET
@app.callback(
    Output('HEADER_ROWS', 'value'),
    Output('FOOTER_ROWS', 'value'),
    Output('TIMESTAMP_COLUMN', 'value'),
    Output('DATA_COLUMN', 'value'),
    Input('File_Structure', 'value'),
    Input('Parameter', 'value'))
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
    Input('Parameter', 'value'))
def show_hide_baro(Select_Data_Source, value):
    if Select_Data_Source is False:
        if value == "LakeLevel" or value == "Piezometer" or value == "FlowLevel" or value == "discharge" or value == "lake_level" or value == "water_level" or value == "groundwater_level":
            return {'display': 'block'}
        else:
            return {'display': 'none'}
    if Select_Data_Source is True:
        return {'display': 'none'}


# Show or hide barometric search
@app.callback(
    Output(component_id='Available_Barometers', component_property='style'),
    Input('Barometer_Button', 'value'),
    Input('Select_Data_Source', 'value'),
    Input('Parameter', 'value'),
    Input(component_id='Barometer_Button', component_property='style'),)
def display_barometer_search(Barometer_Button, Select_Data_Source, value, style):
    if style == {'display': 'none'} or Barometer_Button == 'No_Baro':
        return {'display': 'none'}
    if style != {'display': 'none'}:
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
    print(data_interval)
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
    Output(component_id='site_sql_id', component_property='children'),
    Input(component_id='site', component_property='value'))
def get_sql_number_from_gid(site_value):
    if site_value != "0":
        search = Available_Sites.loc[Available_Sites['SITE_CODE'].isin([site_value])]
        value = search.iloc[0, 1]
        return "{}".format(value)
    else:
        value = "0"
        return 'You have selected "{}"'.format(value)

# companion get sql number
@app.callback(
    Output(component_id='comparison_site_sql_id', component_property='children'),
    Input(component_id='comparison_site', component_property='value'))
def get_companion_sql_number_from_gid(site_value):
    if site_value != "0":
        try:
            search = Available_Sites.loc[Available_Sites['SITE_CODE'].isin([site_value])]
            value = search.iloc[0, 1]
            return "{}".format(value)
        except: # if its an external site it wont show up in available sites
            comparison_list = pd.read_csv('external_comaprison_sites.csv', skipinitialspace=True)
            search = comparison_list.loc[comparison_list['site'].isin([site_value])]
            value = search.iloc[0, 1]
            
            return "{}".format(value)
    else:
        value = "0"
        return 'You have selected "{}"'.format(value)
# Barometer Search


@app.callback(
    Output(component_id='Available_Barometers', component_property='value'),
    Input(component_id='site_sql_id', component_property='children'),
    Input(component_id='Barometer_Button', component_property='value'),
    Input('Available_Barometers', 'style'))
def update_dp(site_sql_id, Barometer_Button, style):
    # if Site == '0':
    # return [{'label': '0', 'value': '0'}]
    if style != {'display': 'none'}:

        with sql_engine.begin() as conn:
            Baro_Gage = pd.read_sql_query('select WaterLevel_GID, Airpressure_GID from '+str(Barometer_Association_Table)+';', conn)
        
        Baro_Gage.reset_index()

        Baro_Gage_search = Baro_Gage.loc[Baro_Gage['WaterLevel_GID'].isin([site_sql_id])]
        # ILOC is row, column
        if Baro_Gage_search.empty:
            Baro_ID = ""
        else:
            Baro_ID = Baro_Gage_search.iloc[0, 1]
            search_2 = Available_Sites.loc[Available_Sites['G_ID'].isin([Baro_ID])]
            B_ID_Lookup = search_2.iloc[0, 0]
            return str(B_ID_Lookup)
    else:
        return str("")


# Delete Barometer Association
'''
@app.callback(
    Output('New_Callback', 'children'),
    Input(component_id='Available_Barometers', component_property='value'),
    Input(component_id='Site', component_property='value'),
    Input('Delete_Association', 'n_clicks'),)

def delete_association(n_clicks, Site, Available_Barometers):
    # if n_clicks == 1:
        # conn_7 = SQL_String
        # get g_id
        # s earch = Available_Sites.loc[Available_Sites['SITE_CODE'].isin([Site])]
        # G_ID_Lookup = search.iloc[0,1]
        # search_2 = Available_Sites.loc[Available_Sites['SITE_CODE'].isin([Available_Barometers])]
        # B_ID_Lookup = search_2.iloc[0,1]
        # cursor = conn_7.cursor()
        # conn_7.execute('delete from '+str(Barometer_Association_Table)+' WHERE WaterLevel_GID = '+str(G_ID_Lookup)+'')
        # conn_7.commit()
        # conn_8 = SQL_String
        # cursor = conn_8.cursor()
        # cursor.execute('INSERT INTO '+str(Barometer_Association_Table)+' (WaterLevel_GID, Airpressure_GID) VALUES(?,?)', str(G_ID_Lookup), str(search_2))
        # conn_8.commit()
        # return html.Div(str(search))
    # else:
        # return html.Div("nada")
'''


# Get Parameters
@app.callback(
    Output(component_id='Parameter', component_property='options'),
    Output(component_id='Parameter', component_property='style'),
    Input(component_id='site', component_property='value'),
    Input(component_id="site_sql_id", component_property="children"))
def update_parameters(site_value, site_sql_id):
    if site_value == '0':
        return [{'label': '0', 'value': '0'}], {'display': 'block'}
    if site_value != '0':
        # NEW CONN STRING
        with sql_engine.begin() as conn:
            Parameters = pd.read_sql_query(f"SELECT {Available_Parameters} from {config['parameters']['parameter_table']} WHERE G_ID = '{site_sql_id}' AND STATUS = 'Active';", conn)
        if "WaterTemp" in Parameters.columns:
            Parameters = Parameters.rename(
                columns={"WaterTemp": "water_temperature"})
        # vlist.set_index('SITE_CODE', inplace=True)
        Parameters = Parameters.loc[:, (Parameters != 0).any(axis=0)]
        # returns list of columns for dropdown
        Parameters = Parameters.columns.values.tolist()
        return [{'label': i, 'value': i} for i in Parameters], {'display': 'block'}

# companion parameters
@app.callback(
    Output(component_id='comparison_parameter', component_property='options'),
    Output(component_id='comparison_parameter', component_property='style'),
    Input(component_id='comparison_site', component_property='value'),
    Input(component_id="comparison_site_sql_id", component_property="children"))
def update_companion_parameters(site_value, site_sql_id):
    if site_value == '0':
        return [{'label': '0', 'value': '0'}], {'display': 'block'}
    if site_value != '0':
        # NEW CONN STRING
        #try:
        with sql_engine.begin() as conn:
                Parameters = pd.read_sql_query(f"SELECT {Available_Parameters} from {config['parameters']['parameter_table']} WHERE G_ID = '{site_sql_id}' AND STATUS = 'Active';", conn)
        if "WaterTemp" in Parameters.columns:
            Parameters = Parameters.rename(
            columns={"WaterTemp": "water_temperature"})
            # vlist.set_index('SITE_CODE', inplace=True)
        Parameters = Parameters.loc[:, (Parameters != 0).any(axis=0)]
            # returns list of columns for dropdown
        Parameters = Parameters.columns.values.tolist()

        return [{'label': i, 'value': i} for i in Parameters], {'display': 'block'}
        #except: # if its an external site use this
        #    comparison_list = pd.read_csv('external_comaprison_sites.csv', skipinitialspace=True)
        #    search = comparison_list.loc[comparison_list['site'].isin([site_value])]
        #    Parameters = search.iloc[0, 2]
        #    Parameters = Parameters.tolist() # each sites has one parameter for external sites
        #    print("external parms: ",Parameters)
        #    return [{'label': i, 'value': i} for i in Parameters], {'display': 'block'}

# Ratings
@app.callback(
    # Output('Ratings', 'value'),
    Output('Ratings', 'options'),
    Output('Ratings', 'style'),
    Input('Parameter', 'value'),
    Input('site_sql_id', 'children'))
def Select_Ratings(Parameter_value, site_sql_id):
   #print("select ratings")
    if Parameter_value == "FlowLevel":
        with sql_engine.begin() as conn:
            Ratings = pd.read_sql_query(f"select RatingNumber as rating_number from tblFlowRatings WHERE G_ID = '{site_sql_id}' GROUP BY RatingNumber ORDER BY RatingNumber DESC;", conn)
        Ratings = Ratings['rating_number'].values.tolist()
    
        return [{'label': i, 'value': i} for i in Ratings], {'display': 'block'}
    else:
        return [{'label': "NONE", 'value': "NONE"}], {'display': 'none'}

# Pick Range, query existing data in SQL Database
@app.callback(
    # Output('output-container-date-picker-range', 'children'),
    Output('import_data', 'data'),
    Output('select_range', 'startDate'),  # startDate is a dash parameter
    Output('select_range', 'endDate'),
    # Input(component_id='Barometer_Button', component_property='value'),
    Input('select_range', 'startDate'),  # startDate is a dash parameter
    Input('select_range', 'endDate'),
    Input('site', 'value'),
    Input("site_sql_id", "children"),
    Input(component_id='Parameter', component_property='value'),
    Input('datatable-upload', 'contents'),
    Input('datatable-upload', 'filename'),
    Input('HEADER_ROWS', 'value'),
    Input('FOOTER_ROWS', 'value'),
    Input('TIMESTAMP_COLUMN', 'value'),
    Input('DATA_COLUMN', 'value'),
    Input('Select_Data_Source', 'value'),
    Input('Barometer_Button', 'value'),
    Input('Available_Barometers', 'value'),
    # Input(component_id='Available_Barometers', component_property='value'),
    # State('Barometer_Button', 'value')
)
def update_daterange(startDate, endDate, site, site_sql_id, parameter, contents, filename, header_rows, footer_rows, timestamp_column, data_column, data_source, barometer_button, available_barometers):
    # Call and process incoming data
    # if there is no csv file (northing in contents) query data from sql server
    # contents = holder of imported data, when data is being imported contents = True
    # data_source is if we are quering the server vs importing data, select_data_source False is file import True is SQL query
    # program corrects off the "data" column but other values are pulled
    # if contents is None:  # nothin in datatable upload
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    from import_data import sql_import
    if data_source == True and startDate != '' and endDate != '' and site != '0' and parameter != '0' and 'select_range' in changed_id:  # query sql server
        df = sql_import(parameter, site_sql_id, (pd.to_datetime(startDate).to_pydatetime()) - timedelta(hours=(7)), (pd.to_datetime(endDate).to_pydatetime()) - timedelta(hours=(7))) #convert start/end date from utc to pdt
        return  df.to_json(orient="split"), startDate, endDate

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
        if barometer_button == "Baro" and data_source is False:  # data source false is file import
            search = Available_Sites.loc[Available_Sites["SITE_CODE"].isin([available_barometers])]
            if search.empty:
                #df = df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], True
                return pd.DataFrame().to_json(orient="split"), startDate, endDate
            else:

                B_ID_Lookup = search.iloc[0, 1]

               
                # THIS IS DUMB, ITS A PLACHHOULDER there needs to be a formula to convert wl feet
                if df['data'].mean() < 30: # if under 30 assumed to be psi
                    df['data'] = round((df['data']*68.9476), 3) # convert to millibar

                if df['data'].mean() > 999: # assumed to be millibar
                    df['data'] = df['data'] # millibar
                  
                
                barometer_query = sql_import("barometer", B_ID_Lookup, df['datetime'].min(), df['datetime'].max()) # fx converts to PST and out of PST
                barometer_query = barometer_query.rename(columns = {"corrected_data": "barometer_data"})
                # resample barometer to 5 minutes
                barometer_query = barometer_query.set_index("datetime").resample('5T').interpolate(method='linear').reset_index(level=None, drop=False)

                df = pd.merge(df,barometer_query[['datetime', "barometer_data"]],on=['datetime'])
                
                df['data'] = ((df['data']-df["barometer_data"]) * 0.0335).round(3)
                # Also a shotty conversion using a standard pressure
                
                df = df.drop(['barometer_data'], axis=1)
        
       
        return  df.to_json(orient="split"), df['datetime'].min() + timedelta(hours=(7)), df['datetime'].max() + timedelta(hours=(7))
    else:
           
            return pd.DataFrame().to_json(orient="split"), startDate, endDate
     

# select data level
@app.callback(
    Output('select_data_level', 'style'),
    Input('Select_Data_Source', 'value'))
def data_level(Select_Data_Source):
    # file import
    if Select_Data_Source is False:
        return {'display': 'none'}
    # database query - data
    elif Select_Data_Source is True:
        return {'display': 'block'}

# FIELD Observations and Data
@app.callback(
    Output("Initial_Data_Correction", "data"),
    Output("Initial_Data_Correction", "columns"),
    Input('site', 'value'),
    Input('Parameter', 'value'),
    Input("import_data", "data"),
    Input(component_id="site_sql_id", component_property="children"),
    Input('select_range', 'startDate'),  # startDate is a dash parameter
    Input('select_range', 'endDate'),
    Input('Select_Data_Source', 'value'),
    Input('data_interval', 'children'),
    )
def get_observations(site, parameter, barometer_corrected_data, site_sql_id, startDate, endDate, select_data_source, data_interval):
    ''''Takes data in question and finds cooresponding observations
    returns data, with columns for observations does not trim or cut
    send to correct_data'''
    
    
    def merge_observations(data, observations):
        if not observations.empty:
            min_series = (60/len((data["datetime"].dt.strftime('%M').copy()).drop_duplicates()))/2

            data = pd.merge_asof(data.sort_values('datetime'), 
                     observations.sort_values('datetime'), 
                     on='datetime', 
                     tolerance=pd.Timedelta(f"{min_series}m"), 
                     direction="nearest")
           
        else:
            data["observation_stage"] = np.nan
        return data
    
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    data_check = pd.read_json(barometer_corrected_data, orient="split")
    #print("data check")
    #print(data_check)
    #if select_data_source is False and not data_check.empty: 
    #    startDate = data_check['datetime'].min() + timedelta(hours=(7)) # since the dash timedate is in utc we need to convert logger pdt to utc for function
    #    endDate = data_check['datetime'].max()  + timedelta(hours=(7)) # since the dash timedate is in utc we need to convert logger pdt to utc for function
    if not data_check.empty and startDate != '' and endDate != '' and site != '0' and parameter != '0' and ('site' in changed_id or 'Parameter' in changed_id or 'select_range' in changed_id or 'data_interval' in changed_id or select_data_source is False): # eventually get rid of data check
        
        try:
            data = pd.read_json(barometer_corrected_data, orient="split")
            from data_cleaning import fill_timeseries
            data = fill_timeseries(data, data_interval) # this doesn do too much anymore
            from import_data import get_observations_join
            observations = get_observations_join(parameter, site_sql_id, (pd.to_datetime(startDate).to_pydatetime()) - timedelta(hours=(7)), (pd.to_datetime(endDate).to_pydatetime()) - timedelta(hours=(7))) # convert start/end date from utc to pdt
         
            #field_observations = get_observations()
            df = merge_observations(data, observations)
            if parameter == "FlowLevel" or parameter == "discharge":
                #df = get_parameter_observation(data, field_observations, parameter_value)
                df.rename(columns={"parameter_observation": "q_observation"}, inplace=True)
            if parameter == "Conductivity" or parameter == "conductivity" or parameter == "water_temperature":
                #df = get_parameter_observation(data, field_observations, parameter_value)
                if "observation_stage" in df.columns:
                    df.drop(columns=["observation_stage"], inplace=True)

            # add per / post data
            from import_data import sql_import
            df_pre =  sql_import(parameter, site_sql_id, df['datetime'].min() - timedelta(hours=12), df['datetime'].min() - timedelta(hours=0)) #fx uses pdt  parm, start, end
            df_pre[f"{config[parameter]['observation_class']}"] = df_pre["corrected_data"] ### corrected data (water level)
            if parameter == "discharge":
                df_pre["q_observation"] = df_pre["discharge"]
        
            df_post = sql_import(parameter, site_sql_id, df['datetime'].max() - timedelta(hours=0), df['datetime'].max() + timedelta(hours=12)) #convert start/end date from utc to pdt
            df_post[f"{config[parameter]['observation_class']}"] = df_post["corrected_data"] ### corrected data (water level)
            if parameter == "discharge":
                df_post["q_observation"] = df_post["discharge"]
            ### need to add q observation here
            df = pd.concat([df_pre, df_post, df])
            df = df.sort_values(by='datetime', ascending=False)
            df = fill_timeseries(df, data_interval) # this doesn do too much anymore
            return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]
        except ValueError as e:
             print(e)
             data = pd.read_json(barometer_corrected_data, orient="split")
             return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]
        

    else:
            return [{}], [] # eventually replace with return dash.no_update
        


@app.callback(
#@app.long_callback(
    # CORRECT FIELD data to FIELD OBSERVATIONS
    Output("paragraph_id", "children"),
    Output("realtime_update_info", "children"),
    Output("Corrected_Data", "data"),
    Output("Corrected_Data", "columns"),
    #Output("Corrected_Data", "style_data_conditional"),
    # returning a blank df cant have deletable rows
    Output("Corrected_Data", "row_deletable"),

    Input('data_interval', 'children'),
    Input("realtime_update", "value"),
    Input("run_job", "n_clicks"),
    Input('interpolate_button', 'n_clicks'),
    Input('dry_indicator_button', 'value'),
    Input('set_dry_indicator_warning_button', "n_clicks"),
    Input('select_range', 'startDate'),  # startDate is a dash parameter
    Input('select_range', 'endDate'),
    Input('checklist', 'value'),
    Input('select_data_level', 'value'),
    Input('site', 'value'),
    Input('site_sql_id', 'children'),
    Input('Parameter', 'value'),

    Input('comparison_site', 'value'),
    Input('comparison_site_sql_id', 'children'),
    Input('comparison_parameter', 'value'),
    Input('Ratings', 'value'),
    
    
    Input("Initial_Data_Correction", "data"),
    Input("Initial_Data_Correction", "columns"),
    Input("Corrected_Data", "data"),
    State("Corrected_Data", "data"),
    State("Corrected_Data", "columns"),)
    # disable outputs while callback is running
    #This example uses running to set the disabled property of the button to True while the callback is running, and False when it completes
   # manager=long_callback_manager,)
    
def correct_data(data_interval, realtime_update, run_job, interpolate_button, dry_indicator_button, set_dry_indicator_warning_button, startDate, endDate, checklist, data_level, site, site_sql_id, Parameter_value, comparison_site, comparison_site_sql_id, comparison_parameter, ratings_value,  Initial_Data_Correction_row, Initial_Data_Correction_column, row, Corrected_Data_row, Corrected_Data_column):

    '''Takes dataframe of data and observations from function: get_observations '''

    from data_cleaning import style_formatting
    from data_cleaning import data_conversion
    from discharge import discharge_calculation, finalize_discharge_dataframe
    from data_cleaning import reformat_data, parameter_calculation

    realtime_update_info = "" # placehoulder, I go back and forth bethween pausing the run until data is present and using placeholders
    callback_state = ""
    
    
    # field observaions function returns the field observation (observation_stage)
    # and parameter_observation where applicable
    # you need the stage observation to get the parameter observation
    try:
        observation = config[Parameter_value]["observation_class"]
    except:
        observation = ""
    # on initial open we can select data vs corrected data
    # this is a bit hacky.  We are correcting off of the data column
    # but we can change the data in the data column on initial import

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    # if there is no data return a blank dataframe
    if pd.DataFrame(Initial_Data_Correction_row).empty:
        return dash.no_update
        # if there is data to look at read the datatable
    else:
        # first calculate parameter
        dff = pd.DataFrame(row) # the data in the table
        if dff.empty or 'select_range' in changed_id or "Parameter" in changed_id or "site" in changed_id or "Initial_Data_Correction" in changed_id: # could probs just do initialdata correction
            df_raw = pd.DataFrame(Initial_Data_Correction_row)
            
        #if not dff.empty:
        else:
            df_raw = pd.DataFrame(row)
        
        df_raw = reformat_data(df_raw) 
        df_raw = data_conversion(df_raw, Parameter_value)
        if (realtime_update is True or 'run_job' in changed_id):
            df_raw = parameter_calculation(df_raw, observation, data_level)

        if Parameter_value != 'FlowLevel':
            desired_order = ["datetime", "data", "corrected_data", "observation", "observation_stage", "offset", "estimate", "warning", "comparison", "comments"] # observation and observation_stage are kinda redundent at some point and should be clarified
            # Filter out columns that exist in the DataFrame
            existing_columns = [col for col in desired_order if col in df_raw.columns]
            # Reorder the DataFrame columns
            df_raw = df_raw[existing_columns]
        
            # calculate discharge
        if Parameter_value == 'FlowLevel':
            if not 'discharge' in df_raw.columns:
                df_raw['discharge'] = 0
            if not 'q_observation' in df_raw.columns:
                df_raw['q_observation'] = 'nan'
                
            if ratings_value == 'NONE':  # If there is no discharge calculate a discharge of zero
                df_raw["q_offset"] = "nan"
                df_raw["precent_q_change"] = "nan"
                df_raw["rating_number"] = "NONE"
            else:
                if (realtime_update is True or 'run_job' in changed_id):
                    df_raw = discharge_calculation(df_raw, ratings_value, site_sql_id)
                        
                df_raw = finalize_discharge_dataframe(df_raw)

            ### ADD comparison site
        from data_cleaning import add_comparison_site
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            # and 'comparison_site' in changed_id  - this seems to be less stable
        if "comparison_site" in checklist and comparison_parameter != "0":
                if "comparison" in df_raw.columns:
                    df_raw.drop(columns=['comparison'], inplace=True)
                df_raw = add_comparison_site(comparison_site_sql_id, comparison_parameter, df_raw)
        if 'comparison_site' in changed_id and "comparison_site" in checklist and comparison_site.startswith("USGS"):
                if "comparison" in df_raw.columns:
                    df_raw.drop(columns=['comparison'], inplace=True)
                from import_data import usgs_data_import
                df_comp = usgs_data_import(comparison_site_sql_id, (pd.to_datetime(startDate)) - timedelta(hours=(7)), (pd.to_datetime(endDate)) - timedelta(hours=(7))) # convert start/end date from utc to pdt
                df_raw = df_raw.merge(df_comp, on="datetime", how = "outer")
        if ("comparison_site" not in checklist) and "comparison" in df_raw.columns:
            df_raw.drop(columns=['comparison'], inplace=True)

        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'interpolate_button' in changed_id:
            from interpolation import cache_comparison_interpolation
            print("run interpolation")
            df_raw = cache_comparison_interpolation(df_raw, site, site_sql_id, Parameter_value, df_raw["datetime"].min(), df_raw["datetime"].max()) # convert start/end date from utc to pdt)
        if 'data_interval' in changed_id:
            from interpolation import resample
            df_raw = resample(df_raw, data_interval)
           
        df_raw = df_raw.sort_values(by='datetime', ascending=False)
        df_raw["datetime"] = df_raw["datetime"].dt.strftime('%Y-%m-%d %H:%M') # allows dash graph to display datetime without the "Td"

        ## dry indicator
        if dry_indicator_button is True:
            
            df_raw['dry_indicator'] = (df_raw['data'] - np.mean(df_raw['data'])) - abs((df_raw['data'] - np.mean(df_raw['data'])))
            df_raw.loc[df_raw['dry_indicator'] == 0, "dry_indicator"] = 1
            df_raw.loc[df_raw['dry_indicator'] < 0, "dry_indicator"] = 0 # dry

            df_raw.loc[df_raw['dry_indicator'] == 1, "dry_indicator"] = " "
            df_raw.loc[df_raw['dry_indicator'] == 0, "dry_indicator"] = "dry indicator"# dry

            changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
            if 'set_dry_indicator_warning_button' in changed_id:
                df_raw.loc[df_raw["dry_indicator"] == "dry indicator", "warning"] = 1
        if dry_indicator_button is False and 'dry_indicator' in df_raw.columns:

            df_raw.drop(columns=['dry_indicator'], inplace = True)


        if realtime_update is False:
            realtime_update_info = "realtime updating  - paused - "
        if realtime_update is True:
            realtime_update_info = "realtime updating"
        
        return [f"{realtime_update_info}"],[f"{callback_state}"], df_raw.to_dict('records'), [{"name": i, "id": i} for i in df_raw.columns], True
  
@app.callback(
        Output(component_id='graph_output', component_property='children'),
        Input("Corrected_Data", "data"),
        Input('site', 'value'),
        Input('site_sql_id', 'children'),
        Input('Parameter', 'value'),

        Input('comparison_site', 'value'),
        #Input('comparison_site_sql_id', 'children'),
        Input('comparison_parameter', 'value'),
        Input('Ratings', 'value'),
        Input("data_axis","value"),
        Input("corrected_data_axis","value"),
        Input("derived_data_axis","value"),
        Input("observation_axis","value"),
        Input("comparison_axis","value"),
        Input("primary_min", "value"),
        Input("primary_max", "value"),
        Input("secondary_min", "value"),
        Input("secondary_max", "value"),
        
)

def graph(df, site, site_sql_id, parameter, comparison_site, comparison_parameter, rating, data_axis, corrected_data_axis, derived_data_axis, observation_axis, comparison_axis, primary_min, primary_max, secondary_min, secondary_max):
    from data_cleaning import reformat_data, parameter_calculation
    from graph_2 import cache_graph_export
    df = pd.DataFrame(df)
    if df.empty:
        return dash.no_update
    else:
        df = reformat_data(df) 
        
        fig = html.Div(dcc.Graph(figure = go.Figure()), style = {'width': '100%', 'display': 'inline-block'})
        fig = cache_graph_export(df, site_sql_id, site, parameter, comparison_site, comparison_parameter, data_axis, corrected_data_axis, derived_data_axis, observation_axis, comparison_axis, primary_min, primary_max, secondary_min, secondary_max)

    return fig

@app.callback(
    dash.dependencies.Output('upload_data_children', 'children'),
    [dash.dependencies.Input('upload_data_button', 'n_clicks')],
    Input('Corrected_Data', 'data'),
    Input('site', 'value'),
    Input('site_sql_id', 'children'),
    Input('Parameter', 'value'),
    Input('comparison_site', 'value'),
        #Input('comparison_site_sql_id', 'children'),
    Input('comparison_parameter', 'value'),
    Input('Ratings', 'value'),
    Input("data_axis","value"),
    Input("corrected_data_axis","value"),
    Input("derived_data_axis","value"),
    Input("observation_axis","value"),
    Input("comparison_axis","value"),
    Input("primary_min", "value"),
    Input("primary_max", "value"),
    Input("secondary_min", "value"),
    Input("secondary_max", "value"),
    )
def run_upload_data(n_clicks, df, site, site_sql_id, parameter, comparison_site, comparison_parameter, rating, data_axis, corrected_data_axis, derived_data_axis, observation_axis, comparison_axis, primary_min, primary_max, secondary_min, secondary_max):
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
            end_date = df["datetime"].max().date()
            save_fig(df, site, site_sql_id, parameter, comparison_site, comparison_parameter, rating, data_axis, corrected_data_axis, derived_data_axis, observation_axis, comparison_axis, end_date, primary_min, primary_max, secondary_min, secondary_max)
            from sql_upload import full_upload

            desired_order = ["datetime", "data", "corrected_data", "discharge", "estimate", "warning"] # observation and observation_stage are kinda redundent at some point and should be clarified
            # Filter out columns that exist in the DataFrame
            existing_columns = [col for col in desired_order if col in df.columns]
            # Reorder the DataFrame columns
            df = df[existing_columns]
            # rename parameters
            if parameter == "Conductivity" or parameter == "conductivity":
                parameter = "Conductivity"
            if parameter == "water_level" or parameter == "LakeLevel":
                parameter = "water_level"
            if parameter == 'groundwater_level' or parameter == "Piezometer":
                parameter = "groundwater_level"
            if parameter == "discharge" or parameter == "FlowLevel":
                parameter = "discharge"
            full_upload(df, parameter, site_sql_id, 7)
            
            from workup_notes import workup_notes_main

            workup_notes_main(notes_df, parameter, site_sql_id, site)
            
            result = "  uploaded"

            return result
    else:
        return dash.no_update

@app.callback(
    dash.dependencies.Output('export_data_children', 'children'),
    [dash.dependencies.Input('export_data_button', 'n_clicks')],
    Input('Corrected_Data', 'data'),
    Input('site', 'value'),
    Input('site_sql_id', 'children'),
    Input('Parameter', 'value'),
    Input('comparison_site', 'value'),
        #Input('comparison_site_sql_id', 'children'),
    Input('comparison_parameter', 'value'),
    Input('Ratings', 'value'),
    Input("data_axis","value"),
    Input("corrected_data_axis","value"),
    Input("derived_data_axis","value"),
    Input("observation_axis","value"),
    Input("comparison_axis","value"),
    Input("primary_min", "value"),
    Input("primary_max", "value"),
    Input("secondary_min", "value"),
    Input("secondary_max", "value"),
    )
def run_export_data(n_clicks, df, site, site_sql_id, parameter, comparison_site, comparison_parameter, rating, data_axis, corrected_data_axis, derived_data_axis, observation_axis, comparison_axis, primary_min, primary_max, secondary_min, secondary_max):
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
           
            df_export = df.set_index('datetime').copy()
            #end_date = df["datetime"].max().date().strftime("%Y_%m_%d")
            end_date = df["datetime"].max().date()
          
            df_export.to_csv("W:/STS/hydro/GAUGE/Temp/Ian's Temp/" +
                str(site)+"_"+str(parameter)+"_"+str(end_date)+".csv")
           

            save_fig(df, site, site_sql_id, parameter, comparison_site, comparison_parameter, rating, data_axis, corrected_data_axis, derived_data_axis, observation_axis, comparison_axis, end_date, primary_min, primary_max, secondary_min, secondary_max)

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