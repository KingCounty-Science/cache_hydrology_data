#from import_data import get_observations_join
#from import_data import  get_rating_points_and_list

import pandas as pd

import pandas as pd
import datetime as dt
import configparser
import os
import numpy as np
import base64
import io
import pyodbc
import configparser
import dash
import dash_ag_grid as dag
import dash_daq as daq
from dash import html
from dash.dependencies import Input, Output, State
from dash import dcc
#from dash import html
from dash import dash_table
import pandas as pd
import dash_daq as daq
import numpy as np
from sqlalchemy import create_engine
import plotly.graph_objs as go
import datetime as dt
# long call back 
# https://dash.plotly.com/long-callbacks
## launch a new web browser
from web_browser import launch_web_broswer
# launch_web_broswer()
import dash_bootstrap_components as dbc
from waitress import serve
from scipy.signal import find_peaks, find_peaks_cwt
#import dash_core_components as dcc
from dash import dcc
from dash import html
from dash import State

import plotly.io as pio
pio.kaleido.scope.default_format = "svg"
from plotly.subplots import make_subplots
import numpy as nppython 
import datetime as dt
import plotly.graph_objects as go
# gget discharge info
import base64
import datetime as dt
from datetime import timedelta
from datetime import datetime
import plotly.express as px
import pyodbc
import configparser
import pandas as pd
from datetime import date

from plotly.subplots import make_subplots
from sqlalchemy import create_engine
import urllib

from web_browser import launch_web_broswer
#launch_web_broswer()

config = configparser.ConfigParser()
config.read('gdata_config.ini')

#from web_browser import launch_web_broswer
#launch_web_broswer()
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from dash import Dash, Input, Output, callback, dash_table

app = Dash(__name__)

data_type = "sql" # sql/example

# define site_id (58A)
parameter = "discharge"
startDate = pd.to_datetime("1989-01-01")
endDate = pd.to_datetime(pd.to_datetime('today'))

if data_type == "sql": # sql
    from import_data import get_sites
    sites = get_sites("discharge")
    available_sites = sites["site_id"].values.tolist()

if data_type == "example":# example data
    sites = pd.read_csv("example_data/sites.csv")
    available_sites = sites['site_id'].to_list()

##from import_data import get_all_observations
#observations = pd.read_csv("example_data/observations.csv")


app.layout = html.Div([
    html.H4('Interactive scatter plot with Iris dataset'),


    html.Div([
        dcc.Dropdown(id='site_id', options=[{'label': i, 'value': i} for i in available_sites], value=""),
        dcc.Dropdown(id='rating_list', style={'width': '200px'}),
        #html.Button('save rating', id='save_rating_button', n_clicks=0),
    ], style={'display': 'flex', 'align-items': 'center'}),
    #### generic pop up 
    
    #dcc.Dropdown(id='site_id', options=[{'label': i, 'value': i} for _ in available_sites], value = "", style={'display': 'block'}), 
    #dcc.Dropdown(id='rating_list'),
    
    html.P("Filter by rating:"),
    dcc.RangeSlider(id='range-slider', marks=None, tooltip={"placement": "bottom","always_visible": True,},),
        
    #dcc.Graph(id="scatter-plot"),
    html.Div([
    html.Div(daq.ToggleSwitch(id='show_obs', label="Show Observation Number", value=False)), 
    html.Div(daq.ToggleSwitch(id='graph_offset', label="Graph with GZF offsetted stage", value=True)),
   
     ], style={'display': 'flex', 'align-items': 'center'}),


    html.Div(id="scatter-plot"),
    dcc.Store(id='obs', storage_type='memory'),

    #dash_table.DataTable(id="mask_table", style={'display': 'none'}),
   
    
    
    dag.AgGrid(
        id="data-grid",
        #rowData=df.to_dict("records"),
        #columnDefs=[{"field": col, "editable": True} for col in df.columns],
        defaultColDef={"editable": True, "resizable": True, "sortable": True},
        dashGridOptions={
                "undoRedoCellEditing": True,
                "undoRedoCellEditingLimit": 20,
                "editType": "fullRow",
                "animateRows": False,
                "suppressScrollOnNewData": True,
                #"rowSelection": "multiple",  # Allow row selection
            },
    
    ),

    dash_table.DataTable([], id='tbl_addition', editable=True, row_deletable=True),

    html.Div([
         html.Div([
            html.Button('Add Row', id='editing-rows-button', n_clicks=0),
            html.Label("        new rating offset: "),
            dcc.Input(id="new_rating_offset",type="number",value = 0),
            html.Label("        offset existing rating: "),
            dcc.Checklist(id='offset-toggle', options=[{'label': 'offset_rating', 'value': 'offset_rating'}], value=[], labelStyle={'display': 'inline-block'} ),
            html.Label("        new rating name: "),
            dcc.Input(id="new_rating_name", type="text", value=""),
            html.Label(" enter notes for rating: "),
            dcc.Input(id="rating_notes", type="text", value="Ok Rating"),
            html.Label("  -  "),
            html.Button('save rating', id='save_rating_button', n_clicks=0),
            html.Label("  -  "),
            html.Button('delete selected rating', id='delete_selected_rating', n_clicks=0),

            dcc.ConfirmDialog(id='popup',message='' ),
        ], style={'display': 'flex'}),





        html.Label("Rating Point Calculator"),
        html.Div([
                    html.Label("point a    "),
                    html.Label("stage a: "),
                    dcc.Input(id="stage_a",type="number",placeholder="stage a"),
                    html.Label("discharge a: "),
                    dcc.Input(id="discharge_a",type="number",placeholder="discharge a"),
                    html.Label("point b    "),
                    html.Label("stage b: "),
                    dcc.Input(id="stage_b",type="number",placeholder="stage a"),
                    html.Label("discharge a: "),
                    dcc.Input(id="discharge_b",type="number",placeholder="discharge a"),
                    html.Label("solve for    "),
                    html.Label("stage (x): "),
                    dcc.Input(id="stage_x",type="number",placeholder="stage x"),
                    html.Label("discharge (y): "),
                    dcc.Input(id="discharge_x",type="number",placeholder="discharge x"),
                ], style={'display': 'flex'}),
        html.Div([
            html.Label("solver output: "),
            html.Div(id='calculation_output'),
        ], style={'display': 'flex'}),
    ]),
    dcc.Store(id='site_sql_id'),
    dcc.Store(id = 'rating_points'),
    dcc.Store(id = 'new_rating'),
    #dcc.Input(id='site_id', value = "58A"),
])

# min=obs["observation_number"].min(), max=obs["observation_number"].max(),

## get sql_id


# Select file source
@app.callback(
    Output('site_sql_id', 'data'),
    Output("rating_points", "data"),
    Output('rating_list', 'options'),
    Output('range-slider', 'min'),
    Output('range-slider', 'max'),
    Output('range-slider', 'value'),
    Output('obs', 'data'),
    Input('site_id', 'value'))


def update_sql_id(site_id):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if site_id != "":
        
        site_sql_id  = sites["site_sql_id"].loc[sites["site_id"] == site_id].values[0]

        if data_type == "sql":
            from import_data import get_rating_points_and_list, get_observations_join
            rating_points, rating_list = get_rating_points_and_list(site_sql_id)
            rating_list.append('new rating')
            obs = get_observations_join(parameter, site_sql_id, startDate, endDate)

        if data_type == "example":
            rating_points = pd.read_csv("example_data/rating_points.csv")
            rating_points = rating_points.loc[rating_points['site_id'] == site_id]
            rating_points = rating_points.set_index("rating")
            rating_list = list(rating_points.index.unique())
            rating_list.append('new rating')

            observations = pd.read_csv("example_data/observations.csv")
            obs = observations.loc[observations["site_id"] == site_id].copy()
        obs["observation_number"] = obs["observation_number"].round(2)

        min_obs = obs["observation_number"].min()
        max_obs = (obs["observation_number"].max())+1
        obs = obs.to_json(date_format='iso', orient='split')
        rating_points = rating_points.to_json(date_format='iso', orient='split')
        return site_sql_id, rating_points, rating_list, min_obs, max_obs, [min_obs, max_obs], obs
    else:
        return dash.no_update

@callback(
    Output('tbl_addition', 'data'),
  
    Output('new_rating', 'data'),
    Input("new_rating_offset", "value"),
    Input('editing-rows-button', 'n_clicks'),
    Input("rating_points", "data"),
    Input('rating_list', 'value'),
    Input("obs", "data"),
    Input('tbl_addition', "data"),
    Input("site_id", "value"),
    State('tbl_addition', 'data'),
    State('tbl_addition', 'columns'),
    Input('offset-toggle', 'value'),
    prevent_initial_call=True)
 


def add_row(new_rating_offset, n_clicks, rating_points, rating,  obs, edited_data, site_id, rows, columns, offset_rating):
    new_rating = pd.DataFrame({'stage_rating': [], "discharge_rating": [], "gzf": []}) # blank placehoulder
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

   
#def show_offset_status(selected):
    if "n_clicks" in changed_id:
        # Add a blank row to rows if the last row is empty
        edited_data.append({c['id']: np.nan for c in columns})
        #edited_data.append({**{c['id']: np.nan for c in columns}, "entered_offset": np.nan})
      

    if edited_data and not pd.DataFrame(edited_data).empty:
        df = pd.DataFrame(edited_data)
        if rating != "new rating":
            df = pd.DataFrame(edited_data)
            obs = pd.read_json(obs, orient='split')
            rating_points = pd.read_json(rating_points, orient='split')
            rating_points = rating_points.loc[rating_points.index == rating]

            nulls_df = df[df["observation_stage"].isna() & df["parameter_observation"].isna() & df["entered_offset"].isna()].copy()


            df["observation_stage"] = pd.to_numeric(df["observation_stage"], errors="ignore")
            df["parameter_observation"] = pd.to_numeric(df["parameter_observation"], errors = "ignore")
            df["stage_offset"] = pd.to_numeric(round(df["observation_stage"]- (rating_points["gzf"].loc[rating_points.index == rating]).mean(), 2), errors = "ignore")

            rating_points["stage_rating"] = round(pd.to_numeric(rating_points["stage_rating"], errors="ignore"), 2)
            rating_points["discharge_rating"] = round(pd.to_numeric(rating_points["discharge_rating"], errors="ignore"), 2)
            rating_points["gzf"] = round(pd.to_numeric(rating_points["gzf"], errors="ignore"),2)

            desired_order = ["datetime", "observation_number", "observation_stage", "parameter_observation", "entered_offset", "comments", "stage_offset"] # observation and observation_stage are kinda redundent at some point and should be clarified
            # Filter out columns that exist in the DataFrame
            existing_columns = [col for col in desired_order if col in df.columns]
                # Reorder the DataFrame columns
            df = df[existing_columns]

            # returns discharge for given stage
            df_a = df.loc[df["observation_stage"].notna() & df["parameter_observation"].isna() & df["entered_offset"].isna()].copy()
            df_a = pd.merge_asof(df_a.sort_values("stage_offset").dropna(subset=["stage_offset"]), rating_points.loc[rating_points.index == rating], left_on= "stage_offset", right_on = "stage_rating",direction = "nearest")
            df_a["rating_offset"] = round(df_a["stage_offset"] - df_a['stage_rating'], 2)
            df_a["rating_precent_change"] = round((abs(df_a["stage_offset"] - df_a['stage_rating'])/df_a["stage_rating"])*100, 2)
           
            # returns stage for given discharge
            df_b = df.loc[df["observation_stage"].isna() & df["parameter_observation"].notna() & df["entered_offset"].isna()].copy()
            df_b = pd.merge_asof(df_b.sort_values("parameter_observation").dropna(subset=["parameter_observation"]), rating_points.loc[rating_points.index == rating], left_on= "parameter_observation", right_on = "discharge_rating")
            df_b["rating_offset"] = round(df_b["stage_offset"] - df_b['stage_rating'], 2)
            df_b["rating_precent_change"] = round((abs(df_b["stage_offset"] - df_b['stage_rating'])/df_b["stage_rating"])*100, 2)

            df_c = df.loc[df["observation_stage"].notna() & df["parameter_observation"].notna()].copy()
            df_c = pd.merge_asof(df_c.sort_values("parameter_observation").dropna(subset=["parameter_observation"]), rating_points.loc[rating_points.index == rating], left_on= "parameter_observation", right_on = "discharge_rating")
            df_c["rating_offset"] = round(df_c["stage_offset"] - df_c['stage_rating'], 2)
            df_c["rating_precent_change"] = round((abs(df_c["stage_offset"] - df_c['stage_rating'])/df_c["stage_rating"])*100, 2)

            # returns discharge for given stage and offset
            df_d = df.loc[df["observation_stage"].notna() & df["parameter_observation"].isna() & df["entered_offset"].notna()].copy()
            df_d["stage_offset"] = df_d["stage_offset"].astype(float) + df_d["entered_offset"].astype(float)
            #df_d["stage_offset"] = df_d["stage_offset"] + df_d["entered_offset"]
            df_d = pd.merge_asof(df_d.sort_values("stage_offset").dropna(subset=["stage_offset"]), rating_points.loc[rating_points.index == rating], left_on= "stage_offset", right_on = "stage_rating",direction = "nearest")
            df_d["rating_offset"] = round(df_d["stage_offset"] - df_d['stage_rating'], 2)
            df_d["rating_precent_change"] = round((abs(df_d["stage_offset"] - df_d['stage_rating'])/df_d["stage_rating"])*100, 2)
            df_d["parameter_observation"] = df_d["discharge_rating"]
            df_e = df.loc[df["observation_stage"].isna() & df["entered_offset"].notna()].copy()
      
            df = pd.concat([df_a, df_b, df_c, df_d, df_e, nulls_df]).sort_index()
               
            if 'offset_rating' in offset_rating:  # create a 'new rating' that is an offset of the existing rating
                print("offset rating")
                if df["observation_stage"].isna().all():
                #if df[["observation_stage", "parameter_observation"]].isna().all().all():
                #if df["parameter_observation"].isna().all():
                    new_rating = rating_points
                else:
                    df["stage_diff"] = df["stage_offset"] - df["stage_rating"]
                    #new_rating = pd.merge(df[["observation_stage", "stage_offset", "rating_offset", "stage_rating", "parameter_observation"]], rating_points, on = "stage_rating", how = "right")
                    new_rating = pd.merge(df[["observation_stage", "stage_rating", "stage_diff", "parameter_observation"]], rating_points, on = "stage_rating", how = "right")
                    
                    #print("new rating merge")
                    #print(new_rating)
                    #new_rating["stage_diff"] = new_rating["stage_diff"].interpolate(method="linear", limit_area="inside").round(2)###new_rating = pd.merge(df[["observation_stage", "stage_offset", "parameter_observation"]], rating_points, left_on = "parameter_observation", right_on = "discharge_rating", how = "right")
                    new_rating["stage_diff"] = (new_rating["stage_diff"].interpolate(method="linear", limit_area="inside").ffill().bfill()).round(2)
                    #print("new rating interp")
                    #print(new_rating)
                    new_rating["stage_rating"] = (new_rating["stage_rating"]+new_rating["stage_diff"]).round(2)
                    
                    new_rating = new_rating[["stage_rating", "discharge_rating", "gzf"]]
                
                    #print("new rating corrected stage")
                    #print(new_rating)
                    """ new_rating = pd.merge(new_rating[["stage_rating"]], rating_points, on = "stage_rating", how = "inner")
                    print("new rating final merge")
                    print(new_rating)
                    print("origional rating")
                    print(rating_points)"""
                    new_rating = new_rating.sort_values(by="stage_rating", ascending=True)
                    new_rating = new_rating.drop_duplicates(subset=["stage_rating"], keep = "first")
                    print("new rating")
                    print(new_rating)
            edited_data = df.to_dict('records')
            print("edited data")
            print(df)
        # generate new rating
        if rating == "new rating" and df[df["observation_stage"].notna()].shape[0] > 1:
            df = pd.DataFrame(edited_data)
        
            gzf = new_rating_offset
         
            
            df = df.astype('float64', copy=True, errors='ignore')
            df["observation_stage"] = pd.to_numeric(df["observation_stage"])
            df["observation_stage"] = round(df["observation_stage"], 2)
            df["parameter_observation"] = pd.to_numeric(df["parameter_observation"])
            df["parameter_observation"] = round(df["parameter_observation"], 2)
            df.sort_values(by=['observation_stage'], inplace = True)
            

            min = (df['observation_stage'].min()).round(2)
            max = (df['observation_stage'].max()+0.01).round(2) # + 0.01
           
            # Generate the sequence of values from min to max with 0.01 increments
            new_rating = np.arange(min, max, 0.01)
            
            
            # Create a new DataFrame with these values
            new_rating = pd.DataFrame({'stage_rating': new_rating})
            new_rating["stage_rating"] = pd.to_numeric(new_rating["stage_rating"])
            new_rating["stage_rating"] = round(new_rating["stage_rating"], 2)
           
            if "stage_rating" in df.columns:
                df = df.drop(columns="stage_rating")
            
            new_rating = pd.merge(new_rating, df, left_on = "stage_rating", right_on="observation_stage", how='outer')

            new_rating.sort_values(by=['stage_rating'], inplace = True)
           
            x = np.array(np.log(df["observation_stage"]))
            y = np.array(np.log(df["parameter_observation"]))
            #Construct the interpolation

            xnew = np.array(np.log(new_rating["stage_rating"]))
            ynew = np.interp(xnew, x, y)
           
            
            new_rating["discharge_rating"] = np.exp(ynew)
            new_rating['discharge_rating'] = new_rating['discharge_rating'].round(2)
            # linear does not work so well
            """#print(new_rating.dtypes)
            new_rating["discharge_rating"] = np.log(new_rating["parameter_observation"])
            new_rating["discharge_rating"] = new_rating["discharge_rating"].interpolate(method='linear')
            new_rating["discharge_rating"] = np.exp(new_rating["discharge_rating"])
            new_rating["discharge_rating"] = round(new_rating["discharge_rating"], 2)"""
            

            # slope
            new_rating = new_rating[["stage_rating", "discharge_rating", "gzf"]]
            new_rating.sort_values(by=['stage_rating'], inplace = True)
            #new_rating["stage_rating"] = new_rating["stage_rating"]-gzf # rating not calculated with gzf atm
            new_rating["stage_rating"] = new_rating["stage_rating"].apply(lambda x: x - gzf if pd.notnull(x) and pd.notnull(gzf) else np.nan)
            new_rating["gzf"] = gzf
            new_rating["rating"] = "new rating"
            new_rating.set_index('rating', inplace=True)
            new_rating = new_rating.sort_values(by="stage_rating", ascending=True)
            #df["stage_offset"] = df["observation_stage"] - gzf
            df["stage_offset"] = df["observation_stage"].apply(lambda x: x - gzf if pd.notnull(x) and pd.notnull(gzf) else np.nan)
            df["gzf"] = gzf
            
            df = df.sort_values(by="stage_rating", ascending=True)
            edited_data = df.to_dict('records')
            
        

    return edited_data, new_rating.to_json(date_format='iso', orient='split')

@callback(
    Output("popup", "displayed"),
    Output("popup", "message"),
    Input("save_rating_button", "n_clicks"),
    Input('delete_selected_rating', "n_clicks"),
    State("new_rating", "data"),
    State('rating_list', 'value'),
    State('new_rating_name', 'value'),
    State('rating_notes', 'value'),
    State("new_rating_offset", "value"),
    State('site_sql_id', 'data'),
)


def save_rating(save_rating_button, delete_selected_rating, new_rating, rating, new_rating_name, rating_notes, new_rating_offset, site_sql_id):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    
    if "save_rating_button" in changed_id:
        new_rating = pd.read_json(new_rating, orient='split')

        #if rating == "new rating" and new_rating.notna() and new_rating_name != 0:
        #if rating == "new rating" and not new_rating.empty and new_rating_name.strip() != "":
        if not new_rating.empty and new_rating_name.strip() != "":
            
            # rename for sql compatability
            new_rating.reset_index(drop=True, inplace=True)
            new_rating = new_rating.rename(columns={"stage_rating": "WaterLevel", "discharge_rating": "Discharge"})
            new_rating = new_rating.sort_values(by="WaterLevel", ascending=True)
            new_rating["RatingNumber"] = new_rating_name
            # site sql id
            new_rating["G_ID"] = site_sql_id
            new_rating["Marker"] = range(1, len(new_rating) + 1)
            new_rating = new_rating[["G_ID", "RatingNumber", "WaterLevel", "Discharge", "Marker"]]
            
            from sql_upload import upload_rating, upload_rating_notes, check_if_rating_exists
            # check if rating exists
            print("running upload first test if exists")
            count = check_if_rating_exists(site_sql_id, new_rating_name)
            if count > 0: # rating allready exists
                print("exists")
                return True, f"Rating {new_rating_name} allready exists, no upload triggered"

            else: # rating doesnt exist preceed
                print("Rating is new — safe to insert.")
                upload_rating(new_rating)
                rating_notes_df = pd.DataFrame([{"Rating_Number": new_rating_name ,"Offset": new_rating_offset, "AutoDTStamp": datetime.today().strftime("%m/%d/%Y"),"Notes": rating_notes}])
                upload_rating_notes(rating_notes_df)
                # Get today's date in DD/MM/YYYY format
                return True, f"Rating {new_rating_name} saved successfully!"
        else:
            return True, "No new rating to save; please change rating to 'new rating' enter rating points, and name the rating"
    elif "delete_selected_rating" in changed_id and rating != "new rating":
            print("delete selected rating")
            from sql_upload import delete_rating
            delete_rating(site_sql_id, rating)
            return True, f"Rating {rating} deleted"
    else:
        return False, "no message"
    


@app.callback(
        Output(component_id='calculation_output', component_property='children'),
        Input("stage_a", "value"),
        Input("discharge_a", "value"),
        Input("stage_b", "value"),
        Input("discharge_b", "value"),
        Input("stage_x", "value"),
        Input("discharge_x", "value"),
        
 )   

def calculate_points(stage_a, discharge_a, stage_b, discharge_b, stage_x, discharge_x):
    # calculate y=mx+b
    try:  # if two points are entered but no points to calculate
        stage_a = np.log(stage_a)
        discharge_a = np.log(discharge_a)

        stage_b = np.log(stage_b)
        discharge_b = np.log(discharge_b)

        m = (discharge_b-discharge_a)/(stage_b-stage_a)# stage = x    discharge = y # slope
        b = discharge_a-(m*stage_a) # b = y-mx

        # solve for missing
        try:
            # solve for y (discharge): y = mx+b
            stage_x = np.log(stage_x)
            new_discharge = (m*stage_x)+b
            new_discharge = np.exp(new_discharge).round(2)
        except:
            new_discharge = "not_calculated"
        
        try:
            #solve for x (stage): (y-b)/m = x
            discharge_x = np.log(discharge_x)
            new_stage = (discharge_x-b)/m
            new_stage = np.exp(new_stage).round(2)
        except:
            new_stage = "not calculated"

        # mid point
        try:
            if stage_a == stage_b and discharge_a != discharge_b: # find discharge between two poinst
                mid_discharge = np.exp(discharge_a) - ((np.exp(discharge_a) - np.exp(discharge_b))/2)
                mid_stage = np.exp(stage_a).round(2)
                mid_discharge = mid_discharge.round(2)
            elif discharge_a == discharge_b and stage_a != stage_b: # find stage between two points
                mid_stage = np.exp(stage_a) - ((np.exp(stage_a) - np.exp(stage_b))/2).round(2)
                mid_discharge = np.exp(discharge_a).round(2)
                #mid_stage = np.exp(mid_stage).round(2)
                #mid_discharge = np.exp(discharge_a).round(2)    


            elif stage_a != stage_b and discharge_a != discharge_b:
                mid_stage = np.log(np.exp(stage_a) - ((np.exp(stage_a)-np.exp(stage_b))/2))

           
                mid_discharge = (m*mid_stage)+b

                mid_stage = np.exp(mid_stage).round(2)
                mid_discharge = np.exp(mid_discharge).round(2)

        except:
            mid_stage = "x"
            mid_discharge = "y"

        # I was going to add an automatic solver for stage at 0.01 q however that requires the first two points and could get messed up

        return f"values calculated: slope (m): {np.exp(m).round(2)}  y-int (b): {np.exp(b).round(2)}, new stage (x): {new_stage},  new discharge (y): {new_discharge},  middle point (x-stage/y-discharge): (mid stage {mid_stage}, mid discharge {mid_discharge})"
    
    except Exception as e:
        return f"no values calulated - - {e}"

@app.callback(
    #Output("tbl", "data"),
    #Output("tbl", "columns"),
    [Output("data-grid", "rowData"), 
    Output("data-grid", "columnDefs")],
    Output("tbl_addition", "columns"),

    Input("rating_points", "data"),
    Input('rating_list', 'value'),
    Input("range-slider", "value"),
    Input("obs", "data"),
    Input("site_id", "value"))
def get_observations(rating_points, rating, slider_range, obs, site_id):
    """ takes all observations from obs and filters by range selection """
    if site_id != "":
        rating_points = pd.read_json(rating_points, orient='split')
        rating_points['stage_rating'] = round(rating_points['stage_rating'], 2)
        obs = pd.read_json(obs, orient='split')
        obs["comments"] = obs["comments"].str.strip()
        obs['observation_stage'] = round(obs['observation_stage'],2)
        obs['parameter_observation'] = round(obs['parameter_observation'],2)
        
        # reduce observations to range slider range
        low, high = slider_range
        
        mask = obs.loc[(obs['observation_number'] >= low) & (obs['observation_number'] <= high)].copy()
        
        nulls_mask = mask[mask["observation_stage"].isna()].copy()
        mask = mask[mask["observation_stage"].notna()].copy()
        mask["stage_offset"] = round(mask["observation_stage"] - (rating_points["gzf"].loc[rating_points.index == rating]).mean(), 2)
       
    
        mask = pd.merge_asof(mask.sort_values("parameter_observation").dropna(subset=["parameter_observation"]), rating_points.loc[rating_points.index == rating], left_on= "parameter_observation", right_on = "discharge_rating")
     
        #mask = mask.drop(columns="stage_rating")
        mask = mask.sort_values("observation_number")
            
        mask["gzf"] = round(mask["gzf"], 2)
        mask["rating_offset"] = round(mask["stage_offset"] - mask['stage_rating'], 2)
        mask["rating_precent_change"] = round(((mask["stage_offset"] - mask['stage_rating'])/mask["stage_offset"])*100, 2)
        mask['discharge_rating'] = round(mask['discharge_rating'], 2)
       
        if not nulls_mask.empty:
                mask = pd.concat([mask, nulls_mask]).sort_index()
        mask['selection'] = True
        # tbl, tbl columns   grid, grid, addition
        # mask.to_dict('records'),[{"name": i, "id": i} for i in mask.columns], 
        # columns for tbl addition
       
        columns = [{"name": i, "id": i} for i in mask.columns] + [{"name": "entered_offset", "id": "entered_offset"}]
        return mask.to_dict("records"), [{"field": col, "editable": True} for col in mask.columns], columns

    else:
        return dash.no_update
 
@app.callback(
    Output("scatter-plot", "children"), 
    
    Input("rating_points", "data"),
    Input('rating_list', 'value'),
    Input("range-slider", "value"),
    Input("obs", "data"),
    Input('tbl_addition', 'data'),
    #Input('tbl', 'data'),
    Input("data-grid", "rowData"),
    Input("site_id", "value"),
    Input("show_obs", "value"),
    Input("graph_offset", "value"),
    Input("new_rating", 'data'),
    Input('offset-toggle', 'value'),)
def graph(rating_points, rating, slider_range, obs, tbl_addition, data_grid, site_id, show_obs, graph_offset, new_rating, offset_rating):
    if site_id != "":
        tbl_addition = pd.DataFrame(tbl_addition)
        obs = pd.read_json(obs, orient='split')
        obs = obs.sort_values(by="datetime", ascending=True)
        obs["observation_stage"] = pd.to_numeric(obs["observation_stage"], errors="ignore")
        obs["parameter_observation"] = pd.to_numeric(obs["parameter_observation"], errors="ignore")
        obs['observation_age'] = (obs['datetime'].max() - obs['datetime']).dt.days
        gzf = 0
        #if rating != "new rating":
        #    gzf = round(rating_points["gzf"].loc[rating_points.index == rating].mean(), 2)
        
        if rating == "new rating":
           rating_points = pd.read_json(new_rating, orient='split')

        else:
            rating_points = pd.read_json(rating_points, orient='split')
            rating_points = rating_points.loc[rating_points.index == rating]
       

        if graph_offset == True:
                try:
                    gzf = round(rating_points["gzf"].loc[rating_points.index == rating].mean(), 2)
                except:
                     gzf = 0
        else: 
                gzf = 0
        

        tbl = pd.DataFrame(data_grid)
        tbl = tbl.loc[tbl["selection"] == True]
        tbl['datetime'] = pd.to_datetime(tbl['datetime'])
        tbl['observation_age'] = (tbl['datetime'].max() - tbl['datetime']).dt.days
       
        tbl['normalized_age'] = (1 - (tbl['observation_age'] - tbl['observation_age'].min()) / (tbl['observation_age'].max() - tbl['observation_age'].min())).round(1)
        tbl.loc[tbl['normalized_age'] < 0.2, "normalized_age"] = 0.2 # otherwise very old observations will not appear
      

        #if not tbl.empty:
        #    tbl = pd.DataFrame(tbl)


        ### need to make min age anon zero number   ddd 
                # Define a colormap where opacity varies with normalized age
        #colormap = np.linspace(0.1, 1, len(obs)).round(1)  # Adjust the range and steps as needed
        #colorscale = [[i, 'rgb(127, 255, 212)'] for i in colormap]  # Mint color scale
       
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"{site_id} Rating Graph", f"{site_id} Stage Offset Graph"))
        
        fig.update_layout(title=f'{site_id} Rating Graph: Stage - GZF ({gzf}) vs Discharge')
        
        
        
            

        # Helper function to add traces to both columns
        #### graph 1 column 1 normal rating graph
        fig.update_layout(
            # subplot 1 x
            xaxis=dict( title='Discharge', type='log', showticklabels=True, showgrid=True, ticks="outside", minor_ticks="outside", showline=True, linecolor='black', linewidth=1, mirror=True),
            # subplot y1
            yaxis=dict( title=f'Stage - GZF ({gzf})', type='log', showticklabels=True, showgrid=True, ticks="outside", minor_ticks="outside", showline=True, linecolor='black', linewidth=1, mirror=True ),)
        
        # Add rating trace
        fig.add_trace(go.Scatter(
            x=rating_points.loc[rating_points.index == rating, "discharge_rating"],
            y=(rating_points.loc[rating_points.index == rating, "stage_rating"] if graph_offset else
            rating_points.loc[rating_points.index == rating, "stage_rating"] + rating_points.loc[rating_points.index == rating, "gzf"]),
            line=dict(width=1),
            name=f"{rating} GZF: {round(rating_points['gzf'].loc[rating_points.index == rating].mean(), 2)}",
            showlegend=True), row=1, col=1)
        
        #### if offset rating add this as well
        if 'offset_rating' in offset_rating and new_rating:
            new_rating = pd.read_json(new_rating, orient='split')
            fig.add_trace(go.Scatter(
            x=new_rating[ "discharge_rating"],
            y=(new_rating["stage_rating"] if graph_offset else
            new_rating["stage_rating"] + new_rating["gzf"]),
            line=dict(width=1, dash="dash"),  
            name=f"rating offset GZF: {round(new_rating['gzf'].mean(), 2)}",
            showlegend=True), row=1, col=1)

        # Add raw observations
        fig.add_trace(go.Scatter(
            x=obs["parameter_observation"],
            y=round(obs["observation_stage"] - gzf, 2),
            mode='markers',
            marker=dict(size=6, opacity=0.5, color="gray"),
            text=obs["observation_number"],
            name="obs", showlegend=True), row=1, col=1)
        
        # Add mask observations with or without labels
        fig.add_trace(go.Scatter(
            x=tbl["parameter_observation"],
            y=round(tbl["observation_stage"] - gzf, 2),
            mode='markers+text' if show_obs else 'markers',
            marker=dict(size=10, color=tbl['normalized_age'], colorscale="Mint"),
            text=tbl["observation_number"],
            textposition="top center" if show_obs else None,
            name="obs",
            showlegend=True
        ), row=1, col=1)

       
        # Add tbl_addition if it exists
        if not tbl_addition.empty:
            fig.add_trace(go.Scatter(
                x=tbl_addition["parameter_observation"],
                y=round(tbl_addition["observation_stage"].astype(float, errors='ignore') - gzf, 2),
                mode='markers',
                marker=dict(size=10, color='red'),
                text=tbl_addition["observation_number"],
                name="addition", showlegend=True), row=1, col=1)
        
        #### graph 2 column 2 top view rating
        fig.update_layout(xaxis2=dict(  # Subplot 2 X
                title='offset',  # Example: could be linear or with custom ticks
                type='linear', showticklabels=True, showgrid=True, ticks="outside", minor_ticks="outside", showline=True, linecolor='black', linewidth=1, mirror=True),

            yaxis2=dict(  # Subplot 2 Y
                title='Stage (ft)',
                type='log',  # Example: you can use 'linear' here for contrast
                showticklabels=True,
                showgrid=True,
                ticks="outside",
                minor_ticks="outside",
                showline=True,
                linecolor='black',
                linewidth=1,
                mirror=True
            ))
         # Add rating trace  graph stage and offset (zero)
     
        fig.add_trace(go.Scatter(
            x=[0] * len(rating_points.loc[rating_points.index == rating]),  # All x-values = 0
            y=rating_points.loc[rating_points.index == rating, "stage_rating"],
            line=dict(width=1),
            name=f"{rating} GZF: {round(rating_points['gzf'].loc[rating_points.index == rating].mean(), 2)}",
            showlegend=True
        ), row=1, col=2)

        if 'offset_rating' in offset_rating and not new_rating.empty:
            #new_rating = pd.read_json(new_rating, orient='split')
            #rating_off = pd.merge(rating_points, new_rating, on = "stage_rating")

            q_merge = pd.merge(rating_points[["stage_rating", "discharge_rating"]].rename(columns={"stage_rating": "original_stage"}), new_rating, on = "discharge_rating", how = "right")
            q_merge["offset"] = round(q_merge["original_stage"] - q_merge["stage_rating"], 2)
        
           # print(rating_off)
        # Add tbl_addition if it exists

            fig.add_trace(go.Scatter(
                x=q_merge["offset"],  # All x-values = 0
                y=(q_merge["stage_rating"]),
                line=dict(width=1, dash="dash"),
                name=f"offset rating",
                showlegend=True
            ), row=1, col=2)

            
        if not tbl_addition.empty:
            fig.add_trace(go.Scatter(
                x=tbl_addition["rating_offset"],
                y = tbl_addition["observation_stage"].astype(float, errors='ignore'),
                mode='markers',
                marker=dict(size=10, color='red'),
                text=tbl_addition["observation_number"],
                name="addition", showlegend=True), row=1, col=2)

        if not tbl.empty:
            #tbl = pd.DataFrame(tbl)
            """fig.add_trace(go.Scatter(
                x=tbl["rating_offset"],
                y=round(tbl["observation_stage"], 2),
                mode='markers',
                marker=dict(size=6, opacity=0.5, color="gray"),
                text=tbl["observation_number"],
                name="obs",
                showlegend=True), row=1, col=2)
            """


            fig.add_trace(go.Scatter(
                x=tbl["rating_offset"],
                y=round(tbl["observation_stage"], 2),
                mode='markers+text' if show_obs else 'markers',
                marker=dict(size=10, color=tbl['normalized_age'], colorscale="Mint"),
                text=tbl["observation_number"],
                textposition="top center" if show_obs else None,
                name="obs",
                showlegend=True
            ), row=1, col=2)
            
        # Return Dash component
        return html.Div(dcc.Graph(figure=fig), style={'width': '100%', 'height': '100%'}),

                            

    else:
        return dash.no_update

if __name__ == '__main__':
    #serve(app.server, port="8050",host='127.0.0.1') # dont use
    #app.run_server(host='0.0.0.0',debug=True) # dont use
    app.run_server(port="8050",host='127.0.0.1',debug=True) # use for general local host running
    #app.run_server(port=8050, host='0.0.0.0', debug=True) # binds to network not just local compouter

    #app.run_server(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)  # binder dev
    
    ### 
    
#fig.show()
#print(rating_points)
#print(rating_list)

