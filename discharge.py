import pandas as pd
from data_cleaning import reformat_data, parameter_calculation, column_managment

# Full discharge calculation
def discharge_calculation(df_q, ratings_value, site_sql_id, apply_discharge_offset): #apply_discharge_offset disabled=True
        df = df_q.copy()
        df = column_managment(df)
        columns_to_drop = ["discharge", "q_offset", "precent_q_change"]
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
        
        
        from import_data import rating_calculator
                # fx sets flag and wont run unless flag has been reset
        rating_points, gzf = rating_calculator(ratings_value, site_sql_id)
        
        df["q_observation"] = round(df["q_observation"], 2)

        ## get rating discharge at q_obs
        df_obs = pd.merge_asof(df.sort_values('q_observation').dropna(subset=["q_observation"]), rating_points.sort_values('discharge'), left_on = "q_observation", right_on = "discharge", allow_exact_matches=True, direction='nearest')
        # calculated q offset
      
        df_obs["q_offset"] = ((df_obs["observation_stage"]-gzf)-df_obs["water_level_rating"]).round(2)
        df_obs["precent_q_change"] = abs((df_obs["q_offset"]/(df_obs["observation_stage"]-gzf))*100).round(2)
        df_obs = df_obs[["datetime", "q_offset", "precent_q_change"]]
        
        df = df.merge(df_obs, on = "datetime", how = 'left') # add offset and change to df

        # fill q offset
        df["q_offset"] = df["q_offset"].interpolate(method='linear', limit_direction='both')
        if apply_discharge_offset is True: # aka disabled is false apply individual offset for discharge
            print("value is True apply discharge offsets for individual measurements")
            df["adjusted_stage"] = ((df["corrected_data"]-gzf)-df["q_offset"]).round(2)
        if apply_discharge_offset is False:  # dont correct for difference but you still want to show it on like the graph
              print("value is false do not apply individual offsets")
              df["adjusted_stage"] = ((df["corrected_data"]-gzf)).round(2)
        #df["adjusted_stage"] = df["adjusted_stage"]+gzf
        #rating_points["water_level_rating"] = rating_points["water_level_rating"]+gzf
        # get discharge values
        df = df.merge(rating_points, left_on="adjusted_stage", right_on='water_level_rating', how = 'left')
       
        df = df.drop(columns = ["adjusted_stage", "water_level_rating"])
        
        df = df.sort_values(by="datetime")
      
        df.sort_values(by=['datetime'], inplace=True, ascending = False)
       
        df = column_managment(df)
        df = reformat_data(df)
        # round safly
        def safe_round(value, ndigits=2):
            try:
                return round(value, ndigits)
            except Exception:
                return value  # Return the original value if an error occurs

        df['q_offset'] = df['q_offset'].apply(safe_round, ndigits=2)
        df['discharge'] = df['discharge'].apply(safe_round, ndigits=2)

        return df

