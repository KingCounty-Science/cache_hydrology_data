import pandas as pd


# Full discharge calculation
def discharge_calculation(df_q, ratings_value, site_sql_id):

        desired_order = ["datetime", "data", "corrected_data", "observation_stage", "q_observation", "offset", "estimate", "warning", "comparison", "dry_indicator"] # observation and observation_stage are kinda redundent at some point and should be clarified
        existing_columns = [col for col in desired_order if col in df_q.columns]# Filter out columns that exist in the DataFrame
        df = df_q[existing_columns].copy() # Reorder the DataFrame columns

        rating_calculation_status = "not calculated"
        if rating_calculation_status != ratings_value:
                from import_data import rating_calculator
                # fx sets flag and wont run unless flag has been reset
                rating_calculation_status, rating_points, gzf = rating_calculator(ratings_value, site_sql_id)
       
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
        df["adjusted_stage"] = ((df["corrected_data"]-gzf)-df["q_offset"]).round(2)
        
        # get discharge values
        df = df.merge(rating_points, left_on="adjusted_stage", right_on='water_level_rating', how = 'outer')
        df = df.drop(columns = ["adjusted_stage", "water_level_rating"])
        
        df = df.sort_values(by="datetime")
      
        df.sort_values(by=['datetime'], inplace=True)
       
        desired_order = ["datetime", "data", "corrected_data", "discharge", "observation_stage", "q_observation", "offset", "q_offset", "precent_q_change", "rating_number", "estimate", "warning", "comparison", "dry_indicator"]# observation and observation_stage are kinda redundent at some point and should be clarified
        existing_columns = [col for col in desired_order if col in df.columns] # Filter out columns that exist in the DataFrame
        df = df[existing_columns].copy() # Reorder the DataFrame columns

        return df


# finalize discharge dataframe
def finalize_discharge_dataframe(df_q):
        df_q['data'] = df_q['data'].round(2)
        df_q['corrected_data'] = df_q['corrected_data'].round(2)
        df_q['observation_stage'] = df_q['observation_stage'].round(2)
        df_q['q_observation'] = df_q['q_observation'].round(2)
        df_q['discharge'] = df_q['discharge'].round(2)
        # convert datetime
        df_q['datetime'] = pd.to_datetime(
                df_q['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce', infer_datetime_format=True)
        
        desired_order = ["datetime", "data", "corrected_data", "discharge", "observation_stage", "q_observation", "offset", "q_offset", "precent_q_change", "rating_number", "estimate", "warning", "comparison", "dry_indicator"]
        # Filter out columns that exist in the DataFrame
        existing_columns = [col for col in desired_order if col in df_q.columns]
        # Reorder the DataFrame columns
        df_q = df_q[existing_columns].copy()
    
        return df_q