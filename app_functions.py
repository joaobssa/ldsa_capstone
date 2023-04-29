import pandas as pd
import numpy as np

def attempt_predict(obs_dict):

    # Function used to protect APP from unwanted inputs
    
    # Inputs: 
    #     obs_dict - dictionary containing input data
    # Outputs:
    #     observation - dictionary containing approved data
    #     check - boolean indicating whether all checks were passed

    valid_columns = {
      "observation_id",
      "Type",
      "Date",
      "Part of a policing operation", 
      "Latitude", 
      "Longitude", 
      "Gender",
      "Age range",
      "Officer-defined ethnicity",
      "Legislation",
      "Object of search",
      "station"
    }

    valid_categories = {
        "Type": ['Person search', 'Person and Vehicle search', 'Vehicle search'],
        "Gender": ['Female', 'Male', 'Other'],
        "Age range": ['under 10', '10-17', '18-24', '25-34', 'over 34'],
        "Officer-defined ethnicity": ['Asian', 'Black', 'Mixed', 'White', 'Other' ]
    }

    # CHECK THAT observation_id exists ############################################################################

    check = False # setting check to false by default

    try:
        observation_id_ = obs_dict["observation_id"]
    except:
        response = {"observation_id": str(None), "error": "observation_id field is missing from request"}
        return response, check
    if type(observation_id_) != str:
        response = {"observation_id": str(observation_id_), "error": 'Provided "observation_id" field is not of the correct data type'}
        return response, check

    # CHECK THAT THERE ARE NO MISSING OR EXTRA FIELDS IN OBSERVATION ############################################

    keys = set(obs_dict.keys())
    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        response = {"error": error}
        return response, check
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        response = {"error": error}
        return response, check
    

    # CHECK FIELD INTEGRITY #####################################################################################

    
    try:
        type_ = obs_dict["Type"]
    except:
        response = {"type": str(None), "error": "Type field is missing from request"}
        return response, check
    if type(type_) != str:
        response = {"type": str(type_), "error": 'Provided "Type" field is not of the correct data type'}
        return response, check
    # VALIDATE CATEGORY VALUES
    if type_ not in valid_categories["Type"]:
        error = "Invalid value provided for Type: {}. Allowed values are: {}".format(
            type_, ",".join(["'{}'".format(v) for v in valid_categories["Type"]]))
        response = {"error": error}
        return response, check

    try:
        date_ = obs_dict["Date"]
    except:
        response = {"Date": str(None), "error": "Date field is missing from request"}
        return response, check
    if type(date_) != str:
        response = {"Date": str(date_), "error": 'Provided "Date" field is not of the correct data type'}
        return response, check
    # VALIDATE DATES
    try:
        date_test = pd.to_datetime(date_)
    except:
        response = {"Date": str(date_), "error": "Date format is incorrect"}
        return response, check
    if(date_test.year < 2020):
        response = {"Date": str(date_), "error": "Provided date is before 2020"}
        return response, check


    try:
        policing_op_ = obs_dict["Part of a policing operation"]
    except:
        response = {
                "Part of a policing operation": str(None),
                "error": 'Part of a policing operation field is missing from request'
            }
        return response, check
    if (~np.isnan(policing_op_)) & (type(policing_op_) != bool):
        response = {
                "Part of a policing operation": policing_op_,
                "error": 'Provided "Part of a policing operation" field is not of the correct data type'
            }
        return response, check
    
    try:
        lat_ = obs_dict["Latitude"]
    except:
        response = {"Latitude": str(None), "error": "Type field is missing from request"}
        return response, check
    if (~np.isnan(lat_)) & (type(lat_) != float):
        response = {"Latitude": str(lat_), "error": 'Provided "Latitude" field is not of the correct data type'}
        return response, check

    try:
        long_ = obs_dict["Longitude"]
    except:
        response = {"Longitude": str(None), "error": "Type field is missing from request"}
        return response, check
    if (~np.isnan(long_)) &  (type(long_) != float):
        response = {"Longitude": str(long_),"error": 'Provided "Longitude" field is not of the correct data type'}
        return response, check

    try:
        gend_ = obs_dict["Gender"]
    except:
        response = {"Gender": str(None), "error": "Type field is missing from request"}
        return response, check
    if type(gend_) != str:
        response = {"Gender": str(gend_), "error": 'Provided "Gender" field is not of the correct data type'}
        return response, check
    # VALIDATE CATEGORY VALUES
    if gend_ not in valid_categories["Gender"]:
        error = "Invalid value provided for Gender: {}. Allowed values are: {}".format(
            gend_, ",".join(["'{}'".format(v) for v in valid_categories["Gender"]]))
        response = {"error": error}
        return response, check
    

    try:
        age_range_ = obs_dict["Age range"]
    except:
        response = {"Age range": str(None), "error": "Type field is missing from request"}
        return response, check
    if type(age_range_) != str:
        response = {"Age range": str(age_range_), "error": 'Provided "Age range" field is not of the correct data type'}
        return response, check
    # VALIDATE CATEGORY VALUES
    if age_range_ not in valid_categories["Age range"]:
        error = "Invalid value provided for Age range: {}. Allowed values are: {}".format(
            age_range_, ",".join(["'{}'".format(v) for v in valid_categories["Age range"]]))
        response = {"error": error}
        return response, check

    try:
        officer_def_ethnicity_ = obs_dict["Officer-defined ethnicity"]
    except:
        response = {"Officer-defined ethnicity": str(None),
                "error": "Officer-defined ethnicity field is missing from request"}
        return response, check
    if type(officer_def_ethnicity_) != str:
        response = {"Officer-defined ethnicity": str(officer_def_ethnicity_), "error": 'Provided "Officer-defined ethnicity" field is not of the correct data type'}
        return response, check
    # VALIDATE CATEGORY VALUES
    if officer_def_ethnicity_ not in valid_categories["Officer-defined ethnicity"]:
        error = "Invalid value provided for Officer-defined ethnicity: {}. Allowed values are: {}".format(
            officer_def_ethnicity_, ",".join(["'{}'".format(v) for v in valid_categories["Officer-defined ethnicity"]]))
        response = {"error": error}
        return response, check
    

    try:
        legislation_ = obs_dict["Legislation"]
    except:
        response = {"Legislation": str(None), "error": "Legislation field is missing from request"}
        return response
    if (~np.isnan(legislation_)) &  (type(legislation_) != str):
        response = {"Legislation": str(legislation_), "error": 'Provided "Legislation" field is not of the correct data type'}
        return response, check

    try:
        obj_search_ = obs_dict["Object of search"]
    except:
        response = {"Object of search": str(None), "error": "Object of search field is missing from request"}
        return response, check
    if type(obj_search_) != str:
        response = {"Object of search": str(obj_search_), "error": 'Provided "Object of search" field is not of the correct data type'}
        return response, check

    try:
        station_ = obs_dict["station"]
    except:
        response = {"station": str(None), "error": "station field is missing from request"}
        return response, check
    if type(station_) != str:
        response = {"station": str(station_),"error": 'Provided "station" field is not of the correct data type'}
        return response, check

    check = True # all checks were passed

    observation =   {'observation_id': observation_id_,
                    'Type': type_,
                    'Date': date_,
                    'Part of a policing operation': policing_op_,
                    'Latitude': lat_,
                    'Longitude': long_,
                    'Gender': gend_,
                    'Age range': age_range_,
                    'Officer-defined ethnicity': officer_def_ethnicity_,
                    'Legislation': legislation_,
                    'Object of search': obj_search_,
                    'station': station_}
    
    return observation, check