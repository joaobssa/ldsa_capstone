
import aux_functions

import pandas as pd
pd.set_option('display.max_columns', 100)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, make_scorer, confusion_matrix, roc_curve



def clean_data(df, drop_cols):

    # Function to get clean training data with labels
    
    # Inputs: 
    #     df - original dataframe
    #     drop_cols - columns that aren't part of the model
    # Ouput:
    #     df_clean - dataframe with cleaned columns
    
    df_clean = df.copy()

    # dropping metropolitan entries

    # df_clean = df.copy().drop(columns='observation_id')  -- API includes id

    df_clean = df_clean.loc[df_clean["station"]!='metropolitan',:].copy()

    # deal with unordered categorical columns

    cat_columns = ['Type', 'Gender', 'Self-defined ethnicity', 'Officer-defined ethnicity', 'Legislation',
        'Object of search', 'Outcome', 'station']

    for col in cat_columns:
        df_clean[col] = df_clean[col].astype('category').cat.as_unordered()

    # deal with categorical columns

    cat_columns = ['Age range']

    df_clean['Age range'] = df_clean['Age range'].astype('category').cat.as_ordered().cat.reorder_categories(['under 10', '10-17', '18-24', '25-34', 'over 34'], ordered=True)

    # deal with boolean columns

    bool_columns = ['Part of a policing operation','Outcome linked to object of search','Removal of more than just outer clothing']

    for col in bool_columns:
        df_clean[col] = df_clean[col].astype('boolean')


    # generate target columns
    # Note on target: 'A no further action disposal' -> 0 ; not 'A no further action disposal' and 'Outcome linked to object of search' -> 1 
    df_clean['target'] = 0
    df_clean.loc[(df_clean["Outcome"]!='A no further action disposal') & (df_clean['Outcome linked to object of search']==True), 'target'] = 1

    # drop columns that aren't inputs of the model

    df_clean = df_clean.drop(columns=drop_cols).copy()

    return df_clean


def plot_roc_curve(y_test, y_hat):
    # Function to plot ROC Curve
    
    # Inputs: 
    #     y_test - true labels for the test set
    #     y_hat - predicted labels on the test set
    # Outputs:
    #     ROC Curve graph



    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_hat[:,1])


    roc_auc = roc_auc_score(y_true=y_test, y_score=y_hat[:,1])

    
    plt.figure(figsize=(8,6))
    lw = 2
    plt.plot(fpr, tpr, color='orange', lw=lw, label='ROC curve (AUROC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='random')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()



def show_metrics(kind, y_true, y_pred):

    # Function to show all relevant ML metrics
    
    # Inputs: 
    #     kind - data set being measured (train or test)
    #     y_true - true labels
    #     y_pred - predicted labels
    # Outputs:
    #     Bar graphs with major metrics and Confusion Matrix

    accuracy = round(accuracy_score(y_true=y_true, y_pred=y_pred), 3)
    # precision
    precision = round(precision_score(y_true=y_true, y_pred=y_pred), 3)
    # recall
    recall = round(recall_score(y_true=y_true, y_pred=y_pred), 3)
    # f1-score
    f1 = round(f1_score(y_true=y_true, y_pred=y_pred), 3)

    x = ["Accuracy", "Precision", "Recall", "f1-score"]
    y = [accuracy, precision, recall, f1]
    plt.bar(x, height=y)

    for i in range(len(x)):
            plt.text(i,y[i],y[i], ha = 'center')
    plt.title('Metrics for the ' + kind + ' set')
    plt.show()

    # generate confusion matrix
    confmat = confusion_matrix(y_true=y_true, y_pred=y_pred)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.4)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i,
                    s=confmat[i, j],
                    va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.title('Confusion Matrix for the ' + kind + ' set')
    plt.show()

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
        response = {
                "observation_id": str(None),
                "error": "observation_id field is missing from request"
            }
        return response, check
    if type(observation_id_) != str:
        response = {
                "observation_id": str(observation_id_),
                "error": 'Provided "observation_id" field is not of the correct data type'
            }
        return response, check

    # CHECK THAT THERE ARE NO MISSING OR EXTRA FIELDS IN OBSERVATION ############################################

    keys = set(obs_dict.keys())
    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        response = {
                "error": error
            }
        return response, check
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        response = {
                "error": error
            }
        return response, check
    

    # CHECK FIELD INTEGRITY #####################################################################################

    
    try:
        type_ = obs_dict["Type"]
    except:
        response = {
                "type": str(None),
                "error": "Type field is missing from request"
            }
        return response, check
    if type(type_) != str:
        response = {
                "type": str(type_),
                "error": 'Provided "Type" field is not of the correct data type'
            }
        return response, check
    # VALIDATE CATEGORY VALUES
    if type_ not in valid_categories["Type"]:
        error = "Invalid value provided for Type: {}. Allowed values are: {}".format(
            type_, ",".join(["'{}'".format(v) for v in valid_categories["Type"]]))
        response = {
                "error": error
            }
        return response, check

    try:
        date_ = obs_dict["Date"]
    except:
        response = {
                "Date": str(None),
                "error": "Date field is missing from request"
            }
        return response, check
    if type(date_) != str:
        response = {
                "Date": str(date_),
                "error": 'Provided "Date" field is not of the correct data type'
            }
        return response, check
    # VALIDATE DATES
    try:
        date_test = pd.to_datetime(date_)
    except:
        response = {
                "Date": str(date_),
                "error": "Date format is incorrect"
            }
        return response, check
    if(date_test.year < 2020):
        response = {
                "Date": str(date_),
                "error": "Provided date is before 2020"
            }
        return response, check


    try:
        policing_op_ = obs_dict["Part of a policing operation"]
    except:
        response = {
                "Part of a policing operation": str(None),
                "error": 'Part of a policing operation field is missing from request'
            }
        return response, check
    if type(policing_op_) != bool:
        response = {
                "Part of a policing operation": policing_op_,
                "error": 'Provided "Part of a policing operation" field is not of the correct data type'
            }
        return response, check
    
    try:
        lat_ = obs_dict["Latitude"]
    except:
        response = {
                "Latitude": str(None),
                "error": "Type field is missing from request"
            }
        return response, check
    if type(lat_) != float:
        response = {
                "Latitude": str(lat_),
                "error": 'Provided "Latitude" field is not of the correct data type'
            }
        return response, check

    try:
        long_ = obs_dict["Longitude"]
    except:
        response = {
                "Longitude": str(None),
                "error": "Type field is missing from request"
            }
        return response, check
    if type(long_) != float:
        response = {
                "Longitude": str(long_),
                "error": 'Provided "Longitude" field is not of the correct data type'
            }
        return response, check

    try:
        gend_ = obs_dict["Gender"]
    except:
        response = {
                "Gender": str(None),
                "error": "Type field is missing from request"
            }
        return response, check
    if type(gend_) != str:
        response = {
                "Gender": str(gend_),
                "error": 'Provided "Gender" field is not of the correct data type'
            }
        return response, check
    # VALIDATE CATEGORY VALUES
    if gend_ not in valid_categories["Gender"]:
        error = "Invalid value provided for Gender: {}. Allowed values are: {}".format(
            gend_, ",".join(["'{}'".format(v) for v in valid_categories["Gender"]]))
        response = {
                "error": error
            }
        return response, check
    

    try:
        age_range_ = obs_dict["Age range"]
    except:
        response = {
                "Age range": str(None),
                "error": "Type field is missing from request"
            }
        return response, check
    if type(age_range_) != str:
        response = {
                "Age range": str(age_range_),
                "error": 'Provided "Age range" field is not of the correct data type'
            }
        return response, check
    # VALIDATE CATEGORY VALUES
    if age_range_ not in valid_categories["Age range"]:
        error = "Invalid value provided for Age range: {}. Allowed values are: {}".format(
            age_range_, ",".join(["'{}'".format(v) for v in valid_categories["Age range"]]))
        response = {
                "error": error
            }
        return response, check

    try:
        officer_def_ethnicity_ = obs_dict["Officer-defined ethnicity"]
    except:
        response = {
                "Officer-defined ethnicity": str(None),
                "error": "Officer-defined ethnicity field is missing from request"
            }
        return response, check
    if type(officer_def_ethnicity_) != str:
        response = {
                "Officer-defined ethnicity": str(officer_def_ethnicity_),
                "error": 'Provided "Officer-defined ethnicity" field is not of the correct data type'
            }
        return response, check
    # VALIDATE CATEGORY VALUES
    if officer_def_ethnicity_ not in valid_categories["Officer-defined ethnicity"]:
        error = "Invalid value provided for Officer-defined ethnicity: {}. Allowed values are: {}".format(
            officer_def_ethnicity_, ",".join(["'{}'".format(v) for v in valid_categories["Officer-defined ethnicity"]]))
        response = {
                "error": error
            }
        return response, check
    

    try:
        legislation_ = obs_dict["Legislation"]
    except:
        response = {
                "Legislation": str(None),
                "error": "Legislation field is missing from request"
            }
        return response
    if type(legislation_) != str:
        response = {
                "Legislation": str(legislation_),
                "error": 'Provided "Legislation" field is not of the correct data type'
            }
        return response, check

    try:
        obj_search_ = obs_dict["Object of search"]
    except:
        response = {
                "Object of search": str(None),
                "error": "Object of search field is missing from request"
            }
        return response, check
    if type(obj_search_) != str:
        response = {
                "Object of search": str(obj_search_),
                "error": 'Provided "Object of search" field is not of the correct data type'
            }
        return response, check

    try:
        station_ = obs_dict["station"]
    except:
        response = {
                "station": str(None),
                "error": "station field is missing from request"
            }
        return response, check
    if type(station_) != str:
        response = {
                "station": str(station_),
                "error": 'Provided "station" field is not of the correct data type'
            }
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