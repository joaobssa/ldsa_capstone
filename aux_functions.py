import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, make_scorer, confusion_matrix, roc_curve



def clean_data(df, drop_cols, drop_stations=['metropolitan', 'humberside', 'leicestershire', 'lancashire']):

    # Function to get clean training data with labels
    
    # Inputs: 
    #     df - original dataframe
    #     drop_cols - columns that aren't part of the model
    # Ouput:
    #     df_clean - dataframe with cleaned columns
    
    df_clean = df.copy()

    # dropping metropolitan entries

    df_clean = df_clean.loc[~df_clean["station"].isin(drop_stations),:].copy()


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

def clean_new_data(df):
    
    df_clean = df.copy()
    
    column_mapper = {'type': 'Type', 
                    'date': 'Date',
                    'part_of_a_policing_operation': 'Part of a policing operation', 
                    'latitude': 'Latitude', 
                    'longitude': 'Longitude', 
                    'gender': "Gender",
                    'age_range': 'Age range', 
                    'officer_defined_ethnicity': 'Officer-defined ethnicity', 
                    'legislation': 'Legislation',
                    'object_of_search': 'Object of search',
                    'true_outcome': "target"
                    }

    df_clean = df_clean.drop(columns=["Unnamed: 0", "id", "proba", "outcome"]).rename(columns=column_mapper)

    cat_columns = ['Type', 'Gender', 'Officer-defined ethnicity', 'Legislation',
            'Object of search', 'station']

    for col in cat_columns:
        df_clean[col] = df_clean[col].astype('category').cat.as_unordered()

    # deal with categorical columns

    cat_columns = ['Age range']

    df_clean['Age range'] = df_clean['Age range'].astype('category').cat.as_ordered().cat.reorder_categories(['under 10', '10-17', '18-24', '25-34', 'over 34'], ordered=True)

    # deal with boolean columns

    bool_columns = ['Part of a policing operation']

    for col in bool_columns:
        df_clean[col] = df_clean[col].astype('boolean')

    df_clean["target"] = df_clean["target"].astype(int)

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


def verify_success_rate(y_true, y_pred, min_success_rate=0.8):
    """
    Verifies the success rate on a test set is above a provided minimum
    
    We shall consider success for this project as the model having at least 0.8 recall
    
    """
    
    recall = recall_score(y_true, y_pred, pos_label=True)
    is_satisfied = (recall >= min_success_rate)
    
    return is_satisfied, recall

def verify_no_discrimination(X_test, y_true, y_pred, sensitive_column='Officer-defined ethnicity', max_diff=0.05, min_samples=5):
    """
    Verifies that no subdeparment has discrimination in between protected races
    """
    
    departments = X_test['station'].unique()
    sensitive_classes = X_test[sensitive_column].unique()
    
    is_satisfied = True
    problematic_departments = []
    good_deparments = []
    ignored_departments = []
    for department in departments:
        precisions = {}
        for sensitive_class in sensitive_classes:

            mask = (X_test[sensitive_column] == sensitive_class) & (X_test['station'] == department)
            if np.sum(y_true[mask]) > min_samples:   # the department needs to have at least some positive labels so that it makes sense to measure the precision
                precisions[sensitive_class] = precision_score(y_true[mask], y_pred[mask], pos_label=1, zero_division=0) # defaults to 0 if the model predicted 0 success outcomes
                
        if len(precisions) > 1:

            diff = np.max(list(precisions.values())) - np.min(list(precisions.values()))

            if diff > max_diff:
                is_satisfied = False
                problematic_departments.append((department, diff, precisions))
            else:
                good_deparments.append((department, diff, precisions))
        else:
            print(department + "was ignored")
            ignored_departments.append((department, None, []))
    
    global_precisions = {}
    for sensitive_class in sensitive_classes:
        mask = (X_test[sensitive_column] == sensitive_class)
        if np.sum(y_true[mask]) > min_samples: # the department needs to have at least some positive labels so that precision makes sense
            global_precisions[sensitive_class] = precision_score(y_true[mask], y_pred[mask], pos_label=1, zero_division=0) # defaults to 0 if the model predicted 0 success outcomes
    
    if len(precisions) > 1:    
        diff = np.max(list(precisions.values())) - np.min(list(precisions.values()))
        if diff > max_diff:
            is_satisfied = False
        
    return is_satisfied, problematic_departments, good_deparments, global_precisions

def verify_no_discrimination_v2(X_test, y_true, y_pred, max_diff=0.05, min_samples=5):
    """
    Verifies that no subdeparment has discrimination in between protected races
    """
    
    departments = X_test['station'].unique()
    ethnicity_classes = X_test['Officer-defined ethnicity'].unique()
    gender_classes = X_test['Gender'].unique()
    
    sensitive_classes = [(x,y) for x in ethnicity_classes for y in gender_classes]

    is_satisfied = True
    problematic_departments = []
    good_deparments = []
    ignored_departments = []
    all_departments = {}
    for department in departments:
        precisions = {}
        for sensitive_class in sensitive_classes:

            mask = (X_test['Officer-defined ethnicity'] == sensitive_class[0]) & (X_test['Gender'] == sensitive_class[1]) & (X_test['station'] == department)
            if np.sum(y_true[mask]) > min_samples:   # the department needs to have at least some positive labels so that it makes sense to measure the precision
                precisions[sensitive_class] = precision_score(y_true[mask], y_pred[mask], pos_label=1, zero_division=0) # defaults to 0 if the model predicted 0 success outcomes
                
        if len(precisions) > 1:

            diff = np.max(list(precisions.values())) - np.min(list(precisions.values()))

            if diff > max_diff:
                is_satisfied = False
                problematic_departments.append((department, diff, precisions))
            else:
                good_deparments.append((department, diff, precisions))
        else:
            print(department + "was ignored")
            ignored_departments.append((department, None, []))
        
        all_departments[department] = precisions
    
    global_precisions = {}
    for sensitive_class in sensitive_classes:
        mask = (X_test['Officer-defined ethnicity'] == sensitive_class[0]) & (X_test['Gender'] == sensitive_class[1])
        if np.sum(y_true[mask]) > min_samples: # the department needs to have at least some positive labels so that precision makes sense
            global_precisions[sensitive_class] = precision_score(y_true[mask], y_pred[mask], pos_label=1, zero_division=0) # defaults to 0 if the model predicted 0 success outcomes
        
    return is_satisfied, problematic_departments, good_deparments, global_precisions, all_departments


def verify_no_discrimination_v3(X_test, y_true, y_pred, max_diff=0.05, min_samples=30):  

    departments = X_test['station'].unique()
    ethnicity_classes = X_test['Officer-defined ethnicity'].unique()
    gender_classes = X_test['Gender'].unique()

    sensitive_classes = [(x,y) for x in ethnicity_classes for y in gender_classes]

    department_rates = {}
    for department in departments:
        precision_rates = {}
        for sensitive_class in sensitive_classes:

            mask = (X_test['Officer-defined ethnicity'] == sensitive_class[0]) & (X_test['Gender'] == sensitive_class[1]) & (X_test['station'] == department)
            
            total = len(X_test[mask])

            # client suggests checking discrimination only for race-gender tuples that have a minimum of 30 samples
            if(total < min_samples):
                precision_rate = np.nan 
            else:
                precision_rate = precision_score(y_true[mask], y_pred[mask], pos_label=1, zero_division=0)

            precision_rates[sensitive_class] = precision_rate
        department_rates[department] = precision_rates

    rates = pd.DataFrame.from_dict(department_rates).transpose()

    global_precisions = rates.transpose().mean(axis=1).to_dict()

    rates.transpose().mean(axis=1).sort_index().plot.barh()
    plt.title("Avg Precision Across Race-Gender Tuples")
    plt.show();
    
    rates["diff"] = rates.max(axis=1).round(5) - rates.min(axis=1).round(5)

    good_deparments = list(rates.loc[rates["diff"]<max_diff].index)

    problematic_departments = list(rates.loc[rates["diff"]>=max_diff].index)

    if len(problematic_departments) > 0:
        is_satisfied = False
    else:
        is_satisfied = True

    all_departments = rates.copy()

    rates["diff"].sort_values().plot.barh()
    plt.title("Highest Precision Difference across stations")
    plt.show();

    # pd.DataFrame(global_precisions, index=["avg difference"]).plot.barh()
    # plt.title("Difference across stations")


    return is_satisfied, problematic_departments, good_deparments, global_precisions, all_departments