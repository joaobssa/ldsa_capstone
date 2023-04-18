import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError, BooleanField
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect

# import custom transformers

from transformers import TimeTransformer, BoolTransformer

########################################
# Begin database stuff

# The connect function checks if there is a DATABASE_URL env var.
# If it exists, it uses it to connect to a remote postgres db.
# Otherwise, it connects to a local sqliåte db stored in predictions.db.
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    # observation_id = IntegerField(unique=True)
    # observation = TextField()
    observation_id: TextField()(unique=True)
    type: TextField()
    date: TextField()
    part_of_a_policing_operation: BooleanField()
    latitude: FloatField()
    longitude: FloatField()
    gender: TextField()
    age_range: TextField()
    officer_defined_ethnicity: TextField()
    legislation: TextField()
    object_of_search: TextField()
    station:TextField()
    proba: FloatField()
    outcome: BooleanField()
    true_outcome: BooleanField()

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/should_search', methods=['POST'])
def should_search():
    # Flask provides a deserialization convenience function called
    # get_json that will work if the mimetype is application/json.
    obs_dict = request.get_json()
    
    # observation = obs_dict['observation']
    observation_id_ = obs_dict["observation_id"]
    type_ = obs_dict["Type"]
    date_ = obs_dict["Date"]
    policing_op_ = obs_dict["Part of a policing operation"]
    lat_ = obs_dict["Latitude"]
    long_ = obs_dict["Longitude"]
    gend_ = obs_dict["Gender"]
    age_range_ = obs_dict["Age range"]
    officer_def_ethnicity_ = obs_dict["Officer-defined ethnicity"]
    legislation_ = obs_dict["Legislation"]
    obj_search_ = obs_dict["Object of search"]
    station_ = obs_dict["station"]

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

    # Now do what we already learned in the notebooks about how to transform
    # a single observation into a dataframe that will work with a pipeline.
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    # Now get ourselves an actual prediction of the positive class.
    pred_proba = pipeline.predict_proba(obs)[0, 1]
    pred_outcome = pipeline.predict(obs).astype(bool)
    response = {'outcome': pred_outcome}
    p = Prediction(
        observation_id = observation_id_,
        type = type_,
        date= date_,
        part_of_a_policing_operation = policing_op_,
        latitude = lat_,
        longitude = long_,
        gender = gend_,
        age_range = age_range_,
        officer_defined_ethnicity = officer_def_ethnicity_,
        legislation = legislation_,
        object_of_search = obj_search_,
        station = station_,
        proba = pred_proba,
        outcome = pred_outcome
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = 'Observation ID: "{}" already exists'.format(_id)
        response['error'] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


@app.route('/search_result', methods=['POST'])
def search_result():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['id'])
        return jsonify({'error': error_msg})


@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
