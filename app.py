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

# import aux functions

from app_functions import attempt_predict

########################################
# Begin database stuff

# The connect function checks if there is a DATABASE_URL env var.
# If it exists, it uses it to connect to a remote postgres db.
# Otherwise, it connects to a local sqliåte db stored in predictions.db.
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    # observation_id = IntegerField(unique=True)
    # observation = TextField()
    observation_id = TextField(unique=True)
    type = TextField()
    date = TextField()
    part_of_a_policing_operation = BooleanField(null=True)
    latitude = FloatField(null=True)
    longitude = FloatField(null=True)
    gender = TextField()
    age_range = TextField()
    officer_defined_ethnicity = TextField()
    legislation = TextField(null=True)
    object_of_search = TextField()
    station = TextField()
    proba = FloatField()
    outcome = BooleanField(null=True)
    true_outcome = BooleanField(null=True)

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


@app.route('/should_search/', methods=['POST'])
def should_search():
    # Flask provides a deserialization convenience function called
    # get_json that will work if the mimetype is application/json.
    obs_dict = request.get_json()
    
    observation, check =   attempt_predict(obs_dict)

    if not check:  # IF check is false, then one of the integrity tests didn't succeed
        return jsonify(observation)

    ################################################################################################################################################################################################

    # Now do what we already learned in the notebooks about how to transform
    # a single observation into a dataframe that will work with a pipeline.
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    # Now get ourselves an actual prediction of the positive class.

    pred_proba = pipeline.predict_proba(obs)[0, 1]

    # considering threshold of 0.4
    if pred_proba < 0.4:
        pred_outcome = False
    else:
        pred_outcome = True

    response = {'outcome': str(pred_outcome)}

    p = Prediction(
        observation_id = observation["observation_id"],
        type = observation["Type"],
        date = observation["Date"],
        part_of_a_policing_operation = observation["Part of a policing operation"],
        latitude = observation["Latitude"],
        longitude = observation["Longitude"],
        gender = observation["Gender"],
        age_range = observation["Age range"],
        officer_defined_ethnicity = observation["Officer-defined ethnicity"],
        legislation = observation["Legislation"],
        object_of_search = observation["Object of search"],
        station = observation["station"],
        proba = pred_proba,
        outcome = pred_outcome
    )
    try:
        p.save()
    except IntegrityError:
        error_msg = 'Observation ID: "{}" already exists'.format(obs_dict["observation_id"])
        response['error'] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


@app.route('/search_result/', methods=['POST'])
def search_result():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.true_outcome = obs['outcome']
        p.save()

        response = { "observation_id": p.observation_id,
                "outcome": str(p.true_outcome),
                "predicted_outcome": str(p.outcome)}

        # return jsonify(model_to_dict(p))
        return jsonify(response)
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
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
