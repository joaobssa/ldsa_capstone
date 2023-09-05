# ldsa_capstone

## Project Summary

Disclaimer: The below description and all scenarios in this repository are just hypothetical scenarios and they do not reflect reality

This project was developed as part of the [Lisbon Data Science Starters Academy](https://www.lisbondatascience.org/starters-academy/). More information about this organization can be found [here](https://www.lisbondatascience.org/about-us/).

A police department has been dealing with accusations of racial, gender, and age discrimination in its stop-and-search operations. This police department is composed of several stations that are distributed geographically. Patrolling police officers are supposed to stop and search individuals or vehicles based on probable cause. The officer will usually state the object of the search and the legislation that supports such object before interpellating the vehicle or individual. Identifying probable cause can be subjective, and affected by pre-existing bias. As such, the police department has requested our assistance on two separate issues.

First, we need to provide an unbiased analysis of the data that they have gathered for past stop-and-search situations and identify whether there is any evidence of discrimination against the protected classes - ethnicity, and gender. They would like this analysis to be broken down by department. The UKDP would also like to understand whether there is any bias on requests to remove more than outer clothing against the following protected classes - age, ethnicity, and gender.

We will consider proof of discrimination when the rate of success between the protected class with the highest rate and the one with the lowest rate is higher than 5 percentage points. If the success rate of search for Female individuals is 5% and the success rate for Male individuals is 15%, the difference between these is 10% which indicates there could be some bias that leads to more unwarranted searches of Female individuals. This difference should be level across all stations and objects of search.

The second request involves creating a system, in the form of a resting Application Programming Interface (API) that approves stop-and-search operations based on the patrolling officer’s input. This system is aimed at leveling the success rate of stop-and-search activities across protected classes and search objectives, providing an extra layer of confidence against discrimination. Furthermore, the leveling of discovery across protected classes should not diminish the ability to detect offenses significantly.

We will consider that the system should approve stop-and-search with at least 80% recall. This means that the model will be able to capture the large majority of offenses.

## Results Summary

We were able to deliver a model with a recall higher than 0.80 and a decision threshold higher than 0.1 as requested by the client, but we were not able to level the discovery rate across race-gender tuples below 0.05. We believe that this is due to the bias that is present in the dataset that we were not able to remove completely. Ultimately we decided on keeping as much data as possible to try and capture any underlying subtleties. This might not have been a good approach since the model seemed to pick some bias from the data and was unable to completely level the discovery rate.

The final model was deployed as a resting Application Programming Interface (API) on railway.app under the flask framework with two modules - /should_search/, which handles prediction requests, and /search_result/which handles updates to a record with the true outcome result. The requests are stored on a PostgreSQL database.

## Contents

In the main folder, you will find all notebooks used for EDA, developing and improving the model, and evaluating the results. They are numbered from 1 to 15 in the order that they were created during the project.

You will also find the outputs of the model that are picked up by the API while handling requests. These are:
* pipeline.pickle - pickle that includes the trained model pipeline. This is used by the API to make predictions.
* dtypes.pickle - lists the expected types of each feature considered by the model
* columns.json - json file that holds a list of the necessary features for the model to produce a prediction

Also in the main folder, you will find Python files:
* app.py - defines the API main routines
* app_functions.py - holds necessary functions for the API to handle some tasks such as predicting and validating requests
* aux_functions.py - holds all recurring functions used in the notebooks for cleaning data and evaluating results
* transformes.py - defines custom transformers used by the model for training and predicting

You can also find here some general-purpose files:
* requirements.txt - holds the necessary library requirements to train the model. Use these for running the notebooks
* requirements_prod.txt - holds the necessary library requirements to run the API
* Dockerfile - used to handle API initialization

Finally, you can see below a description of the subfolders:
* Report - In this folder, you'll find the reports produced to complete the [Lisbon Data Science Starters Academy](https://www.lisbondatascience.org/starters-academy/). Check these out to get further information on the project and a discussion of the main findings. Once again, all references to real entities are solely for the project and do not reflect reality.
* Results - holds files used while evaluating the model
