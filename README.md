# rbs-ranking
Ranking problem to rank legal representatives (rv) regarding legal hearings.

## Classifying Features
Timeline of rv, gender of rv, type of event, place of event, gender of client..

## Libraries
- tensorflow-ranking 
- sklearn random forest classifier

## Procedure
- import event and rv data from Algoterm
- data preperation ("sampling")
- split into train and test data
- prediction
- evaluation

## Hyperparameter evaluation
Hyperparameters can be modified in globalVars.py

## Installation
Install using pip: `python -m pip install requirements.txt`

## Tests
Sampling tests: `python -m unittest rvranking/sampling/tests.py`