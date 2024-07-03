import joblib 
import pickle
from pathlib import Path
from flask import Flask, request, jsonify


class PredictionPipeline:
    def __init__(self):
        with open(Path('research/model.bin'), 'rb') as f_in:
            (self.dv, self.model) = pickle.load(f_in)


    def prepare_features(self, ride):
        features = {}
        features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
        features['trip_distance'] = ride['trip_distance']
        return features


    def predict(self, features):
        X = self.dv.transform(features)
        preds = self.model.predict(X)
        return float(preds[0])
