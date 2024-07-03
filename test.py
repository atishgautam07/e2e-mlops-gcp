from mlProject.pipeline.predict import PredictionPipeline
import requests

ride = {
    "PULocationID": 50,
    "DOLocationID": 50,
    "trip_distance": 10
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())

# obj = PredictionPipeline()
# features = obj.prepare_features(ride)
# print (features)
# pred = obj.predict(features)

# result = {
#     'ride': ride,
#     'duration': pred
# }

# print(result)