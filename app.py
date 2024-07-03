from flask import Flask, render_template, request, jsonify
import os 
from mlProject.pipeline.predict import PredictionPipeline


# app = Flask(__name__) # initializing a flask app

app = Flask('duration-prediction')

# @app.route('/train',methods=['GET'])  # route to train the pipeline
# def training():
#     os.system("python main.py")
#     return "Training Successful!" 

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    obj = PredictionPipeline()
    features = obj.prepare_features(ride)
    pred = obj.predict(features)

    result = {
        # 'ride': ride,
        'duration': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)