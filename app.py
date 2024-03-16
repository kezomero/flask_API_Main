from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
import threading
from threading import Lock  # Import Lock

app = Flask(__name__)
CORS(app)

feature_importances = None
model = None
label_encoder = None
X_train = None
y_train = None
cancel_prediction = False
lock = Lock()  # Create a Lock

data = pd.read_csv('/home/sir-derrick/Desktop/4.1/sorted.csv')
def train_model():
    global model, label_encoder, feature_importances, X_train, y_train, X_test, y_test

    target_column = 'label'

    label_encoder = LabelEncoder()
    data[target_column] = label_encoder.fit_transform(data[target_column])

    features = data.drop(target_column, axis=1)
    features = features.drop(['rainfall'], axis=1)
    target = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(features, target)

    feature_names = features.columns
    perm_importance = permutation_importance(model, features, target, n_repeats=30, random_state=42)
    feature_importances = dict(zip(feature_names, perm_importance.importances_mean))

# Start training the model in a separate thread
train_thread = threading.Thread(target=train_model)
train_thread.start()

def convert_np_int64_to_int(value):
    if isinstance(value, np.int64):
        return int(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, dict):
        return {k: convert_np_int64_to_int(v) for k, v in value.items()}
    return value

def calculate_improvements(current_value, target_value):
    return target_value - current_value

def perform_prediction(data):
    global cancel_prediction

    with lock:  # Acquire the lock
        try:
            if cancel_prediction:
                cancel_prediction = False 
                return {"message": "Prediction canceled"}

            new_data = pd.DataFrame([data])
            new_data = new_data.applymap(convert_np_int64_to_int)

            if 'label' in new_data:
                new_data['label'] = label_encoder.transform([new_data['label'].iloc[0]])
            else:
                new_data['label'] = 0

            if cancel_prediction:
                cancel_prediction = False 
                return {"message": "Prediction canceled"}

            probability_predictions = model.predict_proba(new_data.drop('label', axis=1))

            if cancel_prediction:
                cancel_prediction = False 
                return {"message": "Prediction canceled"}

            classes = label_encoder.classes_
            probabilities = probability_predictions[0]
            crop_probabilities = dict(zip(classes, probabilities))

            threshold = 0.5
            potential_crops = {crop: prob for crop, prob in crop_probabilities.items() if prob >= threshold}

            response_data = {"success": True, "improvements": {}}

            if cancel_prediction:
                cancel_prediction = False 
                return {"message": "Prediction canceled"}
            # Calculate F1 score and accuracy on the test set
            predictions = model.predict(X_test)
            f1 = f1_score(y_test, predictions, average='weighted')
            accuracy = accuracy_score(y_test, predictions)

            response_data["metrics"] = {"f1_score": f1, "accuracy": accuracy}

            top_crop = max(potential_crops, key=potential_crops.get, default=None)
            top_probability = potential_crops.get(top_crop, 0.0)

            if potential_crops and top_probability >= 0.5:
                response_data["message"] = f"{top_crop} is the main crop that can be grown."

                if cancel_prediction:
                    cancel_prediction = False 
                    return {"message": "Prediction canceled"}

                feature_importances = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42).importances_mean
                feature_names = X_test.columns

                if feature_importances is not None:
                    significant_features = [(feature_names[i], feature_importances[i]) for i, importance in enumerate(feature_importances) if importance > 0]

                    if significant_features:
                        response_data["improvements"][top_crop] = {"probability": top_probability, "suggested_improvements": []}

                        for feature, _ in significant_features:
                            target_value = X_train[feature].mean()
                            current_value = new_data[feature].values[0]
                            improvement = calculate_improvements(current_value, target_value)

                            response_data["improvements"][top_crop]["suggested_improvements"].append({
                                "feature": feature,
                                "current_value": convert_np_int64_to_int(current_value),
                                "target_value": convert_np_int64_to_int(target_value),
                                "improvement": convert_np_int64_to_int(improvement)
                            })

                    else:
                        response_data["improvements"][top_crop]["message"] = "No specific improvement recommendations."

                    if cancel_prediction:
                        cancel_prediction = False 
                        return {"message": "Prediction canceled"}

                    alternative_crops = {crop: prob for crop, prob in crop_probabilities.items() if 0 < prob < 0.5}

                    if alternative_crops:
                        alternative_crops = dict(sorted(alternative_crops.items(), key=lambda item: item[1], reverse=True))
                        response_data["alternative_crops"] = {}

                        for alt_crop, alt_prob in alternative_crops.items():
                            response_data["alternative_crops"][alt_crop] = {"probability": alt_prob, "suggested_improvements": []}

                            if cancel_prediction:
                                cancel_prediction = False 
                                return {"message": "Prediction canceled"}

                            alt_significant_features = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42).importances_mean
                            feature_names = X_test.columns

                            if alt_significant_features is not None:
                                alt_significant_features = [(feature_names[i], alt_significant_features[i]) for i, importance in enumerate(alt_significant_features) if importance > 0]

                            alt_crop = alt_crop

                            if alt_significant_features:
                                for feature, _ in alt_significant_features:
                                    target_value = X_train[feature].mean()
                                    current_value = new_data[feature].values[0]
                                    improvement = calculate_improvements(current_value, target_value)

                                    response_data["alternative_crops"][alt_crop]["suggested_improvements"].append({
                                        "feature": feature,
                                        "current_value": convert_np_int64_to_int(current_value),
                                        "target_value": convert_np_int64_to_int(target_value),
                                        "improvement": convert_np_int64_to_int(improvement)
                                    })

                            else:
                                response_data["alternative_crops"][alt_crop]["message"] = "No specific improvement recommendations."

                    else:
                        response_data["message"] = "No alternative crops found with a higher probability threshold."

                else:
                    response_data["message"] = "No feature importances available."

            else:
                response_data["message"] = "Soil is not suitable for a crop with a 50% probability"

                if cancel_prediction:
                    cancel_prediction = False 
                    return {"message": "Prediction canceled"}

                alternative_crops = {crop: prob for crop, prob in crop_probabilities.items() if 0 < prob < 0.5}

                if alternative_crops:
                    alternative_crops = dict(sorted(alternative_crops.items(), key=lambda item: item[1], reverse=True))
                    response_data["alternative_crops"] = {}

                    for alt_crop, alt_prob in alternative_crops.items():
                        response_data["alternative_crops"][alt_crop] = {"probability": alt_prob, "suggested_improvements": []}

                        if cancel_prediction:
                            cancel_prediction = False 
                            return {"message": "Prediction canceled"}

                        alt_significant_features = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42).importances_mean
                        feature_names = X_test.columns

                        if alt_significant_features is not None:
                            alt_significant_features = [(feature_names[i], alt_significant_features[i]) for i, importance in enumerate(alt_significant_features) if importance > 0]

                        alt_crop = alt_crop

                        if alt_significant_features:
                            for feature, _ in alt_significant_features:
                                target_value = X_train[feature].mean()
                                current_value = new_data[feature].values[0]
                                improvement = calculate_improvements(current_value, target_value)

                                response_data["alternative_crops"][alt_crop]["suggested_improvements"].append({
                                    "feature": feature,
                                    "current_value": convert_np_int64_to_int(current_value),
                                    "target_value": convert_np_int64_to_int(target_value),
                                    "improvement": convert_np_int64_to_int(improvement)
                                })

                            else:
                                response_data["alternative_crops"][alt_crop]["message"] = "No specific improvement recommendations."

                    # print("No alternative crops.")

                else:
                    response_data["message"] = "No more alternative crops found with a higher probability threshold."

            return response_data

        except Exception as e:
            if cancel_prediction:
                return {"message": "Prediction canceled"}
            cancel_prediction = False  # Reset the flag in case of unexpected exceptions
            return {"success": False, "message": str(e)}

def get_crop_characteristics(crop_label):
    try:
        # Transform the crop label to the encoded value
        encoded_crop_label = label_encoder.transform([crop_label])[0]

        # Check if the encoded crop label exists in the data
        if encoded_crop_label in data['label'].values:
            # Retrieve data for the selected encoded crop label
            crop_data = data[data['label'] == encoded_crop_label]

            # Calculate mean for numeric columns, mode for non-numeric columns
            crop_characteristics = {
                "N": round(crop_data['N'].mean(), 5),
                "P": round(crop_data['P'].mean(), 5),
                "K": round(crop_data['K'].mean(), 5),
                "temperature": round(crop_data['temperature'].mean(), 5),
                "humidity": round(crop_data['humidity'].mean(), 5),
                "ph": round(crop_data['ph'].mean(), 5),
                "rainfall": round(crop_data['rainfall'].mean(), 5),
                "label": label_encoder.inverse_transform([crop_data['label'].mode().iloc[0]])[0],  # Decode the label
            }

            return crop_characteristics
        else:
            return {}
    except Exception as e:
        return {}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('data')
        result = perform_prediction(data)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}) 
# New endpoint to get crop labels
@app.route('/crop_labels', methods=['GET'])
def get_crop_labels():
    try:
        crop_labels = label_encoder.classes_.tolist()
        return jsonify({"crop_labels": crop_labels})

    except Exception as e:
        return jsonify({"error": str(e)})
# New endpoint to cancel prediction
@app.route('/cancel', methods=['POST'])
def cancel():
    global cancel_prediction
    cancel_prediction = True
    return jsonify({"message": "Cancel request received"})

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json.get('data')
        crop_label = data.get('crop_label')

        if not crop_label:
            return jsonify({"error": "Crop label not provided"})

        crop_characteristics = get_crop_characteristics(crop_label)

        if not crop_characteristics:
            print("Crop label not found or characteristics not available for:", crop_label)  # Add this line for debugging
            return jsonify({"error": "Crop label not found or characteristics not available"})

        return jsonify({"success": True, "characteristics": crop_characteristics})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
