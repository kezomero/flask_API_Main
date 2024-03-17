from flask import Flask, request, jsonify, current_app
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

app = Flask(__name__)
CORS(app)

lock = threading.Lock()  # Create a Lock
data = pd.read_csv('sorted.csv')  # Load data

@app.before_first_request
def train_model():
    global model, label_encoder, X_test, y_test

    target_column = 'label'
    label_encoder = LabelEncoder()
    data[target_column] = label_encoder.fit_transform(data[target_column])

    features = data.drop(target_column, axis=1)
    features = features.drop(['rainfall'], axis=1)
    target = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(features, target)

    current_app.model = model
    current_app.label_encoder = label_encoder
    current_app.X_train = X_train
    current_app.y_train = y_train
    current_app.X_test = X_test
    current_app.y_test = y_test

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('data')
        if not data:
            return jsonify({"error": "Data not provided"})

        with lock:  # Acquire the lock
            model = current_app.model
            if model is None:
                return jsonify({"message": "Model is not yet initialized", "success": False})

            label_encoder = current_app.label_encoder
            X_test = current_app.X_test
            y_test = current_app.y_test

            new_data = pd.DataFrame([data])
            new_data = new_data.applymap(convert_np_int64_to_int)

            if 'label' in new_data:
                new_data['label'] = label_encoder.transform([new_data['label'].iloc[0]])
            else:
                new_data['label'] = 0

            probability_predictions = model.predict_proba(new_data.drop('label', axis=1))

            classes = label_encoder.classes_
            probabilities = probability_predictions[0]
            crop_probabilities = dict(zip(classes, probabilities))

            threshold = 0.5
            potential_crops = {crop: prob for crop, prob in crop_probabilities.items() if prob >= threshold}

            response_data = {"success": True, "improvements": {}}

            predictions = model.predict(X_test)
            f1 = f1_score(y_test, predictions, average='weighted')
            accuracy = accuracy_score(y_test, predictions)

            response_data["metrics"] = {"f1_score": f1, "accuracy": accuracy}

            top_crop = max(potential_crops, key=potential_crops.get, default=None)
            top_probability = potential_crops.get(top_crop, 0.0)

            if potential_crops and top_probability >= 0.5:
                response_data["message"] = f"{top_crop} is the main crop that can be grown."

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

                    alternative_crops = {crop: prob for crop, prob in crop_probabilities.items() if 0 < prob < 0.5}

                    if alternative_crops:
                        alternative_crops = dict(sorted(alternative_crops.items(), key=lambda item: item[1], reverse=True))
                        response_data["alternative_crops"] = {}

                        for alt_crop, alt_prob in alternative_crops.items():
                            response_data["alternative_crops"][alt_crop] = {"probability": alt_prob, "suggested_improvements": []}

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

                alternative_crops = {crop: prob for crop, prob in crop_probabilities.items() if 0 < prob < 0.5}

                if alternative_crops:
                    alternative_crops = dict(sorted(alternative_crops.items(), key=lambda item: item[1], reverse=True))
                    response_data["alternative_crops"] = {}

                    for alt_crop, alt_prob in alternative_crops.items():
                        response_data["alternative_crops"][alt_crop] = {"probability": alt_prob, "suggested_improvements": []}

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
        return jsonify({"error": str(e)})
