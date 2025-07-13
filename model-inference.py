import pandas as pd
import xgboost as xgb
import time
import os
import sys

def calculate_all_features_for_group(group):
    nocturnal_hours = (group['hour_of_day'] >= 22) | (group['hour_of_day'] <= 6)
    features = {
        'total_calls': len(group),
        'outgoing_call_ratio': (group['call_direction'] == 'outgoing').mean(),
        'avg_duration': group['duration'].mean(),
        'std_duration': group['duration'].std(),
        'nocturnal_call_ratio': nocturnal_hours.mean(),
        'mobility': group['cell_tower'].nunique(),
    }
    return pd.Series(features)

def feature_engineering_pandas_inference(df):
    print("Performing feature engineering for inference with pandas...")
    # Ensure 'msisdn' is the index for grouping
    if 'msisdn' in df.columns:
        df = df.set_index('msisdn')
    
    # Group by user (msisdn) and apply the feature calculation function
    user_features_df = df.groupby('msisdn').apply(calculate_all_features_for_group)

    return user_features_df

def infer_from_model(booster, features_df):
    """
    Performs inference using the loaded XGBoost model.
    """
    print("\nPerforming inference with the loaded model...")
    
    # Fill any potential NaN values that arose from aggregations
    features_df = features_df.fillna(0)

    dmatrix_inference = xgb.DMatrix(features_df)

    # Predict probabilities for the positive class (fraud)
    predictions = booster.predict(dmatrix_inference)
    
    # Add predictions to the DataFrame
    features_df['fraud_probability'] = predictions
    
    features_df['is_predicted_fraud'] = (predictions > 0.5).astype(int)
    
    return features_df

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python infer_fraud_model.py <path_to_input_csv>")
        print("Example: python infer_fraud_model.py new_unseen_cdr_data.csv")
        sys.exit(1) # Exit the script if the argument is not provided

    inference_data_filename = sys.argv[1]
    model_input_filename = 'fraud_detection_model_xgb2.json'
    base_name = os.path.basename(inference_data_filename)
    name, ext = os.path.splitext(base_name)
    predictions_output_filename = f'predictions_{name}.csv'
    
    if not os.path.exists(model_input_filename):
        print(f"Error: Model file not found at '{model_input_filename}'.")
        print("Please ensure the trained model file is in the correct directory.")
        exit()
        
    print(f"Loading model from '{model_input_filename}'...")
    booster = xgb.Booster()
    booster.load_model(model_input_filename)
    print("Model loaded successfully.")

    try:
        print(f"\nReading inference data '{inference_data_filename}' with pandas...")
        raw_inference_df = pd.read_csv(inference_data_filename)
    except FileNotFoundError:
        print(f"Error: Inference data file not found at '{inference_data_filename}'.")
        print("Please ensure the path and filename are correct.")
        exit()

    start_time = time.time()
    inference_features_df = feature_engineering_pandas_inference(raw_inference_df)
    predictions_df = infer_from_model(booster, inference_features_df)
    print(f"\nSaving predictions to '{predictions_output_filename}'...")
    predictions_df.to_csv(predictions_output_filename)

    fraudulent_predictions = predictions_df[predictions_df['is_predicted_fraud'] == 1]
    num_fraudulent = len(fraudulent_predictions)

    print(f"\n--- Inference Summary ---")
    print(f"Total MSISDNs processed: {len(predictions_df)}")
    print(f"Number of MSISDNs predicted as fraudulent: {num_fraudulent}")

    if num_fraudulent > 0:
        print("\nTop 5 most likely fraudulent MSISDNs (by probability):")
        print(fraudulent_predictions.sort_values(by='fraud_probability', ascending=False).head())
        print("\nReasoning: Predictions are based on the model learning from features such as:")
        print("total_calls, outgoing_call_ratio, avg_duration, std_duration, nocturnal_call_ratio, and mobility.")
        print("High fraud probability suggests these users' activity patterns resemble those of known fraud cases in the training data.")
    else:
        print("\nNo fraudulent activity was predicted in this dataset.")

    print("--------------------------")

    print(f"\nInference process complete in {time.time() - start_time:.2f} seconds.")
