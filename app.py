import joblib
import os
import sys
import numpy as np

# --- Configuration ---
MODEL_FILE = 'trained_linear_regression_model.pkl'
# Feature name inferred from your model file
INPUT_FEATURE_NAME = 'Years of Experience' 
OUTPUT_PREDICTION_NAME = 'Estimated Salary'

def load_model():
    """Loads the trained machine learning model."""
    try:
        if not os.path.exists(MODEL_FILE):
            print(f"Error: Model file '{MODEL_FILE}' not found.")
            print("Please ensure it is in the same directory as this script.")
            sys.exit(1)

        # Load the trained scikit-learn Linear Regression model
        pipeline = joblib.load(MODEL_FILE)
        print(f"Model '{MODEL_FILE}' loaded successfully.")
        return pipeline

    except Exception as e:
        print("-" * 50)
        print(f"CRITICAL ERROR: Failed to load model. Details: {e}")
        print("ACTION: This usually means a scikit-learn version mismatch.")
        print("-" * 50)
        sys.exit(1)

def main():
    """Main function to run the CLI tool."""
    pipeline = load_model()

    print(f"\n--- {OUTPUT_PREDICTION_NAME} Predictor CLI Tool ---")
    
    while True:
        try:
            # --- 1. Collect Input ---
            user_input = input(f"Enter {INPUT_FEATURE_NAME} (e.g., 5.5) or type 'quit': ")
            
            if user_input.lower() in ('quit', 'exit'):
                print("Exiting tool. Goodbye!")
                break
            
            # --- 2. Process and Predict ---
            try:
                # Convert input to float
                float_value = float(user_input)
                
                # Reshape data for prediction (1 sample, 1 feature)
                data_to_predict = np.array([[float_value]])
                
                # Make prediction
                prediction = pipeline.predict(data_to_predict)[0]
                
                # Format the output as a currency, rounded to 2 decimal places
                formatted_prediction = f"${prediction:,.2f}"
                
                # --- 3. Display Result ---
                print("\n" + "=" * 50)
                print(f"{INPUT_FEATURE_NAME} entered: {float_value}")
                print(f"Predicted {OUTPUT_PREDICTION_NAME}: \033[96m{formatted_prediction}\033[0m") # Cyan color
                print("=" * 50 + "\n")

            except ValueError:
                print("Invalid input. Please enter a valid number for experience.")
            except Exception as e:
                print(f"[Prediction Error] Could not classify data. Details: {e}")

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nExiting tool. Goodbye!")
            break
        except EOFError:
            print("\nExiting tool. Goodbye!")
            break

if __name__ == "__main__":
    main()
