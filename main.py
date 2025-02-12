from supermarket_analysis import *

def main():
    # Load the data
    file_path = 'supermarket_sales.csv'  # Replace with your data file path
    df = load_data(file_path)
    
    # Preprocess the data
    X_train_scaled, X_test_scaled, y_train, y_test, feature_columns = preprocess_data(df)
    
    # Train the model
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate the model
    feature_importance = evaluate_model(model, X_test_scaled, y_test, feature_columns)
    
    # Plot the results
    y_pred = model.predict(X_test_scaled)
    plot_results(df, y_test, y_pred, feature_importance)

if __name__ == "__main__":
    main() 