import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and preprocess data
def load_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract date features
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    return df

def preprocess_data(df):
    # Handle missing values
    df = df.fillna(df.select_dtypes(include=[np.number]).mean())
    
    # Create dummy variables for categorical columns
    categorical_columns = ['Branch', 'Customer_Type', 'Payment']
    df = pd.get_dummies(df, columns=categorical_columns)
    
    # Select features for regression
    feature_columns = [col for col in df.columns if col not in ['Date', 'Total']]
    X = df[feature_columns]
    y = df['Total']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, feature_columns):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': abs(model.coef_)
    })
    return feature_importance.sort_values('Importance', ascending=False)

def plot_results(df, y_test, y_pred, feature_importance):
    # Create a figure with multiple subplots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    
    # Plot 2: Feature Importance
    plt.subplot(2, 2, 2)
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importance')
    
    # Plot 3: Sales Distribution
    plt.subplot(2, 2, 3)
    sns.histplot(df['Total'], bins=50)
    plt.title('Distribution of Sales')
    
    # Plot 4: Monthly Sales Trend
    monthly_sales = df.groupby('Month')['Total'].mean()
    plt.subplot(2, 2, 4)
    monthly_sales.plot(kind='line', marker='o')
    plt.title('Average Monthly Sales')
    
    plt.tight_layout()
    plt.show()