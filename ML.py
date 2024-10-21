# Import necessary libraries
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('internet_service_churn.csv')

# Drop id column
df.drop(['id'], axis=1, inplace=True)

# Handling missing values
df.fillna(df.median(), inplace=True)  # Fill missing values with median

# Define features and target variable
X = df.drop('churn', axis=1)
y = df['churn']

# Feature scaling
X_scaled = (X - X.mean()) / X.std()  # Standardization

# Model Training
model = LogisticRegression()
model.fit(X_scaled, y)

# Streamlit Integration
# Streamlit app code for user interaction and model prediction
st.title('Customer Churn Prediction')

# User input for features
# Description for each input field
st.sidebar.header('Input Features')
st.sidebar.write('Please provide the following information:')

# Collect user input
user_input = {}

user_input['is_tv_subscriber'] = st.sidebar.number_input(
    label='Is TV Subscriber (1 for Yes, 0 for No)', 
    min_value=0, 
    max_value=1, 
    step=1,
    help='Indicate whether the customer has a TV subscription or not.'
)

user_input['is_movie_package_subscriber'] = st.sidebar.number_input(
    label='Is Movie Package Subscriber (1 for Yes, 0 for No)', 
    min_value=0, 
    max_value=1, 
    step=1,
    help='Indicate whether the customer has a cinema movie package subscription or not.'
)

user_input['subscription_age'] = st.sidebar.number_input(
    label='Subscription Age (Years)', 
    min_value=0,
    help='Enter the number of years the customer has been subscribed to the service.'
)

user_input['bill_avg'] = st.sidebar.number_input(
    label='Last 3 Months Bill Average', 
    min_value=0.0,
    help='Enter the average bill amount over the last 3 months.'
)

user_input['remaining_contract'] = st.sidebar.number_input(
    label='Remaining Contract (Years)', 
    min_value=0.0,
    help='Enter the remaining duration of the customer\'s contract in years. If the contract has expired or is non-existent, enter 0.'
)

user_input['service_failure_count'] = st.sidebar.number_input(
    label='Service Failure Count (Last 3 Months)', 
    min_value=0,
    help='Enter the number of times the customer has called the call center for service failures in the last 3 months.'
)

user_input['download_avg'] = st.sidebar.number_input(
    label='Download Average (Last 3 Months)', 
    min_value=0.0,
    help='Enter the average internet usage (in GB) over the last 3 months.'
)

user_input['upload_avg'] = st.sidebar.number_input(
    label='Upload Average (Last 3 Months)', 
    min_value=0.0,
    help='Enter the average upload usage (in GB) over the last 3 months.'
)

user_input['download_over_limit'] = st.sidebar.number_input(
    label='Download Over Limit (1 for Yes, 0 for No)', 
    min_value=0, 
    max_value=1, 
    step=1,
    help='Indicate whether the customer has exceeded the download limit (1 for Yes, 0 for No).'
)

# Button to trigger prediction
predict_button = st.sidebar.button('Predict')

if predict_button:
    # Convert user input to DataFrame
    user_input_df = pd.DataFrame([user_input])
    
    # Reorder columns to match the training data
    user_input_df = user_input_df[X.columns]
    
    # Fill missing values
    user_input_df.fillna(user_input_df.median(), inplace=True)
    
    # Scale user input
    user_input_scaled = (user_input_df - X.mean()) / X.std()
    
    # Predict churn probability
    prediction = model.predict_proba(user_input_scaled)[:, 1]  # Probability of churn
    
    # Define threshold for high churn probability
    threshold = 0.5
    
    # Display churn probability
    st.write('Churn Probability:', f"{prediction[0]:.2f}", "%")
    
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_[0]})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
    if prediction >= threshold:
        st.error('High likelihood of churn!')
    
        # Get top features influencing churn
        top_features = feature_importance.head(3)  # Let's consider the top 3 features
    
     # Display insights and recommendations
        st.subheader('Insights and Recommendations')

        for index, row in top_features.iterrows():
            feature_name = row['Feature']
            feature_importance_value = row['Importance']

        # Calculate the change in feature that would result in a 10% reduction in churn probability
            original_prediction = model.predict_proba(user_input_scaled)[:, 1]
            original_churn_probability = original_prediction[0]
        
            # Get the original feature value entered by the user
            original_feature_value = user_input.get(feature_name)
        
         # Calculate the required change in feature value to achieve a 10% reduction in churn probability
            if feature_importance_value != 0:
                change_needed = (10 / (feature_importance_value * original_churn_probability)) * original_feature_value
            else:
                # If feature importance is 0, no change is needed
                change_needed = 0

            # Skip factors with 0 change needed
            if change_needed == 0:
                continue

            # Adjust the change needed based on constraints or feasibility
            if feature_name == 'bill_avg':  # Change to match the key used in user input
                # Apply a constraint to keep the recommended action within a reasonable range
                max_bill_reduction = original_feature_value * 0.5  # Set maximum reduction to 50% of the original bill
                change_needed = min(change_needed, max_bill_reduction)
            elif feature_name in ['service_failure_count', 'download_over_limit']:  # Change to match the keys used in user input
              # Scale down the change needed proportionally based on the original feature value
                change_needed = min(change_needed, original_feature_value / 2)  # Set maximum reduction to half of the original value

            # Calculate the new feature value based on the adjusted change
            new_feature_value = original_feature_value - change_needed

            # Format recommendations in plain language
            st.write(f'**Feature:** {feature_name.replace("_", " ").title()}')
            st.write(f'**Feature Importance:** {feature_importance_value:.2f}')
            st.write(f'**Original Value:** {original_feature_value:.2f}')
            st.write(f'**Change Needed for 10% Reduction in Churn:** {change_needed:.2f}')

            # Check if the change is feasible and makes sense
            if change_needed != 0:
                st.write(f'**Recommended Action:** Consider adjusting {feature_name.replace("_", " ").title()} to {new_feature_value:.2f} to achieve a 10% reduction in churn.')
        st.write('---')
    else:
        st.success('Low likelihood of churn.')

    # Display relevant graphs
    st.header('Insights and Recommendations')

    # Correlation Analysis
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    correlation_fig = plt.gcf()  # Get the current figure
    st.pyplot(correlation_fig)

    # Feature Importance
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    feature_importance_fig = plt.gcf()  # Get the current figure
    st.pyplot(feature_importance_fig)
