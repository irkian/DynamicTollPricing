# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import time
# import sklearn

# # Load the model and encoders
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)
# with open('encoders.pkl', 'rb') as f:
#     encoders = pickle.load(f)

# # Define a mapping for direction labels based on encoder's classes
# direction_mapping = {
#     'Northbound': encoders['direction'].classes_[0],  # Change [0] if necessary
#     'Southbound': encoders['direction'].classes_[1]   # Change [1] if necessary
# }

# # Mapping for day of week
# day_of_week_mapping = {
#     'Sunday': 0,
#     'Monday': 1,
#     'Tuesday': 2,
#     'Wednesday': 3,
#     'Thursday': 4,
#     'Friday': 5,
#     'Saturday': 6
# }

# # Function to determine traffic volume based on time
# def get_traffic_volume(hour):
#     if 7 <= hour <= 9 or 15 <= hour <= 18:
#         return np.random.randint(600, 801)
#     elif 9 < hour < 12 or 18 < hour < 21:
#         return np.random.randint(400, 601)
#     else:
#         return np.random.randint(200, 401)

# # Function to make predictions with original values
# def predict_with_original_values(direction, start_point, end_point, traffic_volume, hour, day_of_week):
#     # Map and encode the input values
#     direction_encoded = encoders['direction'].transform([direction_mapping[direction]])[0]
#     start_point_encoded = encoders['start_point'].transform([start_point])[0]
#     end_point_encoded = encoders['end_point'].transform([end_point])[0]
#     day_of_week_encoded = day_of_week_mapping[day_of_week]

#     # Create the input array
#     input_array = [[direction_encoded, start_point_encoded, end_point_encoded, traffic_volume, hour, day_of_week_encoded]]

#     # Make prediction
#     prediction = model.predict(input_array)
#     return prediction[0]

# # Function to format the hour in 12-hour format with AM/PM
# def format_hour(hour):
#     hour_12 = hour % 12
#     if hour_12 == 0:
#         hour_12 = 12
#     period = "AM" if hour < 12 else "PM"
#     return f"{hour_12} {period}"

# # Streamlit App
# st.title("Dynamic Toll Pricing")

# # User inputs
# direction = st.selectbox('Direction', options=['Northbound', 'Southbound'])
# start_point = st.selectbox('Start Point', options=encoders['start_point'].classes_)
# end_point = st.selectbox('End Point', options=encoders['end_point'].classes_)
# day_of_week = st.selectbox('Day of the Week', options=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

# # Centering the run button
# st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
# if st.button('Run'):
#     st.markdown("</div>", unsafe_allow_html=True)
    
#     # Placeholder for dynamic results
#     placeholder = st.empty()
#     results = []
    
#     # Initial traffic volume
#     traffic_volume = get_traffic_volume(0)  # Starting with initial traffic volume

#     for minute in range(0, 1440, 15):  # 1440 minutes in a day, 15-minute intervals
#         hour = minute // 60
#         formatted_hour = format_hour(hour)
#         predicted_rate = predict_with_original_values(direction, start_point, end_point, int(traffic_volume), hour, day_of_week)
        

#         # Adjust traffic volume based on the predicted rate and keep it within the specified ranges
#         traffic_volume = get_traffic_volume(hour)

#         # Append the results to the list
#         results.append({
#             'Time': formatted_hour,
#             'Traffic Volume': int(traffic_volume),
#             'Predicted Rate': round(predicted_rate, 2)
#         })

#         # Update dynamic display
#         with placeholder.container():
#             st.subheader("Dynamic Prediction")
#             col1, col2, col3 = st.columns(3)
#             col1.metric("Time", formatted_hour)
#             col2.metric("Traffic Volume", int(traffic_volume))
#             col3.metric("Toll Rate", round(predicted_rate, 2))
#             time.sleep(1)  # Simulate dynamic change

#     # Commenting out the display of all results at the end
#     # st.subheader("Predicted and Adjusted Rates Over 24 Hours")
#     # for result in results:
#     #     st.write(f"Time: {result['Time']}, Traffic Volume: {result['Traffic Volume']}, Toll Rate: ${result['Predicted Rate']}")
# else:
#     st.markdown("</div>", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

# Load the model and encoders
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Define a mapping for direction labels based on encoder's classes
direction_mapping = {
    'Northbound': encoders['direction'].classes_[0],  # Change [0] if necessary
    'Southbound': encoders['direction'].classes_[1]   # Change [1] if necessary
}

# Mapping for day of week
day_of_week_mapping = {
    'Sunday': 0,
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6
}

# Function to determine traffic volume based on time
def get_traffic_volume(hour):
    if 7 <= hour <= 9 or 15 <= hour <= 18:
        return np.random.randint(600, 801)
    elif 9 < hour < 12 or 18 < hour < 21:
        return np.random.randint(400, 601)
    else:
        return np.random.randint(200, 401)

# Function to make predictions with original values
def predict_with_original_values(direction, start_point, end_point, traffic_volume, hour, day_of_week):
    # Map and encode the input values
    direction_encoded = encoders['direction'].transform([direction_mapping[direction]])[0]
    start_point_encoded = encoders['start_point'].transform([start_point])[0]
    end_point_encoded = encoders['end_point'].transform([end_point])[0]
    day_of_week_encoded = day_of_week_mapping[day_of_week]

    # Create the input array
    input_array = [[direction_encoded, start_point_encoded, end_point_encoded, traffic_volume, hour, day_of_week_encoded]]

    # Make prediction
    prediction = model.predict(input_array)
    return prediction[0]

# Function to format the hour in 12-hour format with AM/PM
def format_hour(hour):
    hour_12 = hour % 12
    if hour_12 == 0:
        hour_12 = 12
    period = "AM" if hour < 12 else "PM"
    return f"{hour_12} {period}"

# Streamlit App
st.title("Dynamic Toll Pricing")

# User inputs
direction = st.selectbox('Direction', options=['Northbound', 'Southbound'])
start_point = st.selectbox('Start Point', options=encoders['start_point'].classes_)
end_point = st.selectbox('End Point', options=encoders['end_point'].classes_)
day_of_week = st.selectbox('Day of the Week', options=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

# Centering the run button
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
if st.button('Run'):
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Placeholder for dynamic results
    placeholder = st.empty()
    results = []
    
    # Initial traffic volume
    traffic_volume = get_traffic_volume(0)  # Starting with initial traffic volume

    for minute in range(0, 1440, 15):  # 1440 minutes in a day, 15-minute intervals
        hour = minute // 60
        formatted_hour = format_hour(hour)
        predicted_rate = predict_with_original_values(direction, start_point, end_point, int(traffic_volume), hour, day_of_week)
        
        # Adjust traffic volume based on the predicted rate and keep it within the specified ranges
        traffic_volume = get_traffic_volume(hour)

        # Append the results to the list
        results.append({
            'Time': formatted_hour,
            'Traffic Volume': int(traffic_volume),
            'Predicted Rate': round(predicted_rate, 2)
        })

        # Update dynamic display
        with placeholder.container():
            st.subheader("Dynamic Prediction")
            col1, col2, col3 = st.columns(3)
            col1.metric("Time", formatted_hour)
            col2.metric("Traffic Volume", int(traffic_volume))
            col3.metric("Toll Rate", round(predicted_rate, 2))
            time.sleep(1)  # Simulate dynamic change

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Plot graphs using Matplotlib
    st.subheader("Change of Toll Price Over Time")
    plt.figure(figsize=(10, 5))
    plt.plot(df['Time'], df['Predicted Rate'], color='red')
    plt.xlabel('Time')
    plt.ylabel('Toll Price')
    plt.title('Change of Toll Price Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    st.subheader("Change of Traffic Volume Over Time")
    plt.figure(figsize=(10, 5))
    plt.plot(df['Time'], df['Traffic Volume'], color='green')
    plt.xlabel('Time')
    plt.ylabel('Traffic Volume')
    plt.title('Change of Traffic Volume Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)
else:
    st.markdown("</div>", unsafe_allow_html=True)
