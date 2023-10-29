# Import necessary libraries
import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained Random Forest model
model = joblib.load('random_forest_model.pkl')

# Define a function to get country encoding
def get_country_encoding(country_name):
    # Define the country encoding dictionary
    country_encoding = {
    'RUS': 0.37936508, 'PRT': 0.56295827, 'ARG': 0.25233645, 'FRA': 0.18584751, 'GBR': 0.20231023,
    'DEU': 0.16719286, 'BRA': 0.37353735, 'IRL': 0.24659158, 'USA': 0.23936933, 'KOR': 0.41353383,
    'AUT': 0.1821061, 'ITA': 0.35442701, 'BEL': 0.20239112, 'ESP': 0.25432243, 'AUS': 0.25117371,
    'LUX': 0.38111888, 'CHE': 0.24797219, 'EST': 0.21686747, 'NGA': 0.61764706, 'CN': 0.19859265,
    'NLD': 0.18402282, 'POL': 0.23420479, 'TUR': 0.41129032, 'DZA': 0.2038835, 'CYP': 0.21568627,
    'CUB': 0.0, 'SWE': 0.22254902, 'COL': 0.32394366, 'UKR': 0.29411765, 'NOR': 0.29818781,
    'ZAF': 0.3875, 'MAR': 0.42084942, 'CMR': 0.10714286, 'MYS': 0.08641975, 'LTU': 0.2804878,
    'IRN': 0.15470852, 'FIN': 0.30508475, 'THA': 0.2734375, 'GRC': 0.46246246, 'CHN': 0.23178808,
    'IND': 0.25261584, 'MDG': 0.68571429, 'ISR': 0.25057471, 'IDN': 0.07017544, 'DNK': 0.268,
    'ISL': 0.14213198, 'ROU': 0.24615385, 'JPN': 0.48717949, 'CHL': 0.6875, 'TUN': 0.11764706,
    'SAU': 0.84313725, 'MEX': 0.20689655, 'ARE': 0.46153846, 'PER': 0.56629834, 'VEN': 0.2962963,
    'AGO': 0.22222222, 'ECU': 0.28358209, 'OMN': 0.34375, 'IRQ': 0.71428571, 'MOZ': 0.21637427,
    'EGY': 0.02970297, 'AND': 0.16363636, 'CZE': 0.16, 'SRB': 0.14285714, 'LVA': 0.26785714,
    'BGR': 0.34615385, 'JOR': 0.5, 'SVN': 0.42105263, 'BLR': 0.57142857, 'CPV': 0.64285714,
    'SGP': 0.72727273, 'DOM': 0.9375, 'PAK': 0.2745098, 'UZB': 0.26315789, 'SEN': 0.25,
    'MAC': 0.33333333, 'TWN': 0.08108108, 'KAZ': 0.33478261, 'BFA': 0.36923077, 'HRV': 0.16666667,
    'ARM': 0.75, 'KEN': 0.05263158, 'NZL': 0.23076923, 'HUN': 1.0, 'GTM': 0.61111111,
    'SVK': 0.6, 'ALB': 0.11111111, 'GHA': 0.375, 'MDV': 0.29032258, 'ATA': 0.73333333,
    'ASM': 0.28125, 'PAN': 0.68181818, 'CRI': 0.52941176, 'BIH': 0.89655172, 'MUS': 0.27777778,
    'COM': 0.625, 'SUR': 0.8, 'JAM': 0.4, 'CAF': 0.88888889, 'ZWE': 0.2,
    'HND': 0.34782608695652173, 'RWA': 0.1791044776119403, 'GIB': 0.4117647058823529,
    'TZA': 0.30612244897959184, 'LIE': 0.20454545454545456, 'GNB': 0.36000000000000004,
    'LKA': 0.3591549295774648, 'KWT': 0.3496503496503497, 'MCO': 0.17142857142857143,
    'LBN': 0.32894736842105265, 'LBY': 0.17241379310344827, 'SYR': 0.2857142857142857,
    'QAT': 0.49206349206349204, 'TGO': 0.42857142857142855, 'UGA': 0.17592592592592593,
    'CIV': 0.3333333333333333, 'URY': 0.208955223880597, 'GEO': 0.2318840579710145,
    'AZE': 0.38095238095238093, 'HKG': 0.21153846153846154, 'ETH': 0.2891566265060241,
    'MLT': 0.2272727272727273, 'PHL': 0.32941176470588235, 'NPL': 0.33823529411764705,
    'BHS': 0.40625, 'ZMB': 0.23636363636363636, 'KHM': 0.3333333333333333, 'BGD': 0.34579439252336447,
    'IMN': 0.1678832116788321, 'BHR': 0.39473684210526316, 'MNE': 0.19166666666666668,
    'MLI': 0.2909090909090909, 'NAM': 0.25098039215686275, 'PRY': 0.34177215189873416,
    'MRT': 0.19117647058823528, 'FJI': 0.24864864864864863, 'PRI': 0.3894736842105263,
    'BDI': 0.1686746987951807, 'TJK': 0.2807017543859649, 'NIC': 0.1801801801801802,
    'BEN': 0.3220338983050847, 'UMI': 0.4, 'JEY': 0.23404255319148937, 'BOL': 0.22580645161290322,
    'ABW': 0.2833333333333333, 'NCL': 0.25, 'STP': 0.29166666666666663, 'MYT': 0.21428571428571427,
    'SYC': 0.19444444444444445, 'BRB': 0.26666666666666666, 'TMP': 0.23684210526315788,
    'KIR': 0.23529411764705882, 'SDN': 0.26851851851851855, 'VNM': 0.38666666666666666,
    'BWA': 0.3220338983050847, 'PLW': 0.17592592592592593, 'DJI': 0.2388888888888889,
    'GGY': 0.20526315789473685, 'FRO': 0.29850746268656714, 'AIA': 0.18888888888888888,
    'SLV': 0.2698412698412698, 'ATF': 0.2971014492753623, 'KNA': 0.21904761904761904,
    'SLE': 0.19365079365079365, 'VGB': 0.17619047619047618, 'GAB': 0.3111111111111111,
    'DMA': 0.21290322580645162, 'LAO': 0.3310344827586207, 'SMR': 0.17142857142857143,
    'PYF': 0.26515151515151514, 'CYM': 0.3628205128205128, 'MKD': 0.19230769230769232,
    'MMR': 0.35555555555555557, 'GUY': 0.2942028985507246, 'LCA': 0.2789473684210526,
    'MWI': 0.23114754098360654
    }
    return country_encoding.get(country_name)

# Define a function to preprocess user input and make predictions
def predict_cancellation(country, lead_time, previous_cancellations,
        previous_bookings_not_canceled, booking_changes,
        days_in_waiting_list, adr, required_car_parking_spaces,
        total_of_special_requests, total_customer, total_nights,
        deposit_given):
    # Get the country encoding for the selected country
    country_encoded = get_country_encoding(country)

    # Create a dictionary from user input
    user_input = {
        "country": country_encoded,
        "lead_time": lead_time,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "booking_changes": booking_changes,
        "days_in_waiting_list": days_in_waiting_list,
        "adr": adr,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,
        "total_customer": total_customer,
        "total_nights": total_nights,
        "deposit_given": deposit_given
    }

    # Create a DataFrame from the user input
    user_df = pd.DataFrame(user_input, index=[0])

    # You should apply the same preprocessing as done before training the model
    # Scaling the features is one common preprocessing step
    # scaler = StandardScaler()
    # user_df_scaled = scaler.fit_transform(user_df)
    st.write("user_df",user_df)
    # Make the prediction using the trained model
    prediction = model.predict(user_df)
    st.write("prediction",prediction[0])
    return prediction[0]

# Create a Streamlit web app
st.title('Hotel Booking Cancellation Prediction')

st.header('User Input')

# Get user inputs for the selected features
country_options = ['RUS', 'PRT', 'ARG', 'FRA', 'GBR', 'DEU', 'BRA', 'IRL', 'USA',
    'KOR', 'AUT', 'ITA', 'BEL', 'ESP', 'AUS', 'LUX', 'CHE', 'EST',
    'NGA', 'CN', 'NLD', 'POL', 'TUR', 'DZA', 'CYP', 'CUB', 'SWE',
    'COL', 'UKR', 'NOR', 'ZAF', 'MAR', 'CMR', 'MYS', 'LTU', 'IRN',
    'FIN', 'THA', 'GRC', 'CHN', 'IND', 'MDG', 'ISR', 'IDN', 'DNK',
    'ISL', 'ROU', 'JPN', 'CHL', 'TUN', 'SAU', 'MEX', 'ARE', 'PER',
    'VEN', 'AGO', 'ECU', 'OMN', 'IRQ', 'MOZ', 'EGY', 'AND', 'CZE',
    'SRB', 'LVA', 'BGR', 'JOR', 'SVN', 'BLR', 'CPV', 'SGP', 'DOM',
    'PAK', 'UZB', 'SEN', 'MAC', 'TWN', 'KAZ', 'BFA', 'HRV', 'ARM',
    'KEN', 'NZL', 'HUN', 'GTM', 'SVK', 'ALB', 'GHA', 'MDV', 'ATA',
    'ASM', 'PAN', 'CRI', 'BIH', 'MUS', 'COM', 'SUR', 'JAM', 'CAF',
    'ZWE', 'HND', 'RWA', 'GIB', 'TZA', 'LIE', 'GNB', 'LKA', 'KWT',
    'MCO', 'LBN', 'LBY', 'SYR', 'QAT', 'TGO', 'UGA', 'CIV', 'URY',
    'GEO', 'AZE', 'HKG', 'ETH', 'MLT', 'PHL', 'NPL', 'BHS', 'ZMB',
    'KHM', 'BGD', 'IMN', 'BHR', 'MNE', 'MLI', 'NAM', 'PRY', 'MRT',
    'FJI', 'PRI', 'BDI', 'TJK', 'NIC', 'BEN', 'UMI', 'JEY', 'BOL',
    'ABW', 'NCL', 'STP', 'MYT', 'SYC', 'BRB', 'TMP', 'KIR', 'SDN',
    'VNM', 'BWA', 'PLW', 'DJI', 'GGY', 'FRO', 'AIA', 'SLV', 'ATF',
    'KNA', 'SLE', 'VGB', 'GAB', 'DMA', 'LAO', 'SMR', 'PYF', 'CYM',
    'MKD', 'MMR', 'GUY', 'LCA', 'MWI']

country = st.selectbox("Country", country_options)
lead_time = st.number_input("Lead Time", min_value=0.0, format="%.6f")
previous_cancellations = st.number_input("Previous Cancellations", min_value=0, format="%d")
previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, format="%d")
booking_changes = st.number_input("Booking Changes", min_value=0, format="%d")
days_in_waiting_list = st.number_input("Days in Waiting List", min_value=0, format="%d")
adr = st.number_input("ADR (Average Daily Rate)", format="%.6f")
required_car_parking_spaces = st.number_input("Required Car Parking Spaces", min_value=0, format="%d")
total_of_special_requests = st.number_input("Total Special Requests", min_value=0, format="%d")
total_customer = st.number_input("Total Customer", format="%.6f")
total_nights = st.number_input("Total Nights", min_value=0, format="%d")
deposit_options = [0, 1, 2]
deposit_given = st.selectbox("Deposit Given (0: No Deposit, 1: Non Refund, 2: Refundable)", deposit_options)

# Make predictions
if st.button("Predict Cancellation"):
    # Print user input data for debugging or verification
    st.write("User Input Data:")
    st.write("Country:", country)
    st.write("Lead Time:", lead_time)
    st.write("Previous Cancellations:", previous_cancellations)
    st.write("Previous Bookings Not Canceled:", previous_bookings_not_canceled)
    st.write("Booking Changes:", booking_changes)
    st.write("Days in Waiting List:", days_in_waiting_list)
    st.write("ADR (Average Daily Rate):", adr)
    st.write("Required Car Parking Spaces:", required_car_parking_spaces)
    st.write("Total Special Requests:", total_of_special_requests)
    st.write("Total Customer:", total_customer)
    st.write("Total Nights:", total_nights)
    st.write("Deposit Given (0: No Deposit, 1: Non Refund, 2: Refundable):", deposit_given)

    prediction = predict_cancellation(country, lead_time, previous_cancellations,
        previous_bookings_not_canceled, booking_changes,
        days_in_waiting_list, adr, required_car_parking_spaces,
        total_of_special_requests, total_customer, total_nights,
        deposit_given)

    if prediction == 0:
        st.error("The booking is unlikely to be canceled.")
    else:
        st.success("The booking is likely to be canceled.")

# Provide additional information about the model and data
st.write("This is a simple Streamlit app to predict hotel booking cancellations.")
st.write("The model used is a Random Forest Classifier trained on a hotel booking dataset.")

# Disclaimer
st.write("**Disclaimer**: This prediction is based on a machine learning model and should not be used as the sole decision-making factor in real-world scenarios. Various other factors can influence hotel booking cancellations.")
