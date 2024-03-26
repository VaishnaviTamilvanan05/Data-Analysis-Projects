#%%
# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")




#%%
#read dataset and show the shape
airline_df =pd.read_csv("Airline_Passenger_Satisfaction.csv")
airline_df.shape

# %%
#show the first 5 rows of the dataset
airline_df.head()

# %%

data_info = airline_df.info()

# Extract column names and data types from the info() output
column_info = [(col, dtype) for col, dtype in zip(airline_df.columns, airline_df.dtypes)]

# Print column names, data types
print("Features in the data:")
for col, dtype in column_info:
    print(f"- {col}: {dtype}")

# %%
print("""Description of Features in the data:
- Gender: Gender of the passengers (Female, Male)
- Customer Type: The customer type (Loyal customer, disloyal customer)
- Age: The actual age of the passengers
- Type of Travel: Purpose of the flight of the passengers (Personal Travel, Business Travel)
- Class: Travel class in the plane of the passengers (Business, Eco, Eco Plus)
- Flight distance: The flight distance of this journey
- Inflight wifi service: Satisfaction level of the inflight wifi service (0:Not Applicable;1-5)
- Departure/Arrival time convenient: Satisfaction level of Departure/Arrival time convenient
- Ease of Online booking: Satisfaction level of online booking
- Gate location: Satisfaction level of Gate location
- Food and drink: Satisfaction level of Food and drink
- Online boarding: Satisfaction level of online boarding
- Seat comfort: Satisfaction level of Seat comfort
- Inflight entertainment: Satisfaction level of inflight entertainment
- On-board service: Satisfaction level of On-board service
- Leg room service: Satisfaction level of Leg room service
- Baggage handling: Satisfaction level of baggage handling
- Check-in service: Satisfaction level of Check-in service
- Inflight service: Satisfaction level of inflight service
- Cleanliness: Satisfaction level of Cleanliness
- Departure Delay in Minutes: Minutes delayed when departure
- Arrival Delay in Minutes: Minutes delayed when Arrival
- Satisfaction: Airline satisfaction level(Satisfaction, neutral or dissatisfaction)""")

# %%
#check null values

airline_df.isnull().sum()
# %%
# Imputing missing value with mean
airline_df['Arrival Delay in Minutes'] = airline_df['Arrival Delay in Minutes'].fillna(airline_df['Arrival Delay in Minutes'].mean())
airline_df.isnull().sum()
# %%
# Check the list of categorical variables
cat_variables = airline_df.select_dtypes(include=['object']).columns

# Count the number of NaN values in each categorical variable
nan_counts = airline_df[cat_variables].isnull().sum()

# Print the NaN counts for each categorical variable
print("NaN counts in categorical variables:")
print(nan_counts)

# %%
