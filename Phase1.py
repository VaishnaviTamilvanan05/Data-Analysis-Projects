#%%
# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns


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
#drop unnecessary columns
airline_df = airline_df.drop('Unnamed: 0', axis=1)
airline_df = airline_df.drop('id', axis=1)
airline_df.info()



# %%
# Distribution of data
# Plot histogram
airline_df.hist(bins=50, figsize=(20,15), color='lightseagreen', grid=True)
plt.tight_layout()
plt.show()



# %%

#Correlation matrix
# Select only numeric columns
numeric_df = airline_df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(20, 14)) 
sns.set(font_scale=1.2) 

# Create the heatmap with annotated values in each square
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)

# Display the plot
plt.title("Correlation Matrix")
plt.show()



# %%

# What is the general level of satisfaction among passengers within the dataset?


satisfaction_counts = airline_df['satisfaction'].value_counts()
labels = satisfaction_counts.index
sizes = satisfaction_counts.values

# Define colors for the pie slices
colors = ['lightcoral', 'paleturquoise']

# Create the pie chart
fig, ax = plt.subplots(figsize=(8, 6))
patches, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=120, pctdistance=0.85)

# Adjust the legend
legend_labels = [f"{label} ({size})" for label, size in zip(labels, sizes)]
ax.legend(patches, legend_labels, loc='upper left', bbox_to_anchor=(1, 1, 0.5, 0.5))

# Add a title
plt.title('Passenger Satisfaction Levels')

# # Adjust the plot layout to accommodate the legend
# plt.subplots_adjust(right=0.7)

plt.tight_layout()
# Display the plot
plt.show()



#%%

#mapping the satisfaction column for analysis

# Create a dictionary to map the values
satisfaction_map = {'satisfied': 1, 'neutral or dissatisfied': 0}

# Use map function to map values
airline_df['Satisfaction_Coded'] = airline_df['satisfaction'].map(satisfaction_map)

airline_df.head()







# %%

#What is the breakdown of passengers across different age groups, and how does age influence their level of satisfaction?


# Define age bins and labels
bins = [0, 13, 18, 26, 41, 66, 100]
labels = ['0-12', '13-17', '18-25', '26-40', '41-65', '66+']
labels_legend = ['0-12 Children', '13-17 Teenagers', '18-25 Young Adults', '26-40 Early-Middle-Aged',
                 '41-65 Late-Middle-Aged', '66+ Seniors']
airline_df['Age Group'] = pd.cut(airline_df['Age'], bins=bins, labels=labels, right=False)

# Calculate age distribution
age_data = airline_df['Age Group'].value_counts().sort_index()

# Calculate the satisfaction rate for each Age Group
airline_df_new = airline_df.groupby('Age Group')['Satisfaction_Coded'].mean().reset_index()

# Rename the columns
airline_df_new.columns = ['Age Group', 'Satisfaction Rate']



# Create a subplot with 2 rows and 1 column
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 20))

# Plot the pie chart for age distribution
# colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6']
colors = ['#1f77b4', '#aec7e8', '#6baed6', '#3182bd', '#08519c', '#084594']  
ax1.pie(age_data, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
ax1.set_title('Passenger Age Distribution')
ax1.legend(labels_legend, loc='upper left', bbox_to_anchor=(1, 1))


# Plot the bar plot for satisfaction rate by age group
# Define fading colors based on age
fade_colors = ['#1f77b4', '#4477aa', '#66aabb', '#88bbcc', '#aaddcc', '#ccffcc']

for i, (age_group, satisfaction_rate) in enumerate(zip(airline_df_new['Age Group'], airline_df_new['Satisfaction Rate'])):
    ax2.bar(age_group, satisfaction_rate, color=fade_colors[i], label=age_group)
    ax2.text(age_group, satisfaction_rate, f'{satisfaction_rate:.1%}', ha='center', va='bottom')

ax2.set_title('Satisfaction Rate by Age Group')
ax2.set_xlabel('Age Group')
ax2.set_ylabel('Satisfaction Rate (%)')

# Show plot
plt.tight_layout()
plt.show()



# %%


# %%
