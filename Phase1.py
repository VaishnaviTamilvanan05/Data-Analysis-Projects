#%%
# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import numpy as np
from tabulate import tabulate

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

airline_df['Age Group'] = pd.cut(airline_df['Age'], bins=bins, labels=labels, right=False)
airline_df.head()

#%%
#class type influence on satisfaction 

# Count the number of passengers in each age group and class
age_class_count = airline_df.groupby(['Age Group', 'Class']).size().unstack(fill_value=0)

# Define colors for each travel class
colors = ['gold', 'dimgrey', 'saddlebrown']

# Plot stacked bar plot
sns.set_style("whitegrid")
ax3 = age_class_count.plot(kind='bar', stacked=True, figsize=(10, 6), color=colors)
ax3.set_title('Distribution of Travel Class by Age Group', fontsize=16)
ax3.set_xlabel('Age Group', fontsize=14)
ax3.set_ylabel('Number of Passengers', fontsize=14)
ax3.legend(title='Travel Class', fontsize=12)

# Add gridlines
ax3.grid(axis='y', linestyle='--', alpha=0.7)


# Adjust layout
plt.tight_layout()

# Show plot
plt.show()



# %%


# Define gender categories and labels
gender_labels = ['Female','Male']
gender_labels_legend = ['Male', 'Female']

# Calculate gender distribution
gender_data = airline_df['Gender'].value_counts().sort_index()

# Calculate the satisfaction rate for each Gender Group
gender_df_new = airline_df.groupby('Gender')['Satisfaction_Coded'].mean().reset_index()

# Rename the columns
gender_df_new.columns = ['Gender', 'Satisfaction Rate']

# Create a subplot with 2 rows and 1 column
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Plot the pie chart for gender distribution
gender_colors = ['deepskyblue', 'orchid']
ax1.pie(gender_data, labels=gender_labels, autopct='%1.1f%%', startangle=90, colors=gender_colors)
ax1.set_title('Passenger Gender Distribution')
ax1.legend(gender_labels_legend, loc='upper left', bbox_to_anchor=(1, 1))


for i, (gender_group, satisfaction_rate) in enumerate(zip(gender_df_new['Gender'], gender_df_new['Satisfaction Rate'])):
    ax2.bar(i, satisfaction_rate, color=gender_colors[i], label=gender_group)
    ax2.text(i, satisfaction_rate, f'{satisfaction_rate:.1%}', ha='center', va='bottom')

ax2.set_title('Satisfaction Rate by Gender Group')
ax2.set_xlabel('Gender Group')
ax2.set_ylabel('Satisfaction Rate (%)')

# Set x-axis tick labels
ax2.set_xticks(range(len(gender_labels)))
ax2.set_xticklabels(gender_labels)

# Show plot
plt.tight_layout()
plt.show()


# %%

# Separating numeric and categorical columns
numeric_cols = airline_df.select_dtypes(include=['int64', 'float64'])
categorical_cols = airline_df.select_dtypes(include=['object'])

# Calculate numeric statistics
numeric_stats = numeric_cols.describe()
numeric_stats.loc['median'] = numeric_cols.median()
numeric_stats.loc['mode'] = numeric_cols.mode().iloc[0]

# Calculate categorical statistics (mode and count)
categorical_stats = categorical_cols.describe(include=['object'])
categorical_stats.loc['mode'] = categorical_cols.mode().iloc[0]

# Print the statistics in a table format using tabulate
print("# Numeric Statistics of Dataset")
print(tabulate(numeric_stats, headers='keys', tablefmt='psql', floatfmt=".2f"))

print("\n# Categorical Statistics of Dataset")
print(tabulate(categorical_stats, headers='keys', tablefmt='psql'))

#%%
#Distribution of each gender by satisfaction according to the mean of Flight Distance


print("Mean Flight Distance by Gender and Satisfaction\n")
airline_df.pivot_table(index='satisfaction',columns='Gender',values='Flight Distance',aggfunc='mean').style.background_gradient(cmap='BuGn').format("{:.2f}")


# %%
#Dist subplot
# Set up the matplotlib figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

# Plotting 'Arrival Delay in Minutes' on the first subplot
sns.histplot(airline_df['Arrival Delay in Minutes'], bins=30, kde=False, ax=axes[0], label='Arrival Delay', color='blue')
axes[0].set_title('Distribution of Arrival Delays')
axes[0].set_xlabel('Delay in Minutes')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Plotting 'Departure Delay in Minutes' on the second subplot
sns.histplot(airline_df['Departure Delay in Minutes'], bins=30, kde=False, ax=axes[1], label='Departure Delay', color='green')
axes[1].set_title('Distribution of Departure Delays')
axes[1].set_xlabel('Delay in Minutes')
axes[1].set_ylabel('Frequency')
axes[1].legend()

# Improve layout and display the plot
plt.tight_layout()
plt.show()

# %%
