
import dash
import dash.html as html
from dash import dcc
import pandas as pd
from dash import html, dcc
from dash.dependencies import Input, Output
from dash import html, dcc, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from scipy import signal
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import normaltest
import dash_bootstrap_components as dbc
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff
from sklearn.decomposition import PCA
import scipy.stats as stats



import warnings

# Suppress specific FutureWarning
warnings.filterwarnings(
    action='ignore',
    message="The default of observed=False is deprecated",
    category=FutureWarning
)


#read dataset transformed
url1 ='https://raw.githubusercontent.com/VaishnaviTamilvanan/Visualization-of-Complex-Data-DATS-6401/main/Airline_passenger_data_cleaned.csv'
df = pd.read_csv(url1)
# Read dataset orginal
url2= 'https://raw.githubusercontent.com/VaishnaviTamilvanan/Visualization-of-Complex-Data-DATS-6401/main/Airline_Passenger_Satisfaction.csv'
airline_df = pd.read_csv(url2)




# Counting satisfied and not satisfied responses
satisfaction_counts = airline_df['satisfaction'].value_counts().to_dict()
total_passengers = airline_df.shape[0]
# Dataset Information (number of entries and columns)
dataset_info = f"The Airline Passenger Satisfaction dataset contains information about the satisfaction levels of over {total_passengers} airline passengers. The dataset includes details about each passenger's flight, such as class, customer type, travel type, as well as their assessments of various factors like food, service, cleanliness, and seat comfort. This comprehensive dataset allows for in-depth analysis of the key drivers of passenger satisfaction and can help airlines identify areas for improvement to enhance the overall travel experience."



################################################

#Data Cleaning

#################################################

#check null values
airline_df.isnull().sum()

######## Handle Missing Values ########

# Imputing missing value with mean
airline_df['Arrival Delay in Minutes'] = airline_df['Arrival Delay in Minutes'].fillna(airline_df['Arrival Delay in Minutes'].mean())
airline_df.isnull().sum()

# Check the list of categorical variables
cat_variables = airline_df.select_dtypes(include=['object']).columns
# Count the number of NaN values in each categorical variable
nan_counts = airline_df[cat_variables].isnull().sum()
#######drop unnecessary columns #########
airline_df = airline_df.drop('Unnamed: 0', axis=1)
airline_df = airline_df.drop('id', axis=1)






# Counting satisfied and not satisfied responses
satisfaction_counts = df['satisfaction'].value_counts().to_dict()
total_passengers = df.shape[0]

#########################################################################



################################################
#NORMALITY TESTS

def shapiro_test(x, title):
    stats, p = shapiro(x)
    results = f"Shapiro test: {title} dataset: statistics = {stats:.2f}, p-value = {p:.2f}"
    alpha = 0.01
    if p > alpha:
        results += "\nShapiro test: Normal distribution."
    else:
        results += "\nShapiro test: NOT Normal distribution."
    return results

def ks_test(x, title):
    mean = np.mean(x)
    std = np.std(x)
    dist = np.random.normal(mean, std, len(x))
    stats, p = kstest(x, dist)
    results = f"K-S test: {title} dataset: statistics = {stats:.2f}, p-value = {p:.2f}"
    alpha = 0.01
    if p > alpha:
        results += "\nK-S test: Normal distribution."
    else:
        results += "\nK-S test: NOT Normal distribution."
    return results

def da_k_squared_test(x, title):
    stats, p = normaltest(x)
    results = f"Da'gostino's K-squared test: {title} dataset: statistics = {stats:.2f}, p-value = {p:.2f}"
    alpha = 0.01
    if p > alpha:
        results += "\nDa'gostino's K-squared test: Normal distribution."
    else:
        results += "\nDa'gostino's K-squared test: NOT Normal distribution."
    return results


############################################

#Operational serivices 

# Define pre-flight and in-flight services
pre_flight_services = [
    'Departure/Arrival time convenient', 
    'Ease of Online booking', 
    'Gate location', 
    'Checkin service'
]

in_flight_categories = {
    'Entertainment Services': ['Inflight entertainment'],
    'Comfort Services': ['Seat comfort', 'Leg room service'],
    'Hospitality Services': ['Food and drink', 'On-board service', 'Cleanliness'],
    'Connectivity and Handling Services': ['Inflight wifi service', 'Online boarding', 'Baggage handling', 'Inflight service']
}

operational_insights_tab_info = """
This dashboard utilizes checkboxes to enable the selection of specific pre-flight and in-flight services, allowing for tailored analysis crucial to enhancing passenger experiences. The in-flight services are grouped into four main categories:

1. **Entertainment Services:** Focuses on 'Inflight entertainment' to enhance enjoyment through various media options.
2. **Comfort Services:** Includes 'Seat comfort' and 'Leg room service', key to physical comfort and reducing travel fatigue.
3. **Hospitality Services:** Covers 'Food and drink', 'On-board service', and 'Cleanliness', impacting the overall perception of the airline's quality.
4. **Connectivity and Handling Services:** Encompasses services like 'Inflight wifi', 'Online boarding', and 'Baggage handling', which streamline the travel process and are especially valued by business and tech-savvy passengers.

By leveraging insights from passenger feedback on these services, this dashboard helps airlines identify critical improvement areas, driving enhancements where most needed. This focused approach not only helps pinpoint service delivery weaknesses but also facilitates strategic upgrades, potentially boosting passenger satisfaction and loyalty significantly.
"""






############################################
#Charts

def create_gender_distribution_chart(df,selected_gender=None):
    if selected_gender:
        df = df[df['Gender'] == selected_gender]
    gender_counts = df['Gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']
    fig = go.Figure(data=[go.Pie(labels=gender_counts['Gender'], values=gender_counts['Count'],
                                 hole=0.4, textinfo='label+percent', marker=dict(colors=px.colors.sequential.Blues_r))])
    fig.update_traces(textposition='inside')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      showlegend=True, margin=dict(t=50, l=25, r=30, b=25),
                      legend=dict(x=1, y=0.5, xanchor='left', yanchor='middle'), width=350,  # Set the width of the figure
        height=450)
    return fig


def create_class_distribution_chart(df, selected_class=None,selected_gender=None):
    if selected_gender:
        df = df[df['Gender'] == selected_gender]
    class_counts = df['Class'].value_counts().reset_index()
    class_counts.columns = ['Class', 'Count']
    
    # Color logic: Highlight the selected class
    colors = ['goldenrod' if cls == selected_class else 'lightgrey' for cls in class_counts['Class']]
    
    fig = go.Figure(data=[go.Bar(
        x=class_counts['Count'],
        y=class_counts['Class'],
        orientation='h',
        text=class_counts['Count'],
        textposition='inside',
        marker_color=colors,  

    )])
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis={'title': 'Number of Passengers'},
        yaxis={'title': 'Class'},
        margin=dict(t=20, b=20),
        height=300,
        width=700,
    )
    return fig


#############################

# Initialize the app with external CSS for Bootstrap
app = dash.Dash(__name__, external_stylesheets=['https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap'], suppress_callback_exceptions=True)
server = app.server

#background cloud img
background_image_url = '/assets/background.png'  

# Define a CSS dictionary for text and elements
text_style = {'color': '#333', 'fontFamily': 'Roboto, sans-serif'}
tab_style = {'backgroundColor': '#e9ecef'}
selected_tab_style = {'backgroundColor': '#007bff', 'color': 'white'}
div_style = {
    'margin': '20px', 
    'padding': '20px', 
    'backgroundColor': 'rgba(0,0,0,0)',
    'borderRadius': '5px',
    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
    'minHeight': '700px'  # Ensure all main divs have at least 500px height
}

#############################
#Tab-info layout

url3= 'https://raw.githubusercontent.com/VaishnaviTamilvanan/Visualization-of-Complex-Data-DATS-6401/main/Airline_Passenger_Satisfaction.csv'
airline_df_tabinfo = pd.read_csv(url3)
# Define layouts for tabs
tab_info_layout = html.Div([
    html.H3('Dataset Overview', style=text_style),
    html.P(dataset_info, style=text_style),
    html.H3('Features Information', style=text_style),
    html.H4('There are 25 columns in total. Select an option to display feature information:', style=text_style),
    dcc.RadioItems(
        id='feature-selection',
        options=[
            {'label': 'Display Feature Description', 'value': 'desc'},
            {'label': 'Display Datatype Info', 'value': 'info'},
            {'label': 'None', 'value': 'none'}
        ],
        value='none',  # Default to none selected
        labelStyle={'display': 'block'}
    ),
    html.Div(id='feature-info-display'),
    html.H4('This dataset is available for download on Kaggle:'),
    html.A(
        html.Button('Click here to check Dataset', id='download-button'),
        href='https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction',  
          
    )
],style=div_style)

# Define callback for showing feature descriptions or summary statistics
@app.callback(
    Output('feature-info-display', 'children'),
    [Input('feature-selection', 'value')]
)
def update_display(selected_option):
    if selected_option == 'desc':
        return html.Div([
            html.H4('Description of Features in the data:'),
            html.Ul([
                html.Li('Gender: Gender of the passengers (Female, Male)'),
                html.Li('Customer Type: The customer type (Loyal customer, disloyal customer)'),
                html.Li('Age: The actual age of the passengers'),
        html.Li('Type of Travel: Purpose of the flight of the passengers (Personal Travel, Business Travel)'),
        html.Li('Class: Travel class in the plane of the passengers (Business, Eco, Eco Plus)'),
        html.Li('Flight distance: The flight distance of this journey'),
        html.Li('Inflight wifi service: Satisfaction level of the inflight wifi service (0:Not Applicable; 1-5)'),
        html.Li('Departure/Arrival time convenient: Satisfaction level of Departure/Arrival time convenient'),
        html.Li('Ease of Online booking: Satisfaction level of online booking'),
        html.Li('Gate location: Satisfaction level of Gate location'),
        html.Li('Food and drink: Satisfaction level of Food and drink'),
        html.Li('Online boarding: Satisfaction level of online boarding'),
        html.Li('Seat comfort: Satisfaction level of Seat comfort'),
        html.Li('Inflight entertainment: Satisfaction level of inflight entertainment'),
        html.Li('On-board service: Satisfaction level of On-board service'),
        html.Li('Leg room service: Satisfaction level of Leg room service'),
        html.Li('Baggage handling: Satisfaction level of baggage handling'),
        html.Li('Check-in service: Satisfaction level of Check-in service'),
        html.Li('Inflight service: Satisfaction level of inflight service'),
        html.Li('Cleanliness: Satisfaction level of Cleanliness'),
        html.Li('Departure Delay in Minutes: Minutes delayed at departure'),
        html.Li('Arrival Delay in Minutes: Minutes delayed at arrival'),
        html.Li('Satisfaction: Airline satisfaction level (Satisfaction, neutral, or dissatisfaction)')
            ])
        ])
    elif selected_option == 'info':
        # Check the data types of each column
        categorical = [col for col in airline_df_tabinfo.columns if airline_df_tabinfo[col].dtype == 'object']
        numerical = [col for col in airline_df_tabinfo.columns if airline_df_tabinfo[col].dtype in ['int64', 'float64']]
        
        # Create HTML components for each type of data
        numerical_features = html.Ul([html.Li(f'{feature}: Numerical') for feature in numerical])
        categorical_features = html.Ul([html.Li(f'{feature}: Categorical') for feature in categorical])

        return html.Div([
            html.H4('Data Types of Features:'),
            html.Div([
                html.Div([
                    html.H5('Numerical Features:'),
                    numerical_features
                ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
                html.Div([
                    html.H5('Categorical Features:'),
                    categorical_features
                ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'})
            ])
        ])
    
    return''
    

###################################################################################
#tab1 layout
tab_customer_experience_layout = html.Div([
    html.H3('Customer Experience Overview', style=text_style),
    html.Div(id='tab-content-1', children=[
        html.H2('Overall Satisfaction Breakdown'), 
        dcc.RadioItems(
            id='satisfaction-radio',
            options=[
                {'label': 'Satisfied', 'value': 'satisfied'},
                {'label': 'Neutral or dissatisfied', 'value': 'neutral or dissatisfied'}
            ],
            value='satisfied',
            style={'width': '100%', 'padding': '10px'},
            labelStyle={'display': 'block', 'margin': '10px', 'fontSize': '16px'}
        ),
        dcc.Dropdown(
            id='customer-type-dropdown',
            options=[
                {'label': 'Loyal Customers', 'value': 'loyal'},
                {'label': 'Disloyal Customers', 'value': 'disloyal'}
            ],
            value='loyal',
            clearable=False,
            style={'width': '400px', 'padding': '5px'}  # Set a fixed width for the dropdown
        ),
        dcc.Store(id='gender-selection-store', storage_type='session'),
        dcc.Graph(id='customer-satisfaction-graph', style={'alignSelf': 'flex-end', 'width': '50%'}),
        html.Div(style={'position': 'absolute', 'top': '120px', 'right': '270px', 'width': '300px'},
                 children=[dcc.Graph(id='gender-distribution-chart', figure=create_gender_distribution_chart(df))]),
        html.Div(style={'position': 'absolute', 'bottom': '10px', 'right': '100px', 'width': '650px'},
                 children=[dcc.Graph(id='class-distribution-chart', figure=create_class_distribution_chart(df))])
    ])
])
#Call-back for tab1

@app.callback(
    [Output('customer-satisfaction-graph', 'figure'),
     Output('satisfaction-radio', 'options'),
     Output('gender-distribution-chart', 'figure'),
     Output('class-distribution-chart', 'figure')],
    [Input('gender-distribution-chart', 'clickData'),
     Input('satisfaction-radio', 'value'),
     Input('customer-type-dropdown', 'value'),
     Input('class-distribution-chart', 'clickData')],
    [State('gender-distribution-chart', 'figure'),
     State('class-distribution-chart', 'figure'),
     State('gender-selection-store', 'data')]
)





def update_graph_based_on_inputs(clickData_gender, satisfaction_value, customer_type, clickData_class, donut_figure, bar_figure, stored_data):
    
    
    # Initialize selected variables
    selected_gender = None
    selected_class = None
    # Filter based on customer type
    filter_value = 'Loyal Customer' if customer_type == 'loyal' else 'disloyal Customer'
    filtered_df = df[df['Customer Type'] == filter_value]

    # Update based on gender selection
    if clickData_gender:
        selected_gender = clickData_gender['points'][0]['label'] if clickData_gender else None
        filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
        colors = ['goldenrod' if label == selected_gender else 'royalblue' for label in donut_figure['data'][0]['labels']]
        donut_figure['data'][0]['marker']['colors'] = colors

    current_selection = clickData_gender['points'][0]['label'] if clickData_gender else None
    if stored_data and stored_data == current_selection:
        # Reset the selection
        new_data = None
        gender_fig = create_gender_distribution_chart(df)  # Reset to show all data
    else:
        # Set new selection
        new_data = current_selection
        gender_fig = create_gender_distribution_chart(df, selected_gender=new_data)


      

    # Update based on class selection from the bar chart
    if clickData_class:
        selected_class = clickData_class['points'][0]['y'] if clickData_class else None

        filtered_df = filtered_df[filtered_df['Class'] == selected_class]
        # colors = ['goldenrod' if y == selected_class else 'lightslategrey' for y in bar_figure['data'][0]['y']]
        # bar_figure['data'][0]['marker']['color'] = colors


    # Determine the selected class from the bar chart click data
    selected_class = clickData_class['points'][0]['y'] if clickData_class else None
    # Update class distribution chart with potential selected class
    class_fig = create_class_distribution_chart(df, selected_class, selected_gender)




    # Update the satisfaction counts
    local_satisfaction_counts = filtered_df['satisfaction'].value_counts().to_dict()
    new_options = [
        {'label': f'Satisfied ({local_satisfaction_counts.get("satisfied", 0)}) {"✓" if satisfaction_value == "satisfied" else ""}', 'value': 'satisfied'},
        {'label': f'Neutral or dissatisfied ({local_satisfaction_counts.get("neutral or dissatisfied", 0)}) {"✓" if satisfaction_value == "neutral or dissatisfied" else ""}', 'value': 'neutral or dissatisfied'}
    ]

    # Prepare data for plotting satisfaction graph
    plot_data = filtered_df['satisfaction'].value_counts().reset_index()
    plot_data.columns = ['satisfaction', 'count']
    title = f"Satisfaction Ratings for {filter_value} Customers"
    if selected_gender and selected_class:
        title += f" - {selected_gender} & {selected_class} "
    elif selected_gender:
        title += f" - {selected_gender} only"
    elif selected_class:
        title += f" - {selected_class} only"

    fig = px.bar(plot_data, x='satisfaction', y='count', text='count', title=title)
    colors = ['gray' if x != satisfaction_value else 'cornflowerblue' for x in plot_data['satisfaction']]
    fig.update_traces(marker_color=colors, textposition='outside')
    fig.update_layout(
        xaxis_title='Satisfaction',
        yaxis_title='Number of Responses',
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent paper background
        showlegend=False
    )

    return fig, new_options, donut_figure, class_fig

###################################################################


#Tab3 -NTests


tab_normality_tests_layout = html.Div([
    html.H3('Select variable to perform Normality Tests'),
    html.Br(),
    dcc.Dropdown(
        id='column-dropdown',
        options=[{'label': col, 'value': col} for col in airline_df.select_dtypes(include=[np.number]).columns],
        value=None,
        placeholder="Select a column"
    ),
    html.Br(),
    html.H3('Select a normality test to perform on the selected variable'),
    html.Br(),
    dcc.Dropdown(
        id='test-dropdown',
        options=[
            {'label': 'Shapiro-Wilk Test', 'value': 'shapiro'},
            {'label': 'Kolmogorov-Smirnov Test', 'value': 'ks'},
            {'label': "D'Agostino's K-squared Test", 'value': 'dagostino'}
        ],
        value=None,
        placeholder="Select a normality test"
    ),
    html.Br(),
    dcc.Graph(id='normal-distribution-graph'),
    html.Br(),
    html.Div(id='test-result')
], style=div_style)




@app.callback(
    [Output('test-result', 'children'),
     Output('normal-distribution-graph', 'figure')],
    [Input('column-dropdown', 'value'),
     Input('test-dropdown', 'value')]
)
def update_output(selected_column, selected_test):
    if not selected_column:
        return "Please select a column to proceed.", {}
    
    data = airline_df[selected_column].dropna()
    fig = plot_distribution(data, selected_column)
    
    if not selected_test:
        return "Please select a test to proceed.", fig

    if selected_test == 'shapiro':
        result = shapiro_test(data, selected_column)
    elif selected_test == 'ks':
        result = ks_test(data, selected_column)
    elif selected_test == 'dagostino':
        result = da_k_squared_test(data, selected_column)
    else:
        return "Test not implemented.", fig

    return result, fig

def plot_distribution(data, column_name):
    fig = px.histogram(data, x=column_name, marginal='box', nbins=50)
    mean = np.mean(data)
    std_dev = np.std(data)
    x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
    y = stats.norm.pdf(x, mean, std_dev)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Normal Fit'))
    return fig


#############################################################################

##########################################################################
#TAb4 -operational services insights

tab_operational_insights_layout = html.Div([
    html.Div([
        dcc.Markdown(operational_insights_tab_info, style=text_style),
        html.Br(),
        html.H3("Select Pre-Flight Services:"),
        html.Br(),
        
        dcc.Checklist(
            options=[{'label': service, 'value': service} for service in pre_flight_services],
            value=[],
            id='pre-flight-checkboxes',
            inline=True
        )
    ], className="mb-3"),
    html.Br(),
    
    html.Div([
        html.H3("Select In-Flight Service Categories:"),
        html.Br(),
        
        dcc.Checklist(
            options=[{'label': key, 'value': key} for key in in_flight_categories.keys()],
            value=[],
            id='in-flight-checkboxes',
            inline=True
        )
    ], className="mb-3"),
    html.Br(),
    dbc.Button("Submit", id='submit-button', color="primary", className="mb-3"),
    html.Br(),
    html.Div(id='charts-container')
], style=div_style)



@app.callback(
    Output('charts-container', 'children'),
    Input('submit-button', 'n_clicks'),
    [State('pre-flight-checkboxes', 'value'),
     State('in-flight-checkboxes', 'value')]
)





def update_output(n_clicks, pre_flight_selected, in_flight_selected):
    if n_clicks is None:
        return "Please select services and click submit."

    row_children = []  

    # Calculate the number of total services selected
    total_services = len(pre_flight_selected) + len(in_flight_selected)
    if total_services == 0:
        return "Please select at least one service to display."
    column_width = max(2, 12 // total_services)  

    # Handling pre-flight service selections
    if pre_flight_selected:
        pre_flight_data = airline_df[pre_flight_selected]
        avg_pre_flight = pre_flight_data.mean().reset_index()
        avg_pre_flight.columns = ['Service', 'Average Rating']

        for index, row in avg_pre_flight.iterrows():
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=row['Average Rating'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"{row['Service']} Satisfaction"},
                gauge={
                    'axis': {'range': [None, 5], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, row['Average Rating']], 'color': 'lightblue'},
                        {'range': [row['Average Rating'], 5], 'color': 'lightgray'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': row['Average Rating']}
                }
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title_text=f"{row['Service']} Satisfaction (Out of 5)",
                showlegend=False
            )
            col = dbc.Col(dcc.Graph(figure=fig), width=column_width)
            row_children.append(col)

    # Handling in-flight service selections similarly
    if in_flight_selected:
        in_flight_data = pd.concat([airline_df[categories] for key in in_flight_selected for categories in in_flight_categories[key]], axis=1)
        avg_in_flight = in_flight_data.mean().reset_index()
        avg_in_flight.columns = ['Service', 'Average Rating']

        for index, row in avg_in_flight.iterrows():
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=row['Average Rating'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"{row['Service']} Satisfaction"},
                gauge={
                    'axis': {'range': [None, 5], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, row['Average Rating']], 'color': 'lightblue'},
                        {'range': [row['Average Rating'], 5], 'color': 'lightgray'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': row['Average Rating']}
                }
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                title_text=f"{row['Service']} Satisfaction (Out of 5)",
                showlegend=False
            )
            col = dbc.Col(dcc.Graph(figure=fig), width=column_width)
            row_children.append(col)

    return dbc.Row(row_children)  
###############################################################################


#range and heatmap



# Bin the delays into categories
bins = [-1, 0, 19, float('inf')]
labels = ['No delay', '1 to 19', '≥ 20 mins delay']

# Apply the binning
df['Departure Delay Category'] = pd.cut(df['Departure Delay in Minutes'], bins=bins, labels=labels)
df['Arrival Delay Category'] = pd.cut(df['Arrival Delay in Minutes'], bins=bins, labels=labels)

# Create a new DataFrame for satisfied customers
df_satisfied = df[df['satisfaction'] == 'satisfied']

# Group by the new categories and count, passing observed=True to avoid the FutureWarning
grouped = df_satisfied.groupby(['Arrival Delay Category', 'Departure Delay Category'], observed=True).size().reset_index(name='Count')

# Pivot using keyword arguments
pivot_table = grouped.pivot(index='Arrival Delay Category', columns='Departure Delay Category', values='Count')


# Create the heatmap
fig = ff.create_annotated_heatmap(z=pivot_table.to_numpy(), 
                                  x=pivot_table.columns.tolist(), 
                                  y=pivot_table.index.tolist(), 
                                  annotation_text=pivot_table.round(2).astype(str).values, 
                                  colorscale='oranges')

# Update the layout to match your specifications
fig.update_layout(
    title='Satisfaction Count for Departure and Arrival Delays',
    xaxis_title='Departure Delay',
    yaxis_title='Arrival Delay',
    xaxis={'side': 'top'},
    yaxis=dict(autorange='reversed'),  # To match the traditional matrix layout
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
    plot_bgcolor='rgba(0,0,0,0)'  # Transparent background
)










# Reducing the number of unique marks for clarity and performance
unique_distances = np.sort(df['Flight Distance'].unique())
marks = {str(distance): str(distance) for distance in np.linspace(unique_distances.min(), unique_distances.max(), num=10, dtype=int)}

#Flight Metrics Explorer layout

tab_flight_metrics_layout= html.Div([
    html.H4('Select flight distance range:'),
    html.Br(),
    dcc.RangeSlider(
        id='distance-slider',
        min=df['Flight Distance'].min(),
        max=df['Flight Distance'].max(),
        value=[df['Flight Distance'].min(), df['Flight Distance'].max()],
        marks=marks,
        step=100
    ),
    html.Br(),
    html.Br(),
    html.Div(id='output-container-range-slider'),
    html.Br(),
    dcc.Loading(
        id="loading-1",
        type="default",
        children=html.Div(id='loading-output-1')
    ),
    html.Br(),
    dcc.Graph(id='satisfaction-graph')
],style=div_style)

tab_flight_metrics_layout.children.append(html.Div([dcc.Graph(figure=fig)]))
# Callbacks to update graphs

@app.callback(
    [
        Output('satisfaction-graph', 'figure'),
        Output('output-container-range-slider', 'children'),
        Output('loading-output-1', 'children')
    ],
    [Input('distance-slider', 'value')]
)
def update_graph(selected_range):
    filtered_df = df[(df['Flight Distance'] >= selected_range[0]) & (df['Flight Distance'] <= selected_range[1])]
    
    if filtered_df.empty:
        no_data_fig = {
            'data': [],
            'layout': {
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'paper_bgcolor': 'rgba(0,0,0,0)',  # Set to transparent
                'plot_bgcolor': 'rgba(0,0,0,0)',  # Set to transparent
                'annotations': [{'text': 'No data available for the selected range', 'x': 0.5, 'y': 0.5, 'showarrow': False, 'font': {'size': 16, 'color': 'white'}}]
            }
        }
        return [no_data_fig, f'No data for range: {selected_range}', 'No data to display']

    # Calculate counts of satisfaction levels
    satisfaction_counts = filtered_df.groupby('satisfaction').size().reset_index(name='count')

    # Satisfaction Graph (Updated to Bar Chart)
    fig_satisfaction = px.bar(satisfaction_counts, x='satisfaction', y='count', title="Satisfaction Levels Count",
                              text='count', color='satisfaction',  # Assign color based on 'satisfaction'
                              category_orders={"satisfaction": sorted(satisfaction_counts['satisfaction'].unique())})  # Sort satisfaction categories

    fig_satisfaction.update_layout(
        xaxis_title="Satisfaction",
        yaxis_title="Count",
        paper_bgcolor='rgba(0,0,0,0)',  # Set to transparent
        plot_bgcolor='rgba(0,0,0,0)',  # Set to transparent
        font_color="steelblue",
        legend_title_text='Satisfaction Level'  # Add title to legend
    )
    fig_satisfaction.update_traces(texttemplate='%{text}', textposition='outside')

    return [fig_satisfaction, f'Showing results for range: {selected_range}', 'Data loaded']

######################################################################


#Demographic insights

#mapping the satisfaction column for analysis
# Create a dictionary to map the values
satisfaction_map = {'satisfied': 1, 'neutral or dissatisfied': 0}
# Use map function to map values
airline_df['Satisfaction_Coded'] = airline_df['satisfaction'].map(satisfaction_map)

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

#Demographic insights


# Prepare your initial pie chart and bar graph figures
fig_pie_initial = px.pie(airline_df, names='Age Group', title='Passenger Age Distribution')
fig_pie_initial.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',  # Set to transparent
        plot_bgcolor='rgba(0,0,0,0)' ,   # Set to transparent
        legend=dict(x=1, y=0.5, xanchor='auto', yanchor='auto')
    )
fig_bar_initial = px.bar(airline_df_new, x='Age Group', y='Satisfaction Rate', title='Satisfaction Rate by Age Group',
                         )

tab_demographic_insights_layout = html.Div([
    html.H1('Passenger Age and Satisfaction Analysis'),
    dcc.Graph(id='age-distribution-pie', figure=fig_pie_initial),
    dcc.Graph(id='satisfaction-rate-bar', figure=fig_bar_initial),
])

@app.callback(
    Output('satisfaction-rate-bar', 'figure'),
    [Input('age-distribution-pie', 'clickData')],
    # [State('satisfaction-rate-bar', 'figure')]
)
def highlight_selected_age_group(pie_click_data):
    
    fig_bar = px.bar(
        airline_df_new,
        x='Age Group',
        y='Satisfaction Rate',
        title='Satisfaction Rate by Age Group',
        text_auto=True  # Display count of each bar on top
    )
    
    # Set default colors for all bars
    colors = ['lightskyblue'] * len(airline_df_new['Age Group'])
    
    # If a part of the pie chart was clicked, update the bar color of the selected age group
    if pie_click_data:
        # Get the age group that was clicked
        age_group_clicked = pie_click_data['points'][0]['label']
        
        # Find the index of the clicked age group and change its color to green
        if age_group_clicked in airline_df_new['Age Group'].values:
            clicked_index = airline_df_new['Age Group'].tolist().index(age_group_clicked)
            colors[clicked_index] = 'green'  # Set the selected age group bar to green

    # Update the bar chart with the new colors list
    fig_bar.update_traces(marker_color=colors)

    # Set the background to transparent
    fig_bar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',  # Set to transparent
        plot_bgcolor='rgba(0,0,0,0)'    # Set to transparent
    )

    return fig_bar

#####################################################################

Summary_insights= """
- The data reveals that pre-flight services like 'Ease of Online Booking' and 'Gate Location' have an average satisfaction rating below 3, while 'Inflight WiFi Service' also scores poorly. Airlines could use this insight to focus on improving these areas, enhancing the overall passenger experience, boosting satisfaction, and potentially increasing loyalty and competitive advantage.

- Passenger satisfaction increases with the length of the flight, with short-haul flights showing the highest dissatisfaction rates. Airlines can enhance satisfaction on these flights by improving seat comfort and speeding up the boarding process, making short journeys more comfortable and efficient for travelers.

- The flights with no departure or arrival delays have the highest satisfaction count, while significant dissatisfaction arises when either departure or arrival delays exceed 20 minutes. It indicates that on-time performance is a critical factor in passenger satisfaction. To improve satisfaction, airlines should prioritize punctuality and manage delays effectively.

- The data indicates that passenger satisfaction is considerably higher in the 41-65 age group, potentially due to the higher likelihood of business class travel within this demographic. On the other hand, the lower satisfaction rates observed in the youngest and oldest age brackets suggest a need for airlines to focus on age-specific amenities and services to improve their travel experience.
"""

tab_feedback_summary_layout=html.Div([
    html.H1('Summary of Insights'),
    dcc.Markdown(Summary_insights, style=text_style),

    html.H2('About the Author'),
    html.P('This web application was developed by Vaishnavi Tamilvanan for the Visualization of Complex Data DATS 6401 Final Term Project. Your suggestions and feedback are highly valued; please do not hesitate to reach out via email at Vaishnavi.tamilvanan@gwmail.gwu.edu.'),

    html.H2('Feedback'),
    html.P('Please provide your suggestions for improvements or additional features you would like to see:'),
    dcc.Textarea(
        id='feedback-textarea',
        style={'width': '100%', 'height': 200},
        placeholder='Enter your feedback here...'
    ),

    html.Button('Submit Feedback', id='submit-feedback', n_clicks=0),
    html.Div(id='feedback-response')
], style=div_style)

@app.callback(
    Output('feedback-response', 'children'),
    Input('submit-feedback', 'n_clicks'),
    [State('feedback-textarea', 'value')]
)
def update_output(n_clicks, value):
    if n_clicks > 0:
        return 'Thank you for your feedback!'
    return ''
###################################################################

#PCA



# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df.select_dtypes(include=[np.number]))



# # Perform PCA
pca = PCA()
pca_data = pca.fit_transform(features_scaled)



# Layout
PCA_tab_layout = html.Div([
    html.Div([
        dcc.RadioItems(
            id='pca-space-selector',
            options=[
                {'label': 'Original Space', 'value': 'original'},
                {'label': 'Transformed Space', 'value': 'transformed'}
            ],
            value='original'
        ),
        html.Br(),  # Break line for spacing
        html.Div(id='pca-details'),
        html.Br(),  # Break line for spacing
        dcc.Graph(id='pca-explained-variance'),
        html.Br(),  # Break line for spacing
        dcc.Graph(id='pca-correlation-matrix')
    ], style=div_style)
])

@app.callback(
    Output('pca-details', 'children'),
    Output('pca-explained-variance', 'figure'),
    Output('pca-correlation-matrix', 'figure'),
    Input('pca-space-selector', 'value')
)
def update_pca_analysis(space):
    if space == 'original':
        singular_values = pca.singular_values_
        details = (
            f"Features: {df.columns.tolist()}, Original shape: {df.shape}, "
            f"Singular values: {singular_values.round(2)}, "
            f"Condition number: {np.round(np.max(singular_values) / np.min(singular_values), 2)}"
        )
        correlation_matrix_fig = px.imshow(
            np.corrcoef(features_scaled, rowvar=False),
            title='PCA Features Correlation Matrix',
            labels={'color': 'Correlation'}
        )
    else:
        transformed_features = pca.transform(features_scaled)
        transformed_singular_values = pca.singular_values_ * np.sqrt(pca.explained_variance_ratio_)
        details = (
            f"Transformed Features Shape: {transformed_features.shape}, "
            f"Singular values (transformed space): {transformed_singular_values.round(2)}, "
            f"Condition number (transformed space): {np.round(np.max(transformed_singular_values) / np.min(transformed_singular_values), 2)}"

        )
        correlation_matrix_fig = px.imshow(
            np.corrcoef(transformed_features, rowvar=False),
            title='PCA Features Correlation Matrix',
            labels={'color': 'Correlation'}
        )
    
    explained_variance_fig = px.line(
        y=np.cumsum(pca.explained_variance_ratio_),
        x=[f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        title='Cumulative Explained Variance',
        labels={'y': 'Cumulative Explained Variance', 'x': 'Principal Component'}
    )

    return details, explained_variance_fig, correlation_matrix_fig






######################################################################
# Home App layout
app.layout = html.Div(style={
    'backgroundImage': f'url("{background_image_url}")',
    'backgroundSize': 'cover',
    'backgroundPosition': 'center',
    'backgroundRepeat': 'no-repeat',
    'backgroundAttachment': 'fixed',  # Ensures the image does not scroll with the content
    'minHeight': '100vh',  # Minimum height to cover the full viewport height
    'height': 'auto',  # Adjusts height based on content
    'display': 'flex',
    'flexDirection': 'column',
    'justifyContent': 'space-between'
}, children=[
            html.Div(  # This div will act as a flex container for the image and heading
    style={
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center',  # Adjust this to 'flex-start' to align items to the left
        'marginTop': '10px',
    },
    children=[
        html.Img(src='/assets/flight_image.png', style={
            'height': '60px',  # Set the image height to be small
            'marginRight': '20px',  # Add some space between the image and the heading
        }),
        html.H1("Airline Passenger Satisfaction Dashboard", style={
            'fontFamily': 'Roboto, sans-serif',
            'fontWeight': '500',
        }),
    ]
),

    html.Br(),
    dcc.Tabs(id='finalproject', value='tab-info', children=[
        dcc.Tab(label='Dataset Information', value='tab-info'),
        dcc.Tab(label='Customer Experience Overview', value='tab-1'),
        dcc.Tab(label='Normality Tests', value='tab-2'),
        dcc.Tab(label='Operational Insights', value='tab-3'),
        dcc.Tab(label='Flight Metrics Explorer', value='tab-4'),
        dcc.Tab(label='Demographic Insights', value='tab-5'),
        dcc.Tab(label='PCA Analysis', value='tab-7'),
        dcc.Tab(label='Feedback & Summary', value='tab-6')
        
    ]),
    html.Div(id='layout')
])

# Callback to switch tabs
@app.callback(
    Output('layout', 'children'),
    [Input('finalproject', 'value')]
)
def update_layout(tab_name):
    if tab_name == 'tab-info':
        return tab_info_layout
    elif tab_name == 'tab-1':
        return tab_customer_experience_layout
    elif tab_name == 'tab-2':
        return tab_normality_tests_layout
    elif tab_name == 'tab-3':
        return tab_operational_insights_layout
    elif tab_name == 'tab-4':
        return tab_flight_metrics_layout
    elif tab_name == 'tab-5':
        return tab_demographic_insights_layout
    elif tab_name == 'tab-6':
        return tab_feedback_summary_layout
    elif tab_name == 'tab-7':
        return PCA_tab_layout
    else:
        return html.Div()




####################################################################################################
#run server

# if __name__ == '__main__':
#      app.run_server(debug=True, host='0.0.0.0', port=8085)

# Run server
if __name__ == '__main__':
    app.run_server(debug=True, port=8085)
