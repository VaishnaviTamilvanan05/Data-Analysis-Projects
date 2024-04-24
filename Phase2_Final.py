import dash
import dash.html as html
from dash import dcc
import pandas as pd
from dash import html, dcc
from dash.dependencies import Input, Output
from dash import html, dcc, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import dash_table
from scipy import signal
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import normaltest
import dash_bootstrap_components as dbc
import numpy as np










#read dataset transformed
df = pd.read_csv('/Users/vaishnavitamilvanan/Documents/Spring 2024/Visualization/Project/Visualization-of-Complex-Data-DATS-6401/Airline_passenger_data_cleaned.csv')
# Read dataset orginal
airline_df = pd.read_csv('/Users/vaishnavitamilvanan/Documents/Spring 2024/Visualization/Project/Visualization-of-Complex-Data-DATS-6401/Airline_Passenger_Satisfaction.csv')






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
                      showlegend=False, margin=dict(t=50, l=25, r=25, b=25))
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
        marker_color=colors  # Apply the conditional colors here
    )])
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis={'title': 'Number of Passengers'},
        yaxis={'title': 'Class'},
        margin=dict(t=20, b=20),
        height=300,
        width=700
    )
    return fig


#############################

# Initialize the app with external CSS for Bootstrap
app = dash.Dash(__name__, external_stylesheets=['https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap'], suppress_callback_exceptions=True)
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
        href='https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction',  # Replace this URL with the actual URL of your dataset
          
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
        categorical = [col for col in airline_df.columns if airline_df[col].dtype == 'object']
        numerical = [col for col in airline_df.columns if airline_df[col].dtype in ['int64', 'float64']]
        
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
                html.Br(),
                html.Div(id='test-result')
            ],style=div_style)


@app.callback(
    Output('test-result', 'children'),
    [Input('column-dropdown', 'value'),
     Input('test-dropdown', 'value')]
)

def perform_test(selected_column, selected_test):
    if selected_column and selected_test:
        data = airline_df[selected_column]
        if selected_test == 'shapiro':
            return shapiro_test(data, selected_column)
        elif selected_test == 'ks':
            return ks_test(data, selected_column)
        elif selected_test == 'dagostino':
            return da_k_squared_test(data, selected_column)
    return "Please select a column and a test."

#############################################################################


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

    row_children = []  # This will hold all the column components for the row

    # Calculate the number of total services selected
    total_services = len(pre_flight_selected) + len(in_flight_selected)
    if total_services == 0:
        return "Please select at least one service to display."
    column_width = max(2, 12 // total_services)  # Calculate width, minimum size 2 to ensure visibility

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

    return dbc.Row(row_children)  # Return a row with all columns (charts)



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
    html.H1("Airline Passenger Satisfaction Dashboard", style={
        'textAlign': 'center',
        'marginTop': '20px',
        'fontFamily': 'Roboto, sans-serif',
        'fontWeight': '500'
    }),
    html.Br(),
    dcc.Tabs(id='finalproject', children=[
        dcc.Tab(label='Dataset Information', value='tab-info'),
        dcc.Tab(label='Customer Experience Overview', value='tab-1'),
        dcc.Tab(label='Normality Tests', value='tab-2'),
        dcc.Tab(label='Operational Insights', value='tab-3')
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
    else:
        return html.Div()  # Default return if no tab matches

# Run server
if __name__ == '__main__':
    app.run_server(debug=True, port=8080)