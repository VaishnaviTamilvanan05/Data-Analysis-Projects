import dash
import dash.html as html
from dash import dcc
import pandas as pd
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

#read dataset 
df = pd.read_csv('/Users/vaishnavitamilvanan/Documents/Spring 2024/Visualization/Project/Visualization-of-Complex-Data-DATS-6401/Airline_passenger_data_cleaned.csv')
# Counting satisfied and not satisfied responses
satisfaction_counts = df['satisfaction'].value_counts().to_dict()
total_passengers = df.shape[0]

def create_gender_distribution_chart(df):
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








# Initialize the app with external CSS for Bootstrap
app = dash.Dash(__name__, external_stylesheets=['https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap'])


#background cloud img
background_image_url = '/assets/background.png'  

# Define a CSS dictionary for text and elements
text_style = {'color': '#333', 'fontFamily': 'Roboto, sans-serif'}
tab_style = {'backgroundColor': '#e9ecef'}
selected_tab_style = {'backgroundColor': '#007bff', 'color': 'white'}
div_style = {
    'margin': '20px', 
    'padding': '20px', 
    'backgroundColor': '#f8f9fa',  # Light grey background for sections
    'borderRadius': '5px',  # Rounded corners for divs
    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'  # Subtle shadow for depth
}



#####Layout#####

app.layout = html.Div(style={
    'backgroundImage': f'url("{background_image_url}")',
    'backgroundSize': 'cover',
    'backgroundPosition': 'center',
    'backgroundRepeat': 'no-repeat',
    'height': '100vh',
    'display': 'flex',
    'flexDirection': 'column',
    'justifyContent': 'space-between',  # Align items along the main axis
}, children=[
    html.H1("Airline Passenger Satisfaction Dashboard", style={
        'textAlign': 'center',
        'marginTop': '20px',
        'fontFamily': 'Roboto, sans-serif',
        'fontWeight': '500'
    }),
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Customer Experience Overview', value='tab-1', children=[
            html.Div(id='tab-content-1',children=[
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
                dcc.Graph(id='customer-satisfaction-graph', style={'alignSelf': 'flex-end', 'width': '50%'}),

                html.Div(style={'position': 'absolute', 'top': '120px', 'right': '270px', 'width': '300px'},
         children=[dcc.Graph(id='gender-distribution-chart', figure=create_gender_distribution_chart(df))]),
         html.Div(style={'position': 'absolute', 'bottom': '10px', 'right': '100px', 'width': '650px'},
         children =[dcc.Graph(id='class-distribution-chart', figure=create_class_distribution_chart(df)) ])

            ])
        ]),
        dcc.Tab(label='Operational Insights', value='tab-2', children=[
            html.Div(id='tab-content-2', children=[
                html.H3('Content for Tab 2: Operational Insights')
            ])
        ])
    ])
])



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
     State('class-distribution-chart', 'figure')]
)








def update_graph_based_on_inputs(clickData_gender, satisfaction_value, customer_type, clickData_class, donut_figure, bar_figure):
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







# ... Run server ...

if __name__ == '__main__':
    app.run_server(debug=True)

