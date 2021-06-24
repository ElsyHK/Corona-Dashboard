import pandas as pd
import numpy as np
import streamlit as st
#import plotly
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from plotly.subplots import make_subplots
from datetime import date
import matplotlib.pyplot as plt
#import pmdarima
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# hide 'made with streamlit'
st.markdown("""<style>footer{visibility:hidden;}</style>""",unsafe_allow_html=True)
# Define a function to load data for faster computation
@st.cache(allow_output_mutation=True)
def load_data(path):
    df = pd.read_excel(path, engine='openpyxl')
    return df

@st.cache(allow_output_mutation=True)
def load_data1(path):
    df = pd.read_csv(path, encoding='utf-8')
    return df

def place_value(number):
    return ("{:,}".format(number))


# upload image
url1 = 'https://drive.google.com/file/d/1Dfe6fLN7uBRbN9nvw6qte1AHxd-YK8ec/view?usp=sharing'
path1 = 'https://drive.google.com/uc?export=download&id='+url1.split('/')[-2]
add_image = st.sidebar.image(path1, use_column_width=True)

# dashboard made by Elsy Hobeika
st.sidebar.markdown(f'<p style="color:black; font-size: 15px; font-weight: bold; width: 300px;"> Dashboard by: Elsy Hobeika </p>', unsafe_allow_html=True)

#upload file or use the one loaded by default
upload_file = st.sidebar.file_uploader('UPLOAD DATA', type=['CSV','xlsx'])
url2 = 'https://drive.google.com/file/d/10jQggqcPIWoCYLK0cZC_NMFTFVTC1NiP/view?usp=sharing'
path2 = 'https://drive.google.com/uc?export=download&id='+url2.split('/')[-2]
if upload_file is None:
    covid_df = load_data(path2)
if upload_file is not None:
    covid_df = load_data(upload_file)


# clean data
num = covid_df._get_numeric_data()
num[num < 0] = 0

#Header of the dashboard
max_date = max(covid_df.date)
st.markdown(f'<p style="color:white; font-size: 25px; font-weight: bold;background-color:#09044D"> COVID-19 Dashboard - Updated {max_date} </p>', unsafe_allow_html=True)

#side bar menu
add_selectbox = st.sidebar.selectbox(
    'MENU',
    ('Overview', "Governments' Response/Results", 'Social Conditions Impact','Vaccinations & Predictions')
)

# Overview world wide
if add_selectbox == 'Overview':
    # Total cases deaths and vaccinations values for world and country level
    st.markdown(f'<p style="color:black; font-size: 25px; font-weight: bold; width: 300px;"> Worldwide </p>', unsafe_allow_html=True)

    #Metrics calculations
    total_cases = place_value(int(covid_df[covid_df.location == 'World'].total_cases.max()))
    total_deaths =  place_value(int(covid_df[covid_df.location == 'World'].total_deaths.max()))
    people_fully_vaccinated =place_value(int(covid_df[covid_df.location == 'World'].people_fully_vaccinated.max()))
    percentage_vaccinated = round(((covid_df[covid_df.location == 'World'].people_fully_vaccinated.max())/(covid_df.population.max()))*100, 2)

    #Metric display and design
    col1,col2,col3,col4 = st.beta_columns([1.3,1.2,1.5,1.75])

    with col1:
        st.markdown(f'\
        <p style="color:green; font-size:20px; text-align:left;line-height:20px;"> Total Cases \
        <p style="color:green; font-weight: bold; font-size: 30px;text-align:left;line-height:0px;"> {total_cases} \
        </p>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'\
        <p style="color:#FF9933; font-size:20px; text-align:left ;line-height:20px;"> Total Deaths \
        <p style="color:#FF9933; font-weight: bold; font-size: 30px;text-align:left;line-height:0px;"> {total_deaths} \
        </p>', unsafe_allow_html=True)

    with col3:
        st.markdown(f'\
        <p style="color:#3370FF; font-size:20px; text-align:left;line-height:20px;"> People Fully Vaccinated \
        <p style="color:#3370FF; font-weight: bold; font-size: 30px;text-align:left;line-height:0px"> {people_fully_vaccinated} \
        </p>', unsafe_allow_html=True)

    with col4:
        st.markdown(f'\
        <p style="color:#D72C2C; font-size:20px; text-align:left;line-height:20px;"> % of Population Fully Vaccinated \
        <p style="color:#D72C2C; font-weight: bold; font-size: 30px;text-align:left;line-height:0px"> {percentage_vaccinated}% \
        </p>', unsafe_allow_html=True)

    # Regional Metric calculations 
    st.write("")
    st.write("")

    st.markdown(f'<p style="color:black; font-size: 25px; font-weight: bold; line-height:30px"> Regional </p>', unsafe_allow_html=True)

    col5, col6, col7 = st.beta_columns([1,1,3])

    with col5:
        continent = st.selectbox(
        'Choose Continent',
        ["Asia","Europe","South America", "North America","Oceania","Africa"]
        )
    filtered_by_continent = covid_df[covid_df.continent == continent]

    with col6:
        country = st.selectbox(
        'Choose Country',
        filtered_by_continent.location.unique()
        )
    filtered_by_country = covid_df[covid_df.location == country]

    #Country Metric calcultions and check if data is available 
    total_cases = covid_df[covid_df.location == country].total_cases.max()
    total_deaths = covid_df[covid_df.location == country].total_deaths.max()

    if np.isnan(total_cases) == True:
        total_cases = "-"
        fig0 = ""
    elif np.isnan(total_cases) == False:
        total_cases = place_value(int(total_cases))
        fig0 = px.line(filtered_by_country, x='date', y="new_cases", title =f"<b>{country}</b> Daily New Corona Cases", width=600, height=400
        , labels={"date" :" ", "new_cases": " "}, hover_data={'date': True})


    if np.isnan(total_deaths) == True:
        total_deaths = '-'
    elif np.isnan(total_deaths) == False:
        total_deaths = place_value(int(total_deaths))

    people_fully_vaccinated =covid_df[covid_df.location == country].people_fully_vaccinated.max()
    percentage_vaccinated = round(((covid_df[covid_df.location == country].people_fully_vaccinated.max())/(filtered_by_country.population.max()))*100, 2)

    if np.isnan(people_fully_vaccinated) == True:
        people_fully_vaccinated = " - "
        percentage_vaccinated = " - "
    elif np.isnan(people_fully_vaccinated) == False:
        people_fully_vaccinated = place_value(int(people_fully_vaccinated))
        percentage_vaccinated = percentage_vaccinated

    # Country metric display
    col8,col9,col10,col11 = st.beta_columns([1.3,1.2,1.5,1.75])

    with col8:
        st.markdown(f'\
        <p style="color:#0F4913; font-size:20px; text-align:left;line-height:20px;"> Total Cases \
        <p style="color:#0F4913; font-weight:bold; font-size:30px ;text-align:left; line-height:0px;"> {total_cases} \
        </p>', unsafe_allow_html=True)

    with col9:
        st.markdown(f'\
        <p style="color:#C06307; font-size:20px; text-align:left ;line-height:20px;"> Total Deaths \
        <p style="color:#C06307; font-weight: bold; font-size: 30px;text-align:left;line-height:0px"> {total_deaths} \
        </p>', unsafe_allow_html=True)

    with col10:
        st.markdown(f'\
        <p style="color:#2341C2; font-size:20px; text-align:left;line-height:20px;"> People Fully Vaccinated \
        <p style="color:#2341C2; font-weight: bold; font-size: 30px;text-align:left;line-height:0px"> {people_fully_vaccinated} \
        </p>', unsafe_allow_html=True)

    with col11:
        st.markdown(f'\
        <p style="color:#771006; font-size:20px; text-align:left;line-height:20px;"> % of Population Fully Vaccinated \
        <p style="color:#771006; font-weight: bold; font-size: 30px;text-align:left;line-height:0px"> {percentage_vaccinated}% \
        </p>', unsafe_allow_html=True)

    st.write("")

    # contient chart for comparison among all countries
    x = filtered_by_continent.groupby(['date'],as_index=False)['new_cases'].sum()
    fig1 = px.line(x, x='date', y="new_cases", title = f"<b>{continent}</b> Daily New Corona Cases", width=600, height=400,
    labels={"date" :" ", "new_cases": "<b>New Cases"})

    col12, col13 = st.beta_columns([1.5,1.5])
    with col12:
        fig1

    with col13:
        fig0


###############################################################################
### Government Responses to corona
if add_selectbox == 'Governments\' Response/Results':
    st.markdown(f'<p style="color:black; font-size: 25px; font-weight: bold;line-height: 15px"> Governments\' Response to COVID-19 Pandemic  </p>', unsafe_allow_html=True)


    country = st.selectbox(
    'Choose Country',
    covid_df.location.unique()
    )
    filtered_by_country = covid_df[covid_df.location == country]

    col15, co,col16 = st.beta_columns([1.5,0.1,1.4])
    col17, co,col18 = st.beta_columns([1.5,0.1,1.4])
    
    # create 4 charts with stringency index 
    ## reproduction rate
    with col15:
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        fig.add_trace(
                go.Scatter(x=filtered_by_country.date, y=filtered_by_country.stringency_index, name="Stringency Index"),
                secondary_y=True,
            )
        fig.add_trace(
        go.Scatter(x=filtered_by_country.date, y=filtered_by_country.reproduction_rate, name="Reproduction Rate"),
                secondary_y=False,
            )

        fig.update_yaxes(title_text="<b>Stringency Index</b> ", secondary_y=True)
        fig.update_yaxes(title_text="<b>Reproduction Rate</b> ", secondary_y=False)
        fig.update_layout(legend=dict(yanchor="bottom",y=0.99,xanchor="left",x=0.01), width=600, height=400)
        fig
    
    ## New deaths per million
    with col18:
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        fig.add_trace(
                go.Scatter(x=filtered_by_country.date, y=filtered_by_country.stringency_index, name="Stringency Index"),
                secondary_y=True,
            )
        fig.add_trace(
        go.Scatter(x=filtered_by_country.date, y=filtered_by_country.new_deaths_per_million, name="New Deaths per Million", line=dict(color='maroon')),
                secondary_y=False,
            )

        fig.update_yaxes(title_text="<b>Stringency Index</b> ", secondary_y=True)
        fig.update_yaxes(title_text="<b>New Deaths Per Million</b> ", secondary_y=False)
        fig.update_layout(legend=dict(yanchor="bottom",y=0.99,xanchor="left",x=0.01), width=600, height=400)
        fig

    ## New cases per million
    with col16:
        # Create figure with secondary y-axis
        fig1= make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        fig1.add_trace(
                    go.Scatter(x=filtered_by_country.date, y=filtered_by_country.stringency_index, name="Stringency Index"),
                    secondary_y=True,
                )
        fig1.add_trace(
        go.Scatter(x=filtered_by_country.date, y=filtered_by_country.new_cases_per_million, name="New Cases per Million",line=dict(color='darkgreen')),
                    secondary_y=False,
                )

        fig1.update_yaxes(title_text="<b>Stringency Index</b> ", secondary_y=True)
        fig1.update_yaxes(title_text="<b>New Cases Per Million</b> ", secondary_y=False)
        fig1.update_layout(legend=dict(yanchor="bottom",y=0.99,xanchor="left",x=0.01), width=600, height=400)
        fig1

    ##Hospitalized patients per million
    with col17:
        # Create figure with secondary y-axis
        fig1= make_subplots(specs=[[{"secondary_y": True}]])
        # Add traces
        fig1.add_trace(
        go.Scatter(x=filtered_by_country.date, y=filtered_by_country.stringency_index, name="Stringency Index",),
        secondary_y=False,
                    )
        fig1.add_trace(
        go.Scatter(x=filtered_by_country.date, y=filtered_by_country.hosp_patients_per_million, name="Hospitalized Patients per Million",line=dict(color='darkorange')),
                    secondary_y=True,
                )

        fig1.update_yaxes(title_text="<b>Stringency Index</b> ", secondary_y=True)
        fig1.update_yaxes(title_text="<b>Hospitalized Patients per Million</b> ", secondary_y=False)
        fig1.update_layout(legend=dict(yanchor="bottom",y=0.99,xanchor="left",x=0.01), width=600, height=400)
        fig1




    st.markdown(f'<p style="color:black; font-size: 15px; font-weight: bold;line-height: 0px"> Stringency Index:</p>', unsafe_allow_html=True)
    st.write("""This is a composite measure based on nine response indicators including school closures, workplace
        closures, and travel bans, rescaled to a value from 0 to 100 (100 = strictest). If policies vary at the subnational
        level, the index is shown as the response level of the strictest sub-region.""")
    st.write("")

    # Create Comparision Map for Peaths per Million
    st.markdown(f'<p style="color:black; font-size: 25px; font-weight: bold;line-height: 15px"> Country Comparison of COVID-19 Total Deaths per Million  </p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:black;line-height: 5px"> Hover on Map Area for info - Change date with slider to check Deaths per Million due date </p>', unsafe_allow_html=True)

    covid_df['deaths_per_million'] = (covid_df['total_deaths']*1000000)/covid_df['population']
    fig = px.choropleth(covid_df, locations="iso_code", color="deaths_per_million", template='xgridoff',
    hover_name="location",  range_color=[0,2000], width=1000, basemap_visible=False, height=600, animation_frame='date')
    fig


###############################################################################
#Create scatter plots with socials variables
if add_selectbox == 'Social Conditions Impact':
    st.markdown(f'<p style="color:black; font-size: 25px; font-weight: bold;line-height: 15px"> Factors affecting COVID-19 Total Tests & Deaths </p>', unsafe_allow_html=True)
    
    # GDP and poverty impact on deaths and corona detection
    economic_metric = st.selectbox('Choose Metric',
                        ['gdp_per_capita','extreme_poverty']
                        )


    col1,col2 = st.beta_columns([1.5,1.5])
    with col1:

        y3 = covid_df[covid_df.date == '2021-05-01']
        y3 =y3[[ "total_tests_per_thousand", 'location','continent',economic_metric]]
        y3 = y3.dropna()
        figgdp = px.scatter(y3, y=economic_metric, x="total_tests_per_thousand",  color='continent',
        labels={"extreme_poverty" :"<b>Extreme Poverty","gdp_per_capita":'<b>GDP per Capita in $', "total_tests_per_thousand": "<b>Total Tests per Thousand "},
        hover_data ={'location':True}, width=600, height=500)
        figgdp.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.6))
        figgdp


    with col2:
        y4 = covid_df.groupby(['location','continent','population',economic_metric,'iso_code'],as_index=False)['total_deaths'].max()
        y4['deaths_per_million'] = (y4['total_deaths']*1000000)/y4['population']
        fig8 = px.scatter(y4, y=economic_metric, x="deaths_per_million", color='continent',
        labels={"extreme_poverty" :"","gdp_per_capita":'', "deaths_per_million": "<b>Total Deaths Per Million "},
        hover_data ={'location':True}, width=600, height=500)
        fig8.update_layout(legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.6))
        fig8


    ## Age impact on corona deaths relative to continents
    age = st.selectbox('Choose ',
                        ['median_age','aged_65_older','aged_70_older']
                        )
    y1 = covid_df.groupby(['location','continent','aged_65_older','population','median_age','aged_70_older','population_density'],as_index=False)['total_deaths'].max()
    y1['deaths_per_million'] = (y1['total_deaths']*1000000)/y1['population']
    fig2 = px.scatter(y1, y=age, x="deaths_per_million", width=500, height=400,color='continent',
    labels={age :f"<b>{age}</b> ", "deaths_per_million": "<b>Total Deaths per Million"})
    fig2


###############################################################################
#vaccinations and predictions:
if add_selectbox == 'Vaccinations & Predictions':
    st.markdown(f'<p style="color:black; font-size: 25px; font-weight: bold;line-height: 15px"> Vaccinations\' Status </p>', unsafe_allow_html=True)

    col1,col2,col3=st.beta_columns([1,1,1])
    with col1:
        continent = st.selectbox(
                'Choose Continent',
                ["Asia","Europe","South America", "North America","Oceania","Africa"]
                )

    filtered_by_continent = covid_df[covid_df.continent == continent]
    df_fully_vacc = filtered_by_continent.groupby(['location','continent'],as_index=False)['people_fully_vaccinated_per_hundred'].max()
    df_vacc = filtered_by_continent.groupby(['location'],as_index=False)['people_vaccinated_per_hundred'].max()
    df = pd.merge(df_fully_vacc, df_vacc, on='location')
    t1 = df.dropna()
    t=t1.sort_values(by='people_fully_vaccinated_per_hundred', ascending=False).head(10)


    with col2:
        country = st.selectbox(
            'Choose Country',
         t1.location.unique()
        )
    filtered_by_country = t1[t1.location == country]

    st.markdown(f'<p style="color:black;"> \
    In <b>{country}</b>, <b>{filtered_by_country.people_fully_vaccinated_per_hundred.iloc[0]}%</b> of the population is fully vaccinated and <b>{filtered_by_country.people_vaccinated_per_hundred.iloc[0]}%</b> of the population has recieved one dose.  \
    </p>', unsafe_allow_html=True)

    st.write("")
    st.markdown(f'<p style="color:black; font-size: 20px; font-weight: bold;"> Top 10 Countries in Vaccinations in <b>{continent}</b> </p>', unsafe_allow_html=True)

    fig = px.bar(t, y='location', x=['people_fully_vaccinated_per_hundred',"people_vaccinated_per_hundred"],
    width=1000, height=500).update_yaxes(categoryorder="total ascending")


    texts = [t.people_fully_vaccinated_per_hundred, t.people_vaccinated_per_hundred]
    for i, t in enumerate(texts):
        fig.data[i].text = t
        fig.data[i].textposition = 'inside'

    fig.update_layout(legend=dict(yanchor="bottom",y=-0.3,xanchor="left",x=-0.1),legend_title="")
    fig.update_yaxes(title='')
    fig.update_xaxes(title='')

    def customLegend(fig, nameSwap):
        for i, dat in enumerate(fig.data):
            for elem in dat:
                if elem == 'name':
                    fig.data[i].name = nameSwap[fig.data[i].name]
        return(fig)

    fig = customLegend(fig=fig, nameSwap = {'people_fully_vaccinated_per_hundred': '% of people fully vaccinated against COVID-19', 'people_vaccinated_per_hundred':'% of people partially vaccinated against COVID-19'})
    fig


    ###### prediction - prepare dataframe
    st.markdown(f'<p style="color:black; font-size: 20px; font-weight: bold;"> Predict Number of Vaccinations in the Future </p>', unsafe_allow_html=True)
    col1,col2,col3=st.beta_columns([1,1,1])
    
    with col1:
        country1 = st.selectbox(
            'Select Country',
            ["Lebanon","Bahrain",'France', 'Canada']
            )
        
    df = covid_df[['date','new_vaccinations','location','population']]
    df.date = pd.to_datetime(df.date, format='%Y/%m/%d')
    df['Date'] = df.date.dt.date
    filtered = df[df.location == country1]
    population = filtered.population.max()
    z1=filtered.dropna()
    z1 = z1.drop(['location','date','population'],axis=1)
    z=z1.set_index('Date')
    
   
    # split into train and test sets
    X = z.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = []

    ## walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(0,1,4))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
       # st.write('predicted=%f, expected=%f' % (yhat, obs))


    ## evaluate forecasts
    rmse = np.sqrt(mean_squared_error(test, predictions))
    #st.write('Test RMSE: %.3f' % rmse)

 
    ## predict 
    with col2:
        nb_days= st.text_input("Enter number of days",30)


    # enter a number
    nb_days = int(nb_days)-1
    pred_future=model_fit.predict(len(X), len(X)+nb_days, typ='levels')
        
       
    ## plot 
    #f, ax = plt.subplots(1,1,figsize=(10,4))
    #ax.plot(test)
    #ax.plot(predictions, color='red')
    #st.pyplot(f)
    
        
    # create time index  
    start = str(z1.Date.iloc[-1])
    date = datetime.strptime(start, "%Y-%m-%d")
    end = date + timedelta(days=nb_days)
    index_future_dates=pd.Series(pd.date_range(start=start, end=end))
        
        
    #covert the predictions to a dataframe with date index
    df = pd.DataFrame(data =pred_future, columns=['new_vaccinations'])
    df['date']=index_future_dates
    df.set_index('date', drop=True, inplace=True)
    #df
    df1=df[1:-1]
        
    # plot actual and predicted values on teh same plot
    f, ax = plt.subplots(1,1,figsize=(10,4))
    ax.plot(z)
    ax.plot(df, color='red')
    ax.axvline(x=df.index[0], linewidth = 2, color='black', linestyle='--')
    st.pyplot(f) 
        
    
        
    # population
    new_vaccinations=df1.new_vaccinations.sum()
    new_vaccinations=place_value(int(new_vaccinations))
    # perc_new_vaccinations = (new_vaccinations/population)*100
    # perc_new_vaccinations
    # filtered_by_country.people_fully_vaccinated_per_hundred.iloc[0]
    # filtered_by_country.people_vaccinated_per_hundred.iloc[0]
        
    st.markdown(f'<p style="color:black;"> \
        In <b>{country1}</b>, <b>{new_vaccinations}</b> new vaccinations will be made after <b>{nb_days+1}</b> days.  \
        </p>', unsafe_allow_html=True)

