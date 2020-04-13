# %%
import io
import requests
import pandas as pd
import numpy as np

def load_data(data = 'confirmed'):

    # url=f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-{data}.csv'
    url=f'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_{data}_global.csv'

    
    # get csv
    csv = requests.get(url).text
    df = pd.read_csv(io.StringIO(csv), index_col=['Country/Region', 'Province/State', 'Lat', 'Long'])

    # add new col type
    df['type'] = data.lower()

    # define rest cols
    df.columns.name = 'date'
    # 
    df = df.set_index('type', append=True)

    # drop lat lon col
    df = df.reset_index(['Lat', 'Long'], drop=True)
    
    # reshape row to col
    df = df.stack().reset_index()

    # index date
    df = df.set_index('date')

    # index to datime
    df.index = pd.to_datetime(df.index)

    #new names to cols
    df.columns = ['country', 'state', 'type', 'cases']

    return df 

# %%
def agg_states(df):
    ######################## states zusammenfassen ###############
    # split state
    df_country_with_states = df.loc[~df.state.isna()]
    df_country_without_states = df.loc[df.state.isna()]

    df_country_with_states_group = df_country_with_states.groupby(['country', 'date', 'type'])

    df_country_with_states = df_country_with_states_group.sum()

    df_country_with_states = df_country_with_states.rename(index=lambda x: x+' (total)', level=0)
    
    df_country_with_states = df_country_with_states.reset_index(level=['country', 'type'])


    df_con = pd.concat([df_country_without_states, df_country_with_states])

    df_result = df_con.drop(columns=['state']).reset_index()

    return df_result

# %%
data = 'confirmed'
df = load_data(data)
df = agg_states(df)
df.columns = ['date', 'state', 'type', 'cases']
df.to_csv('jhu.csv', index=False)
# %%
