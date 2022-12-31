import pandas as pd
import numpy as np
import streamlit as st
import tensorflow as tf
import sklearn
import itertools
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from utils import Remover


# config

st.set_page_config(
    page_title="MxMH",
    page_icon='https://i.imgur.com/VOJb5at.png'
)


# styling

st.markdown(
     f"""
     <style>
       
     @media only screen and (max-width: 769px) {{
        #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div > div:nth-child(1) > div > div:nth-child(3) > div:nth-child(3) > div:nth-child(1) > div > div:nth-child(1),
        #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div > div:nth-child(1) > div > div:nth-child(3) > div:nth-child(2) > div:nth-child(1) > div > div:nth-child(1),
        #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div > div:nth-child(1) > div > div:nth-child(5) > div:nth-child(2) > div:nth-child(1) > div > div:nth-child(1),
        #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div > div:nth-child(1) > div > div:nth-child(32) > div:nth-child(2) > div:nth-child(1) > div > div:nth-child(2) > div{{
            display: none;
        }}
     }}
     
    .stApp {{
            background: url("https://i.imgur.com/7DeK74z.png");
            background-size: cover
        }}
        
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div > div:nth-child(1) > div{{
        background-color: rgba(255,255,255,0.6);
        border-radius: 25px;
        padding: 20px;
    }}
     
     #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div > div:nth-child(1) > div > div:nth-child(32) > div:nth-child(1) > div:nth-child(1) > div > div:nth-child(1) > div,
     #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div > div:nth-child(1) > div > div:nth-child(32) > div:nth-child(2) > div:nth-child(1) > div > div:nth-child(1){{
         text-align: center
     }}
     
     #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div > div:nth-child(1) > div > div:nth-child(31){{
         text-align: center
     }}
     
     [data-testid="stHeader"] {{
         background-color: #DF523A
    }}
         
    [data-testid="stTickBarMin"] {{
         display: none;
    }}
         
    [data-testid="stTickBarMax"] {{
        display: none;
    }} 
    
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div > div:nth-child(1) > div > div:nth-child(32) > div:nth-child(1),
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div > div:nth-child(1) > div > div:nth-child(32) > div:nth-child(2){{
        background-color: #F3E0C9;
        padding: 20px
        
    }}
    
    .stSelectbox, #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div > div:nth-child(1) > div > div > div{{max-width: calc(100% - 40px);}}
    
        
     </style>
     """,
     unsafe_allow_html=True
 )

# loading

data = pd.read_csv('mxmh_survey_results.csv')

with open("pipeline_1.pkl", 'rb') as file:
    full_pipeline = pickle.load(file)
    
model = tf.keras.models.load_model('model.h5')

# condense num classes from 11 to 6

value_map = {10: 5, 9: 5, 8: 4, 7: 4, 6: 3, 5: 3, 4: 2, 3: 2, 2: 1, 1: 1}

def condense_rankings(value):
  value = round(value)
  return value_map.get(value, 0)

data[['Anxiety', 'Depression', 'Insomnia', 'OCD']] = data[['Anxiety', 'Depression', 'Insomnia', 'OCD']].applymap(condense_rankings)


# demographics questions

st.write('### Demographics')

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age:", value = 20, min_value=1, max_value=90)

with col2:
    st.write("#")
    musician = st.checkbox("I play an instrument.")

with col3:
    st.write("#")
    composer = st.checkbox("I compose music.")

streaming_service = st.selectbox('Primary streaming service:', ['Spotify', 'YouTube Music', 'Apple Music', 'Pandora', 'Other service', 'I do not use a streaming service'])

col4, col5 = st.columns(2)

with col4:
    hours = st.number_input("Hours of listening time per day:", value=0, min_value=0, max_value=24)
    
with col5:
    st.write("#")
    working = st.checkbox("I listen to music while working.")

music_effects = st.selectbox('How do you think music improves your mental health?', ['Improve', 'Worsen', 'No effect'], index=2)
    
# music questions

st.write("### Music Taste")
   
exploratory = st.checkbox("I actively explore new artists/genres.")
foreign = st.checkbox("I regularly listen to music with lyrics in a language I am not fluent in.")

st.write("**How frequently do you listen to each genre?**")

# def hide_labels(value):
#    return ''

classical = st.select_slider(
    'Classical',
    options=['Never', 'Rarely', 'Sometimes', 'Very frequently'], value = 'Sometimes')

country = st.select_slider(
    'Country',
    options=['Never', 'Rarely', 'Sometimes', 'Very frequently'], value = 'Sometimes')

edm = st.select_slider(
    'EDM',
    options=['Never', 'Rarely', 'Sometimes', 'Very frequently'], value = 'Sometimes')

folk = st.select_slider(
    'Folk',
    options=['Never', 'Rarely', 'Sometimes', 'Very frequently'], value = 'Sometimes')

gospel = st.select_slider(
    'Gospel',
    options=['Never', 'Rarely', 'Sometimes', 'Very frequently'], value = 'Sometimes')

hiphop = st.select_slider(
    'Hip Hop',
    options=['Never', 'Rarely', 'Sometimes', 'Very frequently'], value = 'Sometimes')

jazz = st.select_slider(
    'Jazz',
    options=['Never', 'Rarely', 'Sometimes', 'Very frequently'], value = 'Sometimes')

kpop = st.select_slider(
    'K pop',
    options=['Never', 'Rarely', 'Sometimes', 'Very frequently'], value = 'Sometimes')

latin = st.select_slider(
    'Latin',
    options=['Never', 'Rarely', 'Sometimes', 'Very frequently'], value = 'Sometimes')

lofi = st.select_slider(
    'Lofi',
    options=['Never', 'Rarely', 'Sometimes', 'Very frequently'], value = 'Sometimes')

metal = st.select_slider(
    'Metal',
    options=['Never', 'Rarely', 'Sometimes', 'Very frequently'], value = 'Sometimes')

pop = st.select_slider(
    'Pop',
    options=['Never', 'Rarely', 'Sometimes', 'Very frequently'], value = 'Sometimes')

randb = st.select_slider(
    'R&B',
    options=['Never', 'Rarely', 'Sometimes', 'Very frequently'], value = 'Sometimes')

rap = st.select_slider(
    'Rap',
    options=['Never', 'Rarely', 'Sometimes', 'Very frequently'], value = 'Sometimes')

rock = st.select_slider(
    'Rock',
    options=['Never', 'Rarely', 'Sometimes', 'Very frequently'], value = 'Sometimes')

video = st.select_slider(
    'Video game music',
    options=['Never', 'Rarely', 'Sometimes', 'Very frequently'], value = 'Sometimes')

col6, col7 = st.columns(2)

with col6:
    fav_genre = st.selectbox('Favorite genre:', ['Classical', 'Country', 'EDM', 'Folk', 'Gospel', 'Hip hop', 'Jazz', 'K pop', 'Latin', 'Lofi', 'Metal', 'Pop', 'R&B', 'Rap', 'Rock', 'Video game music'])
    
with col7:
    bpm = st.number_input("BPM of favorite song:", value = 100, min_value=40, max_value=250)
    

# submit

submit = st.button("Predict mental health rankings")

def get_medians(fav_genre, grouping):        
    df_fav= data[data[grouping] == fav_genre]
    df_grouped = df_fav.groupby(grouping)
    df_grouped = df_grouped.median()
    df_grouped_list = df_grouped.iloc[:, -4:].values.round().tolist()

    return list(itertools.chain(*df_grouped_list))

if submit:
    input = pd.DataFrame(np.array([['Timestamp', age, streaming_service, hours, working, musician, composer, fav_genre, exploratory, foreign, bpm, classical, country, edm, folk, gospel, hiphop, jazz, kpop, latin, lofi, metal, pop, randb, rap, rock, video, music_effects, 'permissions']]),
                         columns = ['Timestamp', 'Age', 'Primary streaming service', 'Hours per day', 'While working', 'Instrumentalist', 'Composer', 'Fav genre', 'Exploratory', 'Foreign languages', 'BPM', 'Frequency [Classical]', 'Frequency [Country]', 'Frequency [EDM]', 'Frequency [Folk]', 'Frequency [Gospel]', 'Frequency [Hip hop]', 'Frequency [Jazz]', 'Frequency [K pop]', 'Frequency [Latin]', 'Frequency [Lofi]', 'Frequency [Metal]', 'Frequency [Pop]', 'Frequency [R&B]', 'Frequency [Rap]', 'Frequency [Rock]', 'Frequency [Video game music]', 'Music effects', 'Permissions'])
    
    input.replace(['True', 'False'],
                        ['Yes', 'No'], inplace=True)
    
    input = full_pipeline.transform(input)
    
    predictions = model.predict(input)
    
    anxiety_r = str(np.argmax(predictions[0], axis=1))
    depression_r = str(np.argmax(predictions[1], axis=1))
    insomnia_r = str(np.argmax(predictions[2], axis=1))
    ocd_r = str(np.argmax(predictions[3], axis=1))
      
    st.write("## Predictions")
    st.write("Mental health predictions are quantified on a scale of 0-5, where 0 signifies no experience with a condition and 5 signifies regular or extreme experiences with a condition.")
    
    col8, col9, col10, col11 = st.columns(4)

    with col8:    
        st.write("**Anxiety**: ", anxiety_r)
        
    with col9:
        st.write("**Depression**: ", depression_r)
        
    with col10:
        st.write("**Insomnia**: ", insomnia_r)
    
    with col11:
        st.write("**OCD**: ", ocd_r)
        
    strings = [anxiety_r, depression_r, insomnia_r, ocd_r]

    conv_values = np.array([])

    for string in strings:
        conv_value = int(string.strip("[]"))
        conv_values = np.append(conv_values, conv_value)
    
    col12, col13 = st.columns(2)

    with col12:    
        
        st.write("Median comparison for individuals selecting ", fav_genre, " as their favorite genre")
        
        fig, ax = plt.subplots()
        ax.scatter([], [])
        ax.set_xlim(-0.5, 5.5)
        fig.patch.set_facecolor('#F3E0C9')
        ax.set_facecolor('#F3E0C9')

        data2 = get_medians(fav_genre, 'Fav genre')
        markers = conv_values
        
        plt.barh(range(len(data2)), data2)

        for i, v in enumerate(conv_values):
            plt.plot([v, v], [i - 0.25, i + 0.25], 'k-')
            
        ax.set_yticks(range(len(data2)))
        ax.set_yticklabels(['Anxiety', 'Depression', 'Insomnia', 'OCD'], rotation=45)
        
        plt.xlabel("Rankings")

        st.pyplot(fig)
        
        
    with col13:
        
        st.write("Median comparison for the ", str(age), "yrs age group")
        
        fig, ax = plt.subplots()
        ax.scatter([], [])
        ax.set_xlim(-0.5, 5.5)
        fig.patch.set_facecolor('#F3E0C9')
        ax.set_facecolor('#F3E0C9')

        data2 = get_medians(age, 'Age')
        
        plt.barh(range(len(data2)), data2)

        for i, v in enumerate(conv_values):
            plt.plot([v, v], [i - 0.25, i + 0.25], 'k-')
            
        ax.set_yticks(range(len(data2)))
        ax.set_yticklabels(['Anxiety', 'Depression', 'Insomnia', 'OCD'], rotation=45)
        
        plt.xlabel("Rankings")

        st.pyplot(fig)
        
    viz_desc = st.expander('How to interpret plots')
    with viz_desc:
        st.write('Both plots mark *your* predicted mental health rankings with black lines. The orange bars represent the median mental health rankings of other people who selected the same answers for "Fav genre" and "Age." These data points were collected via a 2022 survey with over 700 instances.')
        st.write('To learn more about this data, visit the EDA (exploratory data analysis) notebook [here](https://www.kaggle.com/code/catherinerasgaitis/music-mental-health-eda).')