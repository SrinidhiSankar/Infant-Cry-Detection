#webframework
import streamlit as st

#to display dataframe 
import pandas as pd

#display image
from PIL import Image

#allows descriptor calculation
import subprocess
import os
import base64

import numpy as np

#to load pickle file
import pickle
import Respond



# Model building
def model_predict(input_data):
    load_model = pickle.load(open('model.pkl', 'rb'))
    prediction = load_model.predict(input_data)
    
    st.header('**Prediction output**')
    
    prediction_output = pd.Series(prediction, name='pIC50')
    
    if identifier!="":
        molecule_name = pd.Series(iupac_load_data, name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
    elif identifier2!="":
        molecule_name = pd.Series(draw_data, name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
    else:
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        oral=[] 
        for i in load_data[0]:
            result=lipinski_pass(i)            
            if result==True:
                oral.append("Orally Bioavailable")
            else:
                oral.append("Not Orally Bioavailable")
        sm_col = pd.Series(oral, name='Oral Bioavailability')
        df = pd.concat([molecule_name, prediction_output,sm_col], axis=1)
    
    
    st.write(df)
    st.markdown(download_csv(df), unsafe_allow_html=True)
    
    df = pd.DataFrame(df)
    return df
    
from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave
import datetime
import sys
import test_model

# Importing external python files
from TTS import *
#from nlp import *
from Respond import *
from speech_recogniser import *
from test_model import main

THRESHOLD = 1500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100
BABY_REC = False
REC = True

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.
    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    st.write("Listening")
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False
    num_passed=0

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if not silent:
            num_passed+=1
            num_silent=0

        if snd_started and num_silent > 86: #2 seconds of complete silence
            st.write("2 seconds silence")
            break
        if num_passed>258: #6 seconds since suddden non-silent
            st.write("6 seconds over")
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    r = trim(r)
    r = normalize(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    global BABY_REC
    count={'bp':0,'bu':0,'ch':0,'dc':0,'hu':0,'lo':0,'sc':0,'ti':0}
    finalPrediction="";
    while finalPrediction=="":
        sample_width, data = record()
		
        
        		
        data = pack('<' + ('h'*len(data)), *data)
        wf = wave.open(path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        wf.writeframes(data)
        wf.close()
		
        #response = recognize_speech_from_file("recording.wav")
		
		# If speech recognizer is unable to decipher audio file, it would either mean that 1) The baby is crying, or 2) The person speaking sucks at speaking.
		# In our case we will assume the user has perfect recognizable speech
		# If speech is recognized, then we will assume it is the user that is issuing commands to the application
        
        #if(BABY_REC and response["error"]=="Unable to recognize speech"):
        if(1):
            prediction = test_model.main()  #predict here
            st.write("Guess: ",prediction)
            count[prediction]=count[prediction]+1
            if count[prediction]>1:
                finalPrediction=prediction
            else:
                st.write()
                #st.write(output)
    st.write("Final prediction: "+finalPrediction)
    st.write(count)
    st.write(respond(finalPrediction))
    return prediction

if __name__ == '__main__':
    image = Image.open('voice.jpg')
    st.image(image, use_column_width=True)

    st.markdown("""
    # Infant Cry Detection and Classification

    ***Get instant update on your baby***

    """)
    BABY_REC=False
    st.header("Welcome to Baby Ready")
    tts("Welcome to Baby Ready.")
    lastPrediction=datetime.datetime.now() - datetime.timedelta(minutes=5);
    while(1):
        
        prediction = record_to_file('recording.wav')  
        if(prediction == 'bp'):
            Respond.relieveBellyPain()
        elif(prediction == 'hu'):
            Respond.relieveHunger()
        elif(prediction == 'bu'):
            Respond.relieveBurping()
        elif(prediction == 'dc'):
            Respond.relieveDiscomfort()
        elif(prediction == 'ti'):
            Respond.relieveTired()
        elif(prediction == 'lo'):
            Respond.relieveLonely()
        elif(prediction == 'ch'):
            Respond.relieveTemp()
        elif(prediction == 'sc'):
            Respond.relieveScared()
        #send_email()
        

'''

with st.sidebar.header('Enter input data'):
    uploaded_file = st.sidebar.file_uploader("1. Upload your input file with smiles notation of molecules", type=['txt'])
    

with st.sidebar.header('2. Enter IUPAC name of molecule'):
    iupac_name = st.text_input("2. Enter IUPAC name of molecule")
    identifier  = iupac_name
    
with st.sidebar.header('3. Draw molecule by using below link'):
    st.write('3.Draw molecule by using below link')
    
with st.sidebar.header('3. Draw molecule by using below link'):
    url = 'http://localhost:5006/jsme'

    if st.button('Draw'):
         webbrowser.open_new_tab(url)
       
with st.sidebar.header('Paste the smile notation'):
    draw_data = st.text_input("Paste the smile notation")
    identifier2  = draw_data

flag=1
if st.sidebar.button('Predict'):

    if identifier!="":
        iupac_load_data = CIRconvert(identifier)
        if iupac_load_data=='Did not work':
            st.warning('Did not work. Please enter another name.')
            flag=0
        else:
            path_to_file = 'molecule.smi'
            path = Path(path_to_file)

            if path.is_file():    
                os.remove('molecule.smi')
            else:
                with open("molecule.smi","x") as file:
                    file.write(iupac_load_data + "\n")
            st.header('**Input data**')
            st.write(identifier, '  =>  ',iupac_load_data)
    
    elif identifier2!="":
        path_to_file = 'molecule.smi'
        path = Path(path_to_file)

        if path.is_file():    
            os.remove('molecule.smi')
        else:
            with open("molecule.smi","x") as file:
                file.write(draw_data + "\n")
        st.header('**Input data**')
        st.write(draw_data)
    
    else:    
        load_data = pd.read_table(uploaded_file, sep=' ', header=None)
        temp = load_data
        load_data.to_csv('molecule.smi', sep = '\t', header = False, index = False)

        st.header('**Input data**')
        st.write(load_data) 
 
    #calculating molecular descriptors
    if flag==1:
        with st.spinner("Calculating..."):
            descriptor_calculation()

        # Read in calculated descriptors and display the dataframe
        st.header('**Calculated molecular descriptors**')
        desc = pd.read_csv('descriptors_final_output.csv')
        
        if identifier!="":
            temp = lipinski_iupac(iupac_load_data)
        elif identifier2!="":
            temp = lipinski_iupac(draw_data)
        else:
            temp = lipinski(temp[0])
        
        desc_final = pd.concat([desc,temp], axis=1)
        
        st.write(desc_final)
        st.write(desc_final.shape)

        st.header('**Selected molecular descriptors**')
        Xlist = list(pd.read_csv('descriptor_final_list.csv').columns)
        desc_subset = desc_final[Xlist]
        
        st.write(desc_subset)
        st.write(desc_subset.shape)

        # Apply trained model to make prediction 
        with st.spinner("Model Predicting..."):        
            graph = model_predict(desc_subset)
        
        #Check for oral bioavailabiliy
        if identifier!="" or identifier2!="":
            if identifier!="":
                result=lipinski_pass(iupac_load_data)
            else:
                result=lipinski_pass(draw_data)
            if result==True:
                new_title = '<p style="color:Green; font-size: 42px;">*Orally bioavailable*</p>'
                st.markdown(new_title, unsafe_allow_html=True)
            else:
                new_title = '<p style="color:Red; font-size: 42px;">*Not orally bioavailable*</p>'
                st.markdown(new_title, unsafe_allow_html=True)            

        #plotting graph and finding most compatible drug
        if  identifier=="" and identifier2=="":
            st.header('**pIC50 Graph**')        
            with st.spinner("Plotting graph..."): 
                graph_output = px.bar(
                    graph, 
                    x='molecule_name',
                    y='pIC50',
                    color = "molecule_name")
                st.plotly_chart(graph_output)
            st.header('**Most compatible molecule to be used as drug**')
            column_p=graph["pIC50"]
            column_o=graph["Oral Bioavailability"]
            i=0
            index=0
            max_val=0
            while i<len(graph):
                if column_p[i]>max_val and column_o[i]=="Orally Bioavailable":
                    max_val = column_p[i]
                    index=i
                i=i+1
            comp_mol=graph["molecule_name"]            
            st.write(comp_mol[index],'  => ',max_val)        
    
else:
    st.info('Upload input data in the sidebar for prediction')'''
       
#2,5,5-trimethyl-2-hexene