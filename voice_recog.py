import pickle
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import librosa


def audio_recognition(path_of_audio):
    data = pd.read_csv("D:\projectSTP\training data.csv")

    # we need to let Male = 1 and female = 0 for further data engineering
    data['label'] = data['label'].replace('male', 1)
    data['label'] = data['label'].replace('female', 0)

    X = data[['sd', 'Q25', 'IQR', 'mode', 'meanfun']].values
    Y = data['label'].values

    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = LinearSVC()
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    y_preds = model.predict(x_test)
    print(f"Training score = {metrics.accuracy_score(y_train, train_pred)}")
    print(f"Testing score = {metrics.accuracy_score(y_test, y_preds)}")

    # Load the audio file
    y, sr = librosa.load(path_of_audio)

    # Compute the parameters
    sd = np.std(librosa.fft_frequencies(sr=sr, n_fft=2048))
    Q25 = np.quantile(librosa.hz_to_midi(librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr / 2)), 0.25)
    IQR = np.quantile(librosa.hz_to_midi(librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr / 2)), 0.75) - Q25
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hist, bin_edges = np.histogram(
        np.round(librosa.hz_to_midi(librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr / 2))), bins=np.arange(129))
    mode_bin = np.argmax(hist)
    mode = librosa.midi_to_hz(mode_bin + 1)
    Fun = librosa.feature.rms(y=y)
    meanfun = Fun.mean()
    # minfun = np.min(Fun)
    # maxfun = np.max(Fun)

    input_data = [sd / 10000, librosa.midi_to_hz(Q25) / 1000, librosa.midi_to_hz(IQR) / 1000, mode / 10000, meanfun]

    # saving the model
    filename_LinearSVC = 'trained_model.sav'
    pickle.dump(model, open(filename_LinearSVC, 'wb'))
    # loading the saved model
    loaded_model = pickle.load(open('trained_model.sav', 'rb'))

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    if (prediction[0] == 0):
        return 0
    else:
        return 1



import streamlit as st
import requests
from streamlit_lottie import st_lottie


st.set_page_config(
     page_title="voice recog...?", 
     page_icon='::star::',

     )

def load_lottie(url): # test url if you want to use your own lottie file 'valid url' or 'invalid url'
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
st.snow()

st.write('# We will recognize you by your voice...')

st_lottie(load_lottie('https://assets9.lottiefiles.com/packages/lf20_oa7rhSmqfm.json'), speed=2, height=400,width=1500)
st.write('---')

voice=st.file_uploader("uploade audio from your device", type=["wav"])

with st.container():
  
  col1,col2,col3=st.columns(3) 
  col1.write("Play the audio")
  col2.write("Stop the audio ")
  col3.write("Testing")
  with col1:
    play=st.button("Play")
    if(play):
       if voice is None:
         st.error('You must enter your voice first')
       else:
         st.audio(voice, format="wav", start_time=0,sample_rate=None)
         st_lottie(load_lottie('https://assets1.lottiefiles.com/packages/lf20_ihgw5fap.json'), speed=1, height=200  ,width=500)
    
  with col2:
     stop=st.button("Stop")
     
  with col3:
    if st.button('Predict'):
      predict=audio_recognition(voice)
      if voice is None:
         st.error('You must enter your voice first')
      else:
         if (predict== 0): 
            st.info("The person is female")
         else:
            st.info('The person is male')
            st.balloons


    


