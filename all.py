import streamlit as st
import requests
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
import numpy as np
from PIL import Image
import cv2
import sys
from rembg import remove
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.svm import LinearSVC
import pandas as pd  
import seaborn as sns
from sklearn.model_selection import train_test_split
import librosa

with st.sidebar:
  selected=option_menu(
    menu_title="Knock...Knock.....",
    options=['Home','Facial recognition','Voice recognition']
  )
if selected=='Home':


         def load_lottie(url): # test url if you want to use your own lottie file 'valid url' or 'invalid url'
               r = requests.get(url)
               if r.status_code != 200:
                  return None
               return r.json()
               st.snow()

         st.write('# Welcome to you...')

         st_lottie(load_lottie('https://assets2.lottiefiles.com/packages/lf20_1pxqjqps.json'), speed=2, height=400,width=1500)

         st.write('# Knock...Knock.....')
         st.write("# Who's there...  ???")
         st_lottie(load_lottie('https://assets7.lottiefiles.com/packages/lf20_moAvAprIXV.json'), speed=1, height=500,width=400)

         st.write('---')

         with st.container():
         
            right_column, mid_column ,left_column = st.columns(3)
            right_column.write("# Login                     ")
            mid_column.write("       ")
            left_column.write("#       choice")
         
         with right_column:
               with st.form(key='my_form'):
                  username = st.text_input('Nickname')
                  password = st.text_input('Password')
                  st.form_submit_button('Submit')
            
         with mid_column:
               
               st_lottie(load_lottie('https://assets8.lottiefiles.com/packages/lf20_bmy4u2ew.json'), speed=1, height=600,width=250)
               
         with left_column:
            
               page_names=['Facial recognition','voice recognition']
               page=st.radio('Only one choice is either facial or voice recognition',page_names,index=1)
               if page =='Facial recognition':
                     st.info('Go to the facial recognition page')
                     imge=Image.open("face.PNG")
                     st.image(imge,width=350)
               else:
                     st.info('Go to the voice recognition page')
                     image = Image.open('voice.png')
                     st.image(image,width=350)
if selected=='Facial recognition':
 def remove_background(inputpath):
  outputpath = 'background_removed.png'
  input = Image.open(inputpath)
  output = remove(input)
  output.save(outputpath)
  return output

 def image_treatment(imagePath):
  image = cv2.imread(imagePath)

  # convert image to gray
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
  faces = faceCascade.detectMultiScale(gray,
    scaleFactor=1.3,
    minNeighbors=5, # number of neighbor faces (effect on the accuracy)
    minSize=(30, 30)
  )

  #print("[INFO] Found {0} Faces!".format(len(faces)))

  #print(faces)

  for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

  status = cv2.imwrite('faces_detected.jpg', image)
  #print("[INFO] Image faces_detected.jpg written to filesystem: ", status)

  img = Image.open(imagePath)
  box = (x, y, x+w, y+h)
  cropped_img = img.crop(box)
  Background_removed = cropped_img.save('Background_removed.jpg')
  return cropped_img

 def PCA_analysis(image):
  #this function should take the gray cropped image and apply pca to minimize the features
  img_arr = np.asarray(image)
  img_arr_reshaped = img_arr.reshape(-1, img_arr.shape[-1])
  pca = PCA(n_components=75)
  pca.fit(img_arr_reshaped)
  projected_data = pca.transform(img_arr_reshaped)
  reconstructed_data = pca.inverse_transform(projected_data)
  reconstructed_img = np.reshape(reconstructed_data, img_arr.shape)
  return reconstructed_img

  #Euclidean Distance another way
 def euclidean_distance(img1_vec, img2_vec):
    distance = np.linalg.norm(img1_vec - img2_vec)
    return distance
    
 def recognise(path1,path2,path3,path4):
    front_view = image_treatment(path1).convert("L")
    left_view = remove_background(path2).convert("L")
    right_view = remove_background(path3).convert("L")
    test_img = image_treatment(path4).convert("L")
    front_view_pca  = PCA_analysis(front_view)
    left_view_pca  = PCA_analysis(left_view)
    right_view_pca  = PCA_analysis(right_view)
    test_img_pca  = PCA_analysis(test_img)
    shape = (400, 400)
    front_view_resized = cv2.resize(front_view_pca, shape)
    left_view_resized = cv2.resize(left_view_pca, shape)
    right_view_resized = cv2.resize(right_view_pca, shape)
    test_img_resized = cv2.resize(test_img_pca, shape)
    front_vec = front_view_resized.flatten()
    left_vec = left_view_resized.flatten()
    right_vec = right_view_resized.flatten()
    test_vec = test_img_resized.flatten()
    pos_dist1 = euclidean_distance(front_vec, test_vec)
    pos_dist2 = euclidean_distance(left_vec, test_vec)
    pos_dist3 = euclidean_distance(right_vec, test_vec)
    distance_list = [pos_dist1,pos_dist2,pos_dist3]
    distance = sum(distance_list)/len(distance_list)
    threshold = 33900
    if (distance <= threshold):
       return 1
    else:
       return 0

 
 def filepathcorrect(file_path):
    file_path = str(file_path)
    y = False
    path_l = ''
    i = 0
    for x in file_path:
      if x == "'":
        y = True
        i += 1
        continue
      if i == 2:
        break
      if y:
         path_l = path_l + x
    return path_l
      



 def load_lottie(url): # test url if you want to use your own lottie file 'valid url' or 'invalid url'
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
 st.write('# Choose only one, either a camera or choose from your files ')
 st.write('---')
 page_names=['camera','choose from your files']
 page=st.radio('Choose only one',page_names,index=1)
 if page =='camera':
     st_lottie(load_lottie('https://assets5.lottiefiles.com/packages/lf20_gnhlz2ws.json'), speed=2, height=400,width=1500    )
     st.info('Smile to take photo')
     img=st.camera_input("Take a picture")
     st.image(img,width=350) 
       
 else:
        st.write('# choose from your files')
        st.info("Upload new image to train")
        image_right=st.file_uploader('Upload Right side',type=['png','jpg','jpeg'],accept_multiple_files=True)
        st.image(image_right,width=350)
        image_right = filepathcorrect(image_right)
        print(image_right)
        print('\n\n\n\n\n\n')
        image_left=st.file_uploader('Upload Left side',type=['png','jpg','jpeg'],accept_multiple_files=True)
        st.image(image_left,width=350)
        image_left = filepathcorrect(image_left)
        image_front=st.file_uploader('Upload  front side',type=['png','jpg','jpeg'],accept_multiple_files=True)
        st.image(image_front,width=350)
        image_front = filepathcorrect(image_front)
        st.info("Upload new image to test")
        image_test=st.file_uploader('Upload picture for testing',type=['png','jpg','jpeg'])
        st.image(image_test,width=350)
        image_test = filepathcorrect(image_test)
 # predict= recognise(image_right,image_left,image_front,image_test)

 if st.button('Predict'):
   predict= recognise(image_right,image_left,image_front,image_test)
   st.info(predict)
   if (predict== 0): 
        st.info("This is the same person")
   else:
         st.info('This is not the same person')
         st.balloons
if selected=='Voice recognition':
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


    



   

       

                     


        
