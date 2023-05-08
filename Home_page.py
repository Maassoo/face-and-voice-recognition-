import streamlit as st
import requests
from streamlit_lottie import st_lottie
import numpy as np
from PIL import Image




st.set_page_config(
     page_title="Who's there...?", 
     page_icon='::star::',

     )

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
    left_column.write("#           choice")
    
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
               
                     


        

    # if st.button('Predict'):
    #         pred_Y = loaded_model.predict(sample)
            
    #         if pred_Y == 0:
    #             #st.write("## Predicted Status : ", result)
    #             st.write('### Congratulations ', name, '!! You are placed.')
    #             st.balloons()
    #         else:
    #             st.write('### Sorry ', name, '!! You are not placed.')