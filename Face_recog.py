import cv2
import sys
from PIL import Image
from rembg import remove
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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

# recognise('/content/training_img1.jpeg','/content/training_img2.jpeg','/content/training_img3.jpeg','/content/testing_img.jpeg')
import streamlit as st
import requests
from streamlit_lottie import st_lottie
import numpy as np
from PIL import Image
st.set_page_config(page_title="Facial recognition", page_icon='::star::')

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
st.write('# Facial recognition...')
imge=Image.open("face.PNG")
st.image(imge,width=900)
st.write('---')
st.write('# choose from your files')
st.info("Upload new image to train")
image_front=st.file_uploader('Upload front side',type=['png','jpg','jpeg'],accept_multiple_files=True)
st.image(image_front,width=350)
image_front = filepathcorrect(image_front)
image_left=st.file_uploader('Upload Left side',type=['png','jpg','jpeg'],accept_multiple_files=True)
st.image(image_left,width=350)
image_left = filepathcorrect(image_left)
image_right=st.file_uploader('Upload right side',type=['png','jpg','jpeg'],accept_multiple_files=True)
st.image(image_right,width=350)
image_right = filepathcorrect(image_right)
st.info("Upload new image to test or take it with camera?...")
page_names=['camera','choose from your files']
page=st.radio('Choose only one',page_names,index=1)
if page =='camera':
    st_lottie(load_lottie('https://assets5.lottiefiles.com/packages/lf20_gnhlz2ws.json'), speed=2, height=400,width=600  )
    st.info('Smile to take photo')
    image_test=st.camera_input("testing image")
    st.image(image_test,width=350)
    image_test = filepathcorrect(image_test)
else:
    image_test=st.file_uploader('Upload testing photo',type=['png','jpg','jpeg'],accept_multiple_files=True)
    st.image(image_test,width=350)
    image_test = filepathcorrect(image_test)
        

if st.button('Predict'):
   predict= recognise(image_right,image_left,image_front,image_test)
   st.info(predict)
   if (predict== 0): 
        st.info("This is the same person")
   else:
         st.info('This is not the same person')
         st.balloons

        

       
