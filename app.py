import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#For using type annotations for documenting the code
from typing import List
import imageio
# Import all of the dependencies
from PIL import Image
import streamlit as st
import os
import imageio
import subprocess
import tensorflow as tf
#tf version 2.16.1
#keras version 3.3.3
from preprocessor import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar:
    # st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    logoimg = Image.open('IMAGE 2024-06-09 12:25:39.jpg')
    st.image(logoimg)
    st.title('LIP READER')
    st.info('This application is developed by FREAKY TECHIES.')

st.title('Welcome to LIP Reader')
# Generating a list of options or videos
options = os.listdir(os.path.join( 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns
col1, col2 = st.columns(2)

if options:
    input_file_path = os.path.join('data', 's1', selected_video)
    output_file_path = os.path.join('data', 's1', 'converted_video.mp4')
    # Rendering the video
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        subprocess.run(['ffmpeg', '-i', input_file_path, '-vcodec', 'libx264', output_file_path, '-y'], check=True)
        file_path = os.path.join('..', 'data', 's1', selected_video)
        with open(output_file_path, 'rb') as video:
            video_bytes = video.read()
            st.video(video_bytes)

    with col2:
        st.info('This is what the deep learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400)

        st.info('This is the output of the LIP READER as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [74], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
