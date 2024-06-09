import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import imageio
alphanum=[x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
#we'll use keras stringlookup to convert char->num and vice cersa
char_to_num=tf.keras.layers.StringLookup(vocabulary=alphanum,oov_token="")
#oov_token="" will place "" if it encounters the caracter which isn't seen beofre
num_to_char=tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),oov_token="",invert=True
    )

#Writing a function to load the video
def load_video(path:str)->List[float]:
  #we'll first create a video capture instance
  video_capture=cv2.VideoCapture(path)
  #we'll loop through each one of the frames and store it an array called frames
  frames=[]
  for _ in range(int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))):
    #read the frame
    ret, frame=video_capture.read()
    #convert the frame to grayscale by using tensorflow
    frame=tf.image.rgb_to_grayscale(frame)
    #isolate the mouth region
    frames.append(frame[190:236,80:220,:])
    #But, the original Lipnet paper actually uses an advanced technique to detect the mouth using DLIB
  #release the video capture instance
  video_capture.release()

  #standardise the data using mean,SD
  mean=tf.math.reduce_mean(frames)
  std=tf.math.reduce_std(tf.cast(frames,tf.float32))
  return tf.cast((frames - mean), tf.float32)/std

#create a fucntion to load alingments
def load_alignments(path:str)->List[str]:
  with open(path, 'r') as f:
    lines=f.readlines()
  #create an array tokens to store the numeric val of alinments by using char_to_num
  tokens=[]
  for line in lines:
    #split each one of lines in alignments
    line=line.split()
    #ignore if the alignment has character 'sil'
    if line[2]!='sil':
      #append in tokens
      tokens=[*tokens,'',line[2]]
  #return numeric val of tokens
  return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

#Now we'll create a fucntion to load the data - both video and alignments simultaneously and returns the preprocessed video and alignments
def load_data(path:str):
  path=bytes.decode(path.numpy())
  #splits the filename for
  file_name=path.split('/')[-1].split('.')[0]
  #create sep paths for both align and video
  #appending file_name to get video
  video_path=os.path.join('data','s1',f'{file_name}.mpg')
  #appending file_name to get alignment
  #remember alignment and video file names are same, check dataset
  align_path=os.path.join('data','alignments','s1',f'{file_name}.align')
  frames=load_video(video_path)
  alignments=load_alignments(align_path)
  return frames,alignments


