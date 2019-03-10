from __future__ import division
import numpy as np
from scipy.io.wavfile import read
from LBG import EUDistance
from LBG import lbg
from LPC import lpc
import os
import matplotlib.image as mimage
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn import svm, metrics,datasets

nfiltbank = 16  #(4*speaker)
orderLPC = 20 #(5*speaker)
nSpeaker = 4
nCentroid = 16
def test():
    #code to detect name
    def detect_name(a):
        if(a==1):
            return "Snehankit"
        elif(a==2):
            return "Swapnil"
        elif(a==3):
            return "Saiprasad"
        elif(a==4):
            return "Abhishek"
    #take a query audio'\
    path_q = './sample/u1/s8.wav';
    path_q_img = './sample_images/u1/1.jpg';
    #Now, we should display the user.
    im_q = mimage.imread(path_q_img);
    plt.figure(10);
    plt.subplot(1,2,1);
    plt.imshow(im_q,cmap="gray");
    plt.title("Query Sound");
    user_name = detect_name(1)
    plt.text(350,800,user_name)
    print("Voice of :: %s"%(user_name))
    #Let us test the testing sample with pkl file
    (fs,s) = read(path_q)
    lpc_coeff = lpc(s, fs, orderLPC)
    v1 = lbg(lpc_coeff, nCentroid).reshape(1,-1)
    #load trained model;
    svm_model = joblib.load('trained_model_audio.pkl');
    pred = svm_model.predict(v1);
    print("Detected Voice :: ",pred);
    path_pred = ('./sample_images/u%d/1.jpg'%int(pred[0]));
    im_pred = mimage.imread(path_pred);
    plt.subplot(1,2,2);
    plt.imshow(im_pred,cmap='gray');
    plt.title("Detected Voice");
    user_name = detect_name(int(pred[0]))
    plt.text(450,800,user_name)
    print("Detected Voice is of :: %s"%(user_name))
    plt.show()
