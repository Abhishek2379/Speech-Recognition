import matplotlib.image as  mimage
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm,metrics,datasets
from sklearn.externals import joblib
from scipy.io.wavfile import read
from LBG import lbg
from LPC import lpc
import os

nfiltbank = 16
orderLPC = 20
def train():
    def training(nfiltbank, orderLPC):
        nSpeaker = 4
        nCentroid = 16
        codebooks_lpc = np.zeros((4*7, orderLPC* nCentroid))
        lpc_data = np.zeros((4*7))
        directory = os.getcwd() + '/sample';
        fname = str()

        for i in range(1,4):#this will traverse folders
            for j in range(1,8):#this will travere audio
                fname = (('./u%d/s%d.wav')%(i,j))
                print('Now speaker of ', str(i), 's sample no ',str(j),'features are being trained' )
                (fs,s) = read(directory + fname)
                lpc_coeff = lpc(s, fs, orderLPC)
                codebooks_lpc[i,:] = lbg(lpc_coeff, nCentroid).reshape(1,-1)
                lpc_data[i] = i
       
        print('Training complete')
            
        return (codebooks_lpc,lpc_data)
        
       
    (codebooks_lpc,lpc_data) = training(nfiltbank, orderLPC)
    svm_model = svm.SVC(kernel='linear');
    svm_model = svm_model.fit(codebooks_lpc,lpc_data);
    joblib.dump(svm_model,'trained_model_audio.pkl');
    print("Training Model Saved");
