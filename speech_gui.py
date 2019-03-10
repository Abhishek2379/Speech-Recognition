from tkinter import *
from tkinter import ttk
from PIL import ImageTk,Image
import matplotlib.image as  mimage
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm,metrics,datasets
from sklearn.externals import joblib
from scipy.io.wavfile import read
from LBG import lbg
from LPC import lpc
##from try1 import test
import os
from tkinter import messagebox as tkMessageBox
from train_svm import train
from test_svm import test

 
root = Tk()



canvas = Canvas(root, width = 400, height = 450)  
canvas.pack()  



frame = Frame(root)
frame.pack()

bottomframe = Frame(root)


button = Button(frame, text="Train",command= lambda:train(), fg="black",height=1,width=10)
button.pack(side=BOTTOM)



button = Button(frame, text="Test" ,command= lambda:test(), fg="black",height=1,width=10)
button.pack(side=LEFT)

button = Button(frame, text="Record", fg="black",height=1,width=10)
button.pack(side=RIGHT)



root.mainloop()
