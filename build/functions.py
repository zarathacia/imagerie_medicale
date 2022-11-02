from tkinter import *
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
import os
import numpy as np
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.figure import Figure

import tensorflow as tf
def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)

def Iou(y_test,y_pred):
    intersection = np.logical_and(y_test, y_pred)
    union = np.logical_or(y_test, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    print("IoU socre is: ", iou_score)
    return iou_score

def init():
    print("hello")
    return None
def import_img_path():
    print("Import Image clicked")
    file = filedialog.askopenfile(mode='r', filetypes=[('Images', '*.png')])
    filepath="none"
    if file:
        filepath = os.path.abspath(file.name)
    return os.path.normcase(filepath)
def set_text(text_entry,text):
    text_entry.delete("1.0","end")
    text_entry.insert(INSERT,text)
    
