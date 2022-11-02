import imp
import matplotlib.pyplot as plt
import cv2 as cv
import numpy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import imageio.v2 as imageio
from functions import *
morph_path='build/results/morph/'
#return "results/laplace.png"
def erosion(path):

    frame = cv.imread(path)
    seg=frame
    kernel = numpy.ones((5, 5), numpy.uint8); seg_eroded = cv.erode(seg, kernel, iterations = 1)
    plt.subplot(111),plt.imshow(seg_eroded), plt.title('erosion')   
    save_path=morph_path+"erosion.png"
    plt.savefig(save_path)
    global iou, dice
    dice=dice_coef(frame,seg_eroded)
    iou=Iou(frame,seg_eroded)
    return "results/morph/erosion.png"        

def dilatation(path):
    frame = cv.imread(path)
    seg=frame
    kernel = numpy.ones((5, 5), numpy.uint8); seg_dilated = cv.dilate(seg, kernel, iterations = 1)
    plt.subplot(111),plt.imshow(seg_dilated), plt.title('dilatation')  
    save_path=morph_path+"dilatation.png"
    plt.savefig(save_path)
    return "results/morph/dilatation.png"   

def opening(path):
    frame = cv.imread(path)
    seg=frame
    kernel = numpy.ones((5, 5), numpy.uint8); seg_opening = cv.morphologyEx(seg, cv.MORPH_OPEN, kernel)
    plt.subplot(111),plt.imshow(seg_opening), plt.title('ouverture')  
    save_path=morph_path+"ouverture.png"
    plt.savefig(save_path)
    return "results/morph/ouverture.png"  

def closure(path):
    frame = cv.imread(path)
    seg=frame
    kernel = numpy.ones((5, 5), numpy.uint8); seg_closing = cv.morphologyEx(seg, cv.MORPH_CLOSE, kernel)
    plt.subplot(111),plt.imshow(seg_closing), plt.title('Fermeture')  
    save_path=morph_path+"closing.png"
    plt.savefig(save_path)
    return "results/morph/closing.png"  


def morph_gradient(path):
    orig = imageio.imread(path)
    blue = cv.split(orig)[0]
    (retVal, orig_thresholded) = cv.threshold(blue, 30, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)
    kernel = numpy.ones((3, 3), numpy.uint8)
    Morph_grad = cv.morphologyEx(orig_thresholded, cv.MORPH_GRADIENT, kernel)
    plt.imshow(Morph_grad), plt.title('Morphological gradient') 
    save_path=morph_path+"grad_morph.png"
    plt.savefig(save_path)
    return "results/morph/grad_morph.png"  