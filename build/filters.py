import matplotlib.pyplot as plt
import cv2 as cv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import imageio.v2 as imageio


def save_image(filename,frame):
    if not cv.imwrite('build/results/'+filename, frame):
        raise Exception("Could not write image")
    return 'results/'+filename

def median(path):
    frame = cv.imread(path)
    filepath=save_image("median.png",cv.medianBlur(frame, 5))
    return filepath

def mean(path):
    frame = cv.imread(path)
    filepath=save_image("mean.png",cv.blur(frame,(3, 3)))
    return filepath

def gaussian(path):
    frame = cv.imread(path)
    filepath=save_image("gaussian.png",cv.GaussianBlur(frame,(5, 5), 3))
    return filepath
    

def binary(path):
    frame = cv.imread(path)
    (retVal, newImg) = cv.threshold(frame, 30, 255, cv.THRESH_BINARY) 
    print(retVal)
    filepath=save_image("binary.png",newImg)
    return filepath
def otsu(path):
    frame = cv.imread(path)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, thresholded = cv.threshold(frame_gray, 30, 255, cv.THRESH_BINARY | 
                                            cv.THRESH_OTSU)     
    filepath=save_image("otsu.png",thresholded)
    return filepath 
    
    
def sobel(path):
    orig = imageio.imread(path)
    orig_gray = cv.split(orig)[0]
    contour_sobel_gray = cv.Sobel(orig_gray, cv.CV_8U, 1, 0, ksize = 1)
    plt.subplot(111), plt.imshow(contour_sobel_gray), plt.title('Sobel')
    path='build/results/sobel.png'
    plt.savefig(path)
    return "results/sobel.png"

    
def laplace(path):
    orig = imageio.imread(path)
    orig_gray = cv.split(orig)[0]
    contour_laplace_gray = cv.Laplacian(orig_gray, cv.CV_8U, ksize = 3)
    plt.subplot(111), plt.imshow(contour_laplace_gray), plt.title('Laplace')
    path='build/results/laplace.png'
    plt.savefig(path)
    return "results/laplace.png"
