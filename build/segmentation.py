import matplotlib.pyplot as plt
import cv2 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import imageio.v2 as imageio
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from keras.models import load_model
seg_path='build/results/seg/'

def hist(path):
    img = imageio.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    plt.plot(hist),plt.title("Histogram")
    plt.xlim([0,256])
    save_path=seg_path+"hist.png"
    plt.savefig(save_path)
    return "results/seg/hist.png"  

def meanshift(path):
    img = imageio.imread(path)
    img = cv2.medianBlur(img, 3)

    flat_image = img.reshape((-1,3))
    flat_image = np.float32(flat_image)

    # meanshift
    bandwidth = estimate_bandwidth(flat_image, quantile=.06, n_samples=3000)
    ms = MeanShift(bandwidth=bandwidth,max_iter=800, bin_seeding=True)
    ms.fit(flat_image)
    labeled=ms.labels_
        # get number of segments
    segments = np.unique(labeled)
    print('Number of segments: ', segments.shape[0])

    # get the average color of each segment
    total = np.zeros((segments.shape[0], 3), dtype=float)
    count = np.zeros(total.shape, dtype=float)
    for i, label in enumerate(labeled):
        total[label] = total[label] + flat_image[i]
        count[label] += 1
    avg = total/count
    avg = np.uint8(avg)
    # cast the labeled image into the corresponding average color
    res = avg[labeled]
    result = res.reshape((img.shape))
    # show the result
    plt.imshow(result),plt.title("meanshift segmentation")
    save_path=seg_path+"meanshift.png"
    plt.savefig(save_path)
    return "results/seg/meanshift.png" 

def kmeans(path):
    image = imageio.imread(path)
    # convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)
    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # number of clusters (K)
    k = 3
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
    # show the image
    plt.imshow(segmented_image), plt.title("kmeans segmentation")
    save_path=seg_path+"kmeans.png"
    plt.savefig(save_path)
    return "results/seg/kmeans.png" 
 
def unet(path):
    #img = cv2.imread(path,0)
    
    #img_norm=img.resize(256,256)/255
    #img_input=np.expand_dims(img_norm, 0)
    #model = load_model('build/unetModel.hdf5')
    #prediction = (model.predict(np.aarray(img_input))[0,:,:,0] > 0.5).astype(np.uint8)
    #plt.subplot(233)
    #plt.title('Prediction on test image')
    #plt.imshow(prediction, cmap='gray')
    #save_path=seg_path+"unet.png"
    #plt.savefig(save_path)
    return "assets/image_2.png" 

