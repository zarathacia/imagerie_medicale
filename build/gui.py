
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path
import imageio.v2 as imageio

from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.figure import Figure
from numpy import histogram, outer
from morph import *
from segmentation import *
from functions import *
from filters import *
# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from PIL import ImageTk, Image

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


test_image_path="build/assets/frame442.png"
bg_color = "#FEFAE0"
bg_color_header = "#283618"
bg_color_side = "#606C38"
bg_color_button="#BC6C25"
bg_color_button_side="#606C38"
font_color="#000"
font_color_side="#fff"
font_color_title = "#FFFFFF"
window = Tk()

window.geometry("1300x720")
window.configure(bg = bg_color)
iou=0
dice=0
origin=Image.open(relative_to_assets("frame442.png")).resize((300,250), Image.ANTIALIAS)
res=Image.open(relative_to_assets("image_2.png"))
def update_image(filepath):
    global img
    frame = cv.imread(filepath)
    origine=cv.imread(test_image_path)
    #dice=dice_coef(frame,origine)
    #iou=Iou(frame,origine)
    
    print(OUTPUT_PATH)
    filepath=OUTPUT_PATH/ Path(filepath)
    img=Image.open(str(filepath)).resize((400,300), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    
    canvas.itemconfig(image_resultat,image=img)
    print(dice)
    print(iou)
    
def import_image():
    global img_origin
    global test_image_path
    filepath=import_img_path()
    set_text(entry_import_path,filepath)
    filepath=OUTPUT_PATH/ Path(filepath)
    img_origin=Image.open(str(filepath)).resize((300,250), Image.ANTIALIAS)
    img_origin = ImageTk.PhotoImage(img_origin)
    canvas.itemconfig(image_origin,image=img_origin)
    test_image_path="build/assets/frame.png"
    print(filepath)
    update_image(hist(test_image_path))
    os.replace(filepath, OUTPUT_PATH/ Path("assets/frame.png"))

canvas = Canvas(
    window,
    bg = bg_color,
    height = 720,
    width = 1300,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)


canvas.place(x = 0, y = 0)

canvas.create_rectangle(
    0.0,
    0.0,
    1440.0,
    91.0,
    fill=bg_color_header,
    outline="")
canvas.create_rectangle(
    0.0,
    91.0,
    270.0,
    1024.0,
    fill=bg_color_side,
    outline="")
canvas.create_text(
    34.0,
    31.0,
    anchor="nw",
    text="Medical Imaging Tools",
    fill=font_color_title,
    font=("Inter", 24 * -1)
)

#################Import Image
#Images

canvas.create_text(
    290.0,
    190.0,
    anchor="nw",
    text="Original Image:",
    fill="#000000",
    font=("Montserrat Medium", 15 * -1)
)

image_original = ImageTk.PhotoImage(origin)

image_origin = canvas.create_image(
    497.0,
    350.0,
    image=image_original
)
canvas.create_text(
    930.0,
    190.0,
    anchor="nw",
    text="Result:",
    fill="#000000",
    font=("Montserrat Medium", 15 * -1)
)

image_res = ImageTk.PhotoImage(res)

image_resultat = canvas.create_image(
    1050.0,
    400.0,
    image=image_res
)

canvas.create_text(
    294.0,
    131.0,
    anchor="nw",
    text="Path:",
    fill="#000000",
    font=("Montserrat Medium", 16 * -1)
)

entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    619.0,
    141.0,
    image=entry_image_1
)
entry_import_path = Text(
    bd=0,
    bg="#fff",
    highlightthickness=0
)
entry_import_path.place(
    x=353.0,
    y=121.0,
    width=500,
    height=30.0
)




    
button_image_1 = PhotoImage(
    file=relative_to_assets("import.png"))
button_import_image = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=import_image,
    relief="flat"
)
button_import_image.place(
    x=904.0,
    y=121.0,
    width=130.0,
    height=40.0
)

button_unet = Button(
    borderwidth=0,
    highlightthickness=0,
    bg=bg_color_button_side,
    text="Histogram",
    command=lambda: update_image(hist(test_image_path)),
    relief="flat"
)
button_unet.place(
    x=1200.0,
    y=121.0,
    width=130.0,
    height=40.0
)

##########################################
#Evaluation
canvas.create_text(
    295.0,
    536.0,
    anchor="nw",
    text="Evaluation:",
    fill="#000000",
    font=("Montserrat Medium", 16 * -1)
)
##############################IOU
canvas.create_text(
    348.0,
    572.0,
    anchor="nw",
    text="IOU",
    fill="#000000",
    font=("Montserrat Medium", 16 * -1)
)
entry_image_2 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(
    564.0,
    582.0,
    image=entry_image_2
)
entry_iou = Text(
    bd=0,
    bg="#fff",
    highlightthickness=0
)
entry_iou.place(
    x=406.0,
    y=567.0,
    width=300,
    height=28.0
)

##############################DICE
canvas.create_text(
    343.0,
    616.0,
    anchor="nw",
    text="DICE",
    fill="#000000",
    font=("Montserrat Medium", 16 * -1)
)
entry_image_3 = PhotoImage(
    file=relative_to_assets("entry_3.png"))
entry_bg_3 = canvas.create_image(
    564.0,
    626.0,
    image=entry_image_3
)
entry_dice = Text(
    bd=0,
    bg="#fff",
    highlightthickness=0
)
entry_dice.place(
    x=406.0,
    y=611.0,
    width=300.0,
    height=28.0
)


#################################Filtering
canvas.create_text(
    17.0,
    161.0,
    anchor="nw",
    text="Filtering:",
    fill="#FFFFFF",
    font=("Montserrat Medium", 16 * -1)
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_median = Button(
    image=button_image_2,
    borderwidth=0,
    bg=bg_color_button_side,
    highlightthickness=0,
    command=lambda: update_image(median(test_image_path)),
    relief="flat"
)
button_median.place(
    x=55.0,
    y=190.0,
    width=60.0,
    height=20.0
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_mean = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    bg=bg_color_button_side,
    command=lambda: update_image(mean(test_image_path)),
    relief="flat"
)
button_mean.place(
    x=56.0,
    y=208.0,
    width=45.0,
    height=20.0
)

button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_gaussian = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    bg=bg_color_button_side,
    command=lambda: update_image(gaussian(test_image_path)),
    relief="flat"
)
button_gaussian.place(
    x=55.0,
    y=226.0,
    width=73.0,
    height=20.0
)

############################Thresholding
canvas.create_text(
    14.0,
    255.0,
    anchor="nw",
    text="Thresholding",
    fill=font_color_title,
    font=("Montserrat Medium", 16 * -1)
)

button_image_17 = PhotoImage(
    file=relative_to_assets("button_17.png"))
button_binary = Button(
    image=button_image_17,
    borderwidth=0,
    highlightthickness=0,
    bg=bg_color_button_side,
    command=lambda: update_image(binary(test_image_path)),
    relief="flat"
)
button_binary.place(
    x=55.0,
    y=284.0,
    width=52.0,
    height=20.0
)

button_image_18 = PhotoImage(
    file=relative_to_assets("button_18.png"))
button_otsu = Button(
    image=button_image_18,
    borderwidth=0,
    highlightthickness=0,
    bg=bg_color_button_side,
    command=lambda: update_image(otsu(test_image_path)),
    relief="flat"
)
button_otsu.place(
    x=56.0,
    y=302.0,
    width=39.0,
    height=20.0
)
###################################Morph
canvas.create_text(
    12.0,
    331.0,
    anchor="nw",
    text="Morphological Transformations",
    fill=font_color_title,
    font=("Montserrat Medium", 16 * -1)
)

button_image_5 = PhotoImage(
    file=relative_to_assets("button_5.png"))
button_errosion = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    bg=bg_color_button_side,
    command=lambda: update_image(erosion(test_image_path)),
    relief="flat"
)
button_errosion.place(
    x=55.0,
    y=360.0,
    width=60.0,
    height=20.0
)

button_image_6 = PhotoImage(
    file=relative_to_assets("button_6.png"))
button_dialation = Button(
    image=button_image_6,
    borderwidth=0,
    highlightthickness=0,
    bg=bg_color_button_side,
    command=lambda: update_image(dilatation(test_image_path)),
    relief="flat"
)
button_dialation.place(
    x=55.0,
    y=378.0,
    width=72.0,
    height=20.0
)

button_image_7 = PhotoImage(
    file=relative_to_assets("button_7.png"))
button_opening = Button(
    image=button_image_7,
    borderwidth=0,
    highlightthickness=0,
    bg=bg_color_button_side,
    command=lambda: update_image(opening(test_image_path)),
    relief="flat"
)
button_opening.place(
    x=55.0,
    y=396.0,
    width=70.0,
    height=20.0
)

button_image_8 = PhotoImage(
    file=relative_to_assets("button_8.png"))
button_closing = Button(
    image=button_image_8,
    borderwidth=0,
    highlightthickness=0,
    bg=bg_color_button_side,
    command=lambda: update_image(closure(test_image_path)),
    relief="flat"
)
button_closing.place(
    x=55.0,
    y=414.0,
    width=60.0,
    height=20.0
)
########################################Contours
canvas.create_text(
    15.0,
    443.0,
    anchor="nw",
    text="Contours extraction",
    fill=font_color_title,
    font=("Montserrat Medium", 16 * -1)
)

button_image_9 = PhotoImage(
    file=relative_to_assets("button_9.png"))
button_sobel = Button(
    image=button_image_9,
    borderwidth=0,
    highlightthickness=0,
    bg=bg_color_button_side,
    command=lambda: update_image(sobel(test_image_path)),
    relief="flat"
)
button_sobel.place(
    x=56.0,
    y=472.0,
    width=45.0,
    height=20.0
)

button_image_10 = PhotoImage(
    file=relative_to_assets("button_10.png"))
button_laplacian = Button(
    image=button_image_10,
    borderwidth=0,
    highlightthickness=0,
    bg=bg_color_button_side,
    command=lambda: update_image(laplace(test_image_path)),
    relief="flat"
)
button_laplacian.place(
    x=53.0,
    y=490.0,
    width=77.0,
    height=20.0
)

button_image_11 = PhotoImage(
    file=relative_to_assets("button_11.png"))
button_gradient = Button(
    image=button_image_11,
    borderwidth=0,
    highlightthickness=0,
    bg=bg_color_button_side,
    command=lambda: update_image(morph_gradient(test_image_path)),
    relief="flat"
)
button_gradient.place(
    x=53.0,
    y=508.0,
    width=127.0,
    height=20.0
)
###############################segmentation
canvas.create_text(
    15.0,
    552.0,
    anchor="nw",
    text="Segmentation",
    fill="#FFFFFF",
    font=("Montserrat Medium", 16 * -1)
)

button_image_14 = PhotoImage(
    file=relative_to_assets("button_14.png"))
button_meanshift = Button(
    image=button_image_14,
    borderwidth=0,
    highlightthickness=0,
    bg=bg_color_button_side,
    command=lambda: update_image(meanshift(test_image_path)),
    relief="flat"
)
button_meanshift.place(
    x=54.0,
    y=581.0,
    width=88.0,
    height=20.0
)

button_image_15 = PhotoImage(
    file=relative_to_assets("button_15.png"))
button_kmeans = Button(
    image=button_image_15,
    borderwidth=0,
    highlightthickness=0,
    bg=bg_color_button_side,
    command=lambda: update_image(kmeans(test_image_path)),
    relief="flat"
)
button_kmeans.place(
    x=54.0,
    y=599.0,
    width=80.0,
    height=20.0
)

button_image_16 = PhotoImage(
    file=relative_to_assets("button_16.png"))
button_unet = Button(
    image=button_image_16,
    borderwidth=0,
    highlightthickness=0,
    bg=bg_color_button_side,
    command=lambda: update_image(unet(test_image_path)),
    relief="flat"
)
button_unet.place(
    x=55.0,
    y=617.0,
    width=71.0,
    height=20.0
)

###################################################################

window.resizable(False, False)
window.mainloop()

