# Import the required libraries
from tkinter import *
from PIL import Image, ImageTk

# Create an instance of tkinter frame or window
win=Tk()

# Set the size of the tkinter window
win.geometry("700x350")

# Load the image
image=Image.open('D:/0-LocalData/syn_ateliers/build/median.png')

# Resize the image in the given (width, height)
img=image.resize((450, 350))

# Conver the image in TkImage
my_img=ImageTk.PhotoImage(img)

# Display the image with label
label=Label(win, image=my_img)
label.pack()

win.mainloop()