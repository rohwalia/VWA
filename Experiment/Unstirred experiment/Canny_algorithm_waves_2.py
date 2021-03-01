import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import os
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
frame_rate = 30 #frames per second
time_step = 10 #resolution of graph
videos = glob.glob("Videos_waves/*")
tensor = []
borders = []
for v in videos:
    frames = []
    tensor_picture = []
    cap = cv2.VideoCapture(v)
    i=0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if i == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            imgplot = plt.imshow(frame)
            coords = []
            def onclick(event):
                global ix, iy
                global coords
                ix, iy = event.xdata, event.ydata
                print('x = %d, y = %d' % (ix, iy))
                coords.append((ix, iy))
                if len(coords) == 4:
                    fig.canvas.mpl_disconnect(cid)
                    plt.close()
                return
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
            borders.append(coords)
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    frames = frames[::(int(frame_rate * time_step))]
    picture_number = 0
    for picture in frames:
        frames[picture_number] = frames[picture_number].tolist()
        row_number = 0
        for row in picture:
            column_number = 0
            for column in row:
                if Point(column_number, row_number).within(Polygon(coords)) == True:
                    pass
                else:
                    frames[picture_number][row_number][column_number] = [0,0,0]
                column_number = column_number + 1
            row_number = row_number + 1
        frames[picture_number] = np.array(frames[picture_number])
        frames[picture_number] = frames[picture_number].astype(np.uint8)
        picture_median = cv2.medianBlur(frames[picture_number], 3)
        #picture_median_gray = cv2.cvtColor(picture_median, cv2.COLOR_RGB2GRAY)
        """imgplot = plt.imshow(picture_median_gray, cmap="gray")
        plt.show()
        plt.hist(picture_median_gray.ravel(), 256)
        plt.xticks([]), plt.yticks([])
        plt.show()"""
        #otsu = cv2.threshold(picture_median_gray[np.where(picture_median_gray[:,:] != 0)], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        otsu_red = cv2.threshold(picture_median[np.where(picture_median[:, :, 0] != 0)], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        otsu_green = cv2.threshold(picture_median[np.where(picture_median[:, :, 1] != 0)], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        otsu_blue = cv2.threshold(picture_median[np.where(picture_median[:, :, 2] != 0)], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        otsu = (otsu_red + otsu_green + otsu_blue)/3
        print(otsu)
        canny_picture = cv2.Canny(picture_median, otsu/2, otsu, apertureSize = 3, L2gradient = True)
        tensor_picture.append(canny_picture)
        picture_number = picture_number + 1
    tensor.append(tensor_picture)
print("Pixel from ROI collected and gradient formed")
tensor = np.array(tensor)
print(tensor)
        #out.write(videos[i])
size = (column_number, row_number)
for video in tensor:
    out = cv2.VideoWriter('%s_canny.mp4' % os.path.splitext(os.path.basename(os.path.normpath(glob.glob("Videos_waves/*")[np.where(tensor == video)[0][0]])))[0]
        ,cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    for i in range(len(video)):
        cv2.imwrite("Images_canny/%d.jpg" %i, video[i])
        out.write(cv2.imread(glob.glob("Images_canny/*")[i]))
    out.release()
    cv2.destroyAllWindows()
print("Video and Images exported")

