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
time_step = 5 #resolution of graph
videos = glob.glob("Videos_waves/*")
tensor = []
borders = []
def gradient(index_x, index_y, RGB):
    global R_mag, G_mag, B_mag
    R_gradient_x = RGB[index_x + 1][index_y][0] - RGB[index_x][index_y][0]
    R_gradient_y = RGB[index_x][index_y + 1][0] - RGB[index_x][index_y][0]
    R_vector = np.array([R_gradient_x, R_gradient_y])
    R_mag = np.sqrt(R_vector.dot(R_vector))
    G_gradient_x = RGB[index_x + 1][index_y][1] - RGB[index_x][index_y][1]
    G_gradient_y = RGB[index_x][index_y + 1][1] - RGB[index_x][index_y][1]
    G_vector = np.array([G_gradient_x, G_gradient_y])
    G_mag = np.sqrt(G_vector.dot(G_vector))
    B_gradient_x = RGB[index_x + 1][index_y][2] - RGB[index_x][index_y][2]
    B_gradient_y = RGB[index_x][index_y + 1][2] - RGB[index_x][index_y][2]
    B_vector = np.array([B_gradient_x, B_gradient_y])
    B_mag = np.sqrt(B_vector.dot(B_vector))
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
        gradient_picture = deepcopy(frames[picture_number])
        for i in range(column_number):
            for j in range(row_number):
                if j == row_number - 1:
                    pass
                elif i == column_number - 1:
                    pass
                else:
                    gradient(j, i, frames[picture_number])
                    #frames ist so aufgebaut, dass zuerst die y-koordinate und dann die x-koordinate nimmt
                    # also bei gradient alles anders herum lesen
                    if R_mag > 17:
                        gradient_picture[j][i] = [0, 0, 0]
                    else:
                        gradient_picture[j][i] = [255, 255, 255]
        tensor_picture.append(gradient_picture)
        picture_number = picture_number + 1
    tensor.append(tensor_picture)
print("Pixel from ROI collected and gradient formed")
tensor = np.array(tensor)
print(tensor)
        #out.write(videos[i])
size = (column_number, row_number)
for video in tensor:
    out = cv2.VideoWriter('%s.mp4' % os.path.splitext(os.path.basename(os.path.normpath(glob.glob("Videos_waves/*")[np.where(tensor == video)[0][0]])))[0]
        ,cv2.VideoWriter_fourcc(*"mp4v"), 6, size)
    for i in range(len(video)):
        cv2.imwrite("Images/%d.jpg" %i, video[i])
        out.write(cv2.imread(glob.glob("Images/*")[i]))
    out.release()
    cv2.destroyAllWindows()
print("Video and Images exported")