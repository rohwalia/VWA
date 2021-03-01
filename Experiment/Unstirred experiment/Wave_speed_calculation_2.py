import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
from scipy import fftpack
from scipy.signal import find_peaks
import os
images = glob.glob("Images_canny/*")
tensor = []
for picture in images:
    frame = cv2.imread(picture)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tensor.append(frame)
distance_graph = []
for image_number in range(1, len(tensor)):
    distance_list = []
    y_sel, x_sel = (np.where(abs(tensor[image_number][:, :]) > 200))
    for pixel in list(zip(y_sel, x_sel)):
        origin_y = pixel[0]
        origin_x = pixel[1]
        env_count = 0
        environment = []
        for env_x in range(-5, 6):
            for env_y in range(-5, 6):
                if abs(tensor[image_number][env_y+origin_y][env_x+origin_x]) > 200:
                    environment.append([env_x+origin_x, env_y+origin_y])
                    env_count = env_count + 1
        environment = np.array(environment)
        if env_count >8:
            sum_val = []
            for x_shift in range(-5, 6):
                for y_shift in range(-5, 6):
                    sum_num = 0
                    for i in environment:
                        if abs(tensor[image_number-1][i[1] + y_shift][i[0] + x_shift]) > 200:
                            sum_num =sum_num+1
                    sum_val.append([x_shift, y_shift,sum_num])
            sum_val = np.array(sum_val)
            index_max = np.argmax(sum_val[:,2])
            d_max= np.sqrt(sum_val[index_max][0]**2 + sum_val[index_max][1]**2)
            distance_list.append(d_max)
    distance_graph.append(sum(distance_list)/len(distance_list))

t = []
distance_graph_filtered = []
t_filtered = []
dt = 2
i = 0
while i<len(distance_graph):
    t.append(i*dt)
    if distance_graph[i] > sum(distance_graph) / len(distance_graph):
        distance_graph_filtered.append(distance_graph[i])
        t_filtered.append(t[i])
    i = i+1
plt.plot(t, distance_graph)
plt.savefig('distance_graph.png')
plt.show()
plt.plot(distance_graph_filtered)
plt.savefig('distance_graph_filtered.png')
distance_graph = np.array(distance_graph)
plt.show()
t = np.array(t)

print(sum(distance_graph) / len(distance_graph))
print(sum(distance_graph_filtered)/len(distance_graph_filtered))

data = pd.DataFrame(list(zip(t, distance_graph)), columns=["Time", "Diffusion rate"])
data2 = pd.DataFrame(list(zip(t_filtered, distance_graph_filtered)), columns=["Time", "Diffusion rate"])
with pd.ExcelWriter('Videos.xlsx') as writer:
    sheetname = "Diffusion rate"
    data.to_excel(writer, sheet_name= sheetname , index = False)
    data2.to_excel(writer, sheet_name= sheetname , index = False, startrow = 0, startcol = 5)
    workbook = writer.book
    worksheet = writer.sheets[sheetname]
    worksheet.write(0, 12, "Frequency")
    worksheet.insert_image(3, 12, "distance_graph.png")
    worksheet.insert_image(3, 20, "distance_graph_filtered.png")
