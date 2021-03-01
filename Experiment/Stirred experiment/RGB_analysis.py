import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import os
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy import fftpack
import xlsxwriter
frame_rate = 25 #frames per second
time_step = 0.2 #resolution of graph
videos = glob.glob("Videos/*")
tensor = []
borders = []
for v in videos:
    frames = []
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
            row_index = 0
            for column in row:
                if Point(column_number, row_number).within(Polygon(coords)) == True:
                    pass
                else:
                    del frames[picture_number][row_number][row_index]
                    row_index = row_index - 1
                column_number = column_number + 1
                row_index = row_index + 1
            row_number = row_number + 1
        frames[picture_number] = np.array(frames[picture_number])
        picture_number = picture_number + 1
    frames = np.array(frames)
    tensor.append(frames)
print("Pixel from ROI collected")
tensor_average = [None] * len(tensor)
video_number = 0
for videos in tensor:
    frame_number = 0
    frames_average = []
    for picture in videos:
        pixel = [j for i in picture for j in i]
        pixel = np.array(pixel)
        R = sum(pixel[:,0])/len(pixel[:,0])
        G = sum(pixel[:,1])/len(pixel[:,1])
        B = sum(pixel[:,2])/len(pixel[:,2])
        t = frame_number*time_step
        frames_average.append([t, R, G, B, frame_number, video_number])
        frame_number = frame_number + 1
    tensor_average[video_number] = deepcopy(frames_average)
    video_number = video_number + 1
tensor_average = np.array(tensor_average)
print("Average formed")
up = 0.01
for videos in tensor_average:
    R_fft = fftpack.fft(videos[:,1])
    R_freqs = fftpack.fftfreq(videos[:,1].size, d=time_step)
    R_power = np.abs(R_fft)[np.where(R_freqs > up)]
    R_freqs = R_freqs[np.where(R_freqs > up)]
    G_fft = fftpack.fft(videos[:,2])
    G_freqs = fftpack.fftfreq(videos[:,2].size, d=time_step)
    G_power = np.abs(R_fft)[np.where(G_freqs > up)]
    G_freqs = G_freqs[np.where(G_freqs > up)]
    B_fft = fftpack.fft(videos[:,3])
    B_freqs = fftpack.fftfreq(videos[:,3].size, d=time_step)
    B_power = np.abs(R_fft)[np.where(B_freqs > up)]
    B_freqs = B_freqs[np.where(B_freqs > up)]
    data = pd.DataFrame([l[:4] for l in videos], columns=["Time","R", "G", "B"])
    data2 = pd.DataFrame(list(zip(R_freqs, R_power, G_freqs, G_power, B_freqs, B_power)), columns=["R_freqs", "R_power", "G_freqs", "G_power", "B_freqs", "B_power"])
    with pd.ExcelWriter('../Videos.xlsx') as writer:
        sheetname = os.path.splitext(os.path.basename(os.path.normpath(glob.glob("Videos/*")[int(videos[0][5])])))[0]
        data.to_excel(writer, sheet_name= sheetname , index = False)
        data2.to_excel(writer, sheet_name= sheetname , index = False, startrow = 0, startcol = 5)
        workbook = writer.book
        worksheet = writer.sheets[sheetname]
        worksheet.write(0, 12, "R_T")
        worksheet.write(0, 13, "G_T")
        worksheet.write(0, 14, "B_T")
        worksheet.write(1, 12, 1/R_freqs[R_power.argmax()])
        worksheet.write(1, 13, 1/G_freqs[G_power.argmax()])
        worksheet.write(1, 14, 1/B_freqs[B_power.argmax()])
        chart = workbook.add_chart({'type': 'line'})
        chart2 = workbook.add_chart({'type': 'line'})

        chart.add_series({
            "name": "R",
            "categories": "=%s!$A$2:$A$%d" %(sheetname, len(videos)) ,
            'values': "=%s!$B$2:$B$%d" %(sheetname, len(videos)),
            'line':   {'color': 'red'},
        })
        chart.add_series({
            "name": "G",
            "categories": "=%s!$A$2:$A$%d" %(sheetname, len(videos)) ,
            'values': "=%s!$C$2:$C$%d" %(sheetname, len(videos)),
            'line':   {'color': 'green'},
        })
        chart.add_series({
            "name": "B",
            "categories": "=%s!$A$2:$A$%d" %(sheetname, len(videos)) ,
            'values': "=%s!$D$2:$D$%d" %(sheetname, len(videos)),
            'line':   {'color': 'blue'},
        })
        chart.set_y_axis({'name': 'RGB-values'})
        chart.set_x_axis({'name': 'Time(s)'})
        chart.set_legend({'position': 'bottom'})
        worksheet.insert_chart('M4', chart)

        chart2.add_series({
            "name": "R",
            "categories": "=%s!$F$2:$F$%d" % (sheetname, len(R_freqs)),
            'values': "=%s!$G$2:$G$%d" % (sheetname, len(R_freqs)),
            'line': {'color': 'red'},
        })
        chart2.add_series({
            "name": "G",
            "categories": "=%s!$H$2:$H$%d" % (sheetname, len(G_freqs)),
            'values': "=%s!$I$2:$I$%d" % (sheetname, len(G_freqs)),
            'line': {'color': 'green'},
        })
        chart2.add_series({
            "name": "B",
            "categories": "=%s!$J$2:$J$%d" % (sheetname, len(B_freqs)),
            'values': "=%s!$K$2:$K$%d" % (sheetname, len(B_freqs)),
            'line': {'color': 'blue'},
        })
        chart2.set_y_axis({'name': 'Power'})
        chart2.set_x_axis({'name': 'Frequency(Hz)'})
        chart2.set_legend({'position': 'bottom'})
        worksheet.insert_chart('M20', chart2)
print("Exported to excel")