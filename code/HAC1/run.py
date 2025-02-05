import os

import numpy as np

training_time1 = np.array([10, 10, 15, 20, 32, 32, 42, 35, 56, 55])
training_time1 *= 60
training_time2 = np.array([15, 13, 25, 26, 32, 46, 45, 51, 65, 75])
training_time2 *= 60
training_time3 = np.array([25, 28, 39, 43, 75, 60, 280, 179, 155, 100])
training_time3 *= 60
training_time4 = np.array([52, 30, 49, 39, 59, 119, 68, 79])
training_time4 *= 60

os.system("python train.py -i " + str(7) + " -d 1 -l 3 -t " + str(training_time1[7]))
for i in range(1):
    os.system("python train.py -i " + str(i + 8) + " -d 1 -l 3 -t " + str(training_time1[i + 8]))
    os.system("python train1.py -i " + str(i + 8) + " -d 2 -l 3 -t " + str(training_time2[i + 8]))
for i in range(10):
    os.system("python train.py -i " + str(i) + " -d 1 -l 3 -t " + str(training_time1[i]))
    os.system("python train1.py -i " + str(i) + " -d 2 -l 3 -t " + str(training_time2[i]))
for i in range(10):
    os.system("python train.py -i " + str(i) + " -d 1 -l 4 -t " + str(training_time1[i]))
    os.system("python train1.py -i " + str(i) + " -d 2 -l 4 -t " + str(training_time2[i]))
for i in range(10):
    os.system("python train.py -i " + str(i) + " -d 1 -l 5 -t " + str(training_time1[i]))
    os.system("python train1.py -i " + str(i) + " -d 2 -l 5 -t " + str(training_time2[i]))

'''for i in range(10):
    os.system("python train2.py -i " + str(i) + " -d 1 -l 2 -t " + str(training_time3[i]))
for i in range(8):
    os.system("python train3.py -i " + str(i + 2) + " -d 2 -l 2 -t " + str(training_time4[i]))

for i in range(10):
    os.system("python train2.py -i " + str(i) + " -d 1 -l 3 -t " + str(training_time3[i]))
for i in range(8):
    os.system("python train3.py -i " + str(i + 2) + " -d 2 -l 3 -t " + str(training_time4[i]))
'''
