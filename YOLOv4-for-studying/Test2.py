# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, ZeroPadding2D, MaxPool2D, Input, UpSampling2D
# from tensorflow.keras.regularizers import L2
# import cv2
# import colorsys
# import random
# import shutil
# import sys
# import tensorflow_hub as hub




# input = Input([100, 100, 3])
# conv1 = Conv2D(filters=10, kernel_size=3, padding='same')(input)
# conv2 = BatchNormalization()(conv1)
# conv2 = LeakyReLU()(conv2)
# conv2 = Conv2D(filters=20, kernel_size=3, padding='same', strides=2)(conv2)

# model1 = tf.keras.Model(input, conv1)
# model2 = tf.keras.Model(input, conv2)

# # test = tf.fill((1, 100,100,3), 2)

# # test1 = model1(test)
# # test2 = model2(test)

# # print(test1)
# # print("\n")
# # print(test2)
# # print("\n")

# model2.trainable_variables[0].assign(tf.fill(tf.shape(model2.trainable_variables[0]), 10.0))

# for i in range(2):
#     model1.trainable_variables[i].assign(model2.trainable_variables[i])



# print(model1.trainable_variables[0])
# print(model1.trainable_variables[1])
# print("\n")
# print(model2.trainable_variables[0])
# print(model2.trainable_variables[1])

# print(len(model1.trainable_variables))
# print(len(model2.trainable_variables))

import numpy as np

import math
First_line = input()
# print(First_line)
N, M, Q = list(map(int, First_line.split()))
vacants = [[1 for _ in range(M)] for _ in range(N)]


List_staff = []
dict_staff_id = {}
for _ in range(Q):
    temp = input()
    staff_type, staff_id = temp.split()
    List_staff.append([staff_type, staff_id])
    if staff_id not in dict_staff_id:
        dict_staff_id[staff_id] = 0

total_slot = 0
for i in vacants:
    total_slot+= sum(i)


def calculate_safety_level(x, y):
    safety_level = 100
    pos = None
    for i in range(N):
        for j in range(M):
            if vacants[i][j] == 2:
                temp = math.sqrt((x-i)**2 + (y-j)**2)
                if safety_level > temp:
                    safety_level = temp
                    pos = [i, j]
    return safety_level
        

def get_safe_seat():
    best_seat = None
    level_of_safety = 0
    for i in range(N):
        for j in range(M):
            if vacants[i][j] == 1:
                temp = calculate_safety_level(i, j)
                if level_of_safety < temp:
                    level_of_safety = temp
                    best_seat = [i, j]
    return best_seat

def check_having_vacants():
    for i in range(N):
        for j in range(M):
            if vacants[i][j] == 1:
                return True
    return False

def check_permiss_vacant(x,y):
    if x-1 >= 0:
        if vacants[x-1][y] == 2:
            return False
    if x+1 <= N-1:
        if vacants[x+1][y] == 2:
            return False
    if y-1 >= 0:
        if vacants[x][y-1] == 2:
            return False
    if y+1 <= M-1:
        if vacants[x][y+1] == 2:
            return False
    return True

sitting_pos_of_staff = {}


for staff in List_staff:

    if dict_staff_id[staff[1]] == 1 and staff[0] == "In":
        print(staff[1] + " already seated.")
    elif dict_staff_id[staff[1]] == 2 and staff[0] == "In":
        print(staff[1] + " already ate lunch.")
    elif dict_staff_id[staff[1]] == 0 and staff[0] == "Out":
        print(staff[1] + " didn't eat lunch.")
    elif dict_staff_id[staff[1]] == 2 and staff[0] == "Out":
        print(staff[1] + " already left seat.")

    #case "In" and do not have lunch
    elif dict_staff_id[staff[1]] == 0 and staff[0] == "In":
        temp = 0
        for i in vacants:
            temp+= sum(i)
        if check_having_vacants():       #check if having any vacants
            if vacants[0][0] == 1 and temp == total_slot:
                print(staff[1] + " gets the seat (%d, %d)." % (1, 1))
                vacants[0][0] = 2   #assign no vacant
                if M > 1:
                    vacants[0][1] = 0   #assign no vacant
                if N > 1:
                    vacants[1][0] = 0   #assign no vacant
                dict_staff_id[staff[1]] = 1 #assign eating
                sitting_pos_of_staff[staff[1]] = [0, 0]
            else:
                best_seat = get_safe_seat()
                print(staff[1] + " gets the seat (%d, %d)." % (best_seat[0]+1, best_seat[1]+1))
                vacants[best_seat[0]][best_seat[1]] = 2
                #4 corners
                if best_seat[0] == 0 and best_seat[1] == 0:
                    if N > 1:
                        vacants[best_seat[0]+1][best_seat[1]] = 0
                    if M > 1:
                        vacants[best_seat[0]][best_seat[1]+1] = 0

                elif best_seat[0] == N-1 and best_seat[1] == 0:
                    if N > 1:
                        vacants[best_seat[0]-1][best_seat[1]] = 0
                    if M > 1:
                        vacants[best_seat[0]][best_seat[1]+1] = 0
                elif best_seat[0] == 0 and best_seat[1] == M-1:
                    if N > 1:
                        vacants[best_seat[0]+1][best_seat[1]] = 0
                    if M > 1:
                        vacants[best_seat[0]][best_seat[1]-1] = 0
                elif best_seat[0] == N-1 and best_seat[1] == M-1:
                    if N > 1:
                        vacants[best_seat[0]-1][best_seat[1]] = 0
                    if M > 1:
                        vacants[best_seat[0]][best_seat[1]-1] = 0
                #4 edges
                elif best_seat[0] == 0:
                    if M > 1:
                        vacants[best_seat[0]][best_seat[1]-1] = 0
                        if best_seat[1]+1!=M:
                            vacants[best_seat[0]][best_seat[1]+1] = 0
                    if N > 1:
                        vacants[best_seat[0]+1][best_seat[1]] = 0
                elif best_seat[0] == N-1:
                    if M > 1:
                        vacants[best_seat[0]][best_seat[1]-1] = 0
                        if best_seat[1]+1!=M:
                            vacants[best_seat[0]][best_seat[1]+1] = 0
                    if N > 1:
                        vacants[best_seat[0]-1][best_seat[1]] = 0
                elif best_seat[1] == 0:
                    if N > 1:
                        vacants[best_seat[0]-1][best_seat[1]] = 0
                        if best_seat[0]+1!=N:
                            vacants[best_seat[0]+1][best_seat[1]] = 0
                    if M > 1:
                        vacants[best_seat[0]][best_seat[1]+1] = 0
                elif best_seat[1] == M-1:
                    if N > 1:
                        vacants[best_seat[0]-1][best_seat[1]] = 0
                        if best_seat[0]+1!=N:
                            vacants[best_seat[0]+1][best_seat[1]] = 0
                    if M > 1:
                        vacants[best_seat[0]][best_seat[1]-1] = 0
                #inside
                else:
                    if N > 1:
                        vacants[best_seat[0]-1][best_seat[1]] = 0
                        if best_seat[0]+1!=N:
                            vacants[best_seat[0]+1][best_seat[1]] = 0
                    if M > 1:
                        vacants[best_seat[0]][best_seat[1]-1] = 0
                        if best_seat[1]+1!=M:
                            vacants[best_seat[0]][best_seat[1]+1] = 0
                # print(vacants)    
                dict_staff_id[staff[1]] = 1
                sitting_pos_of_staff[staff[1]] = best_seat
        else:
            print("There are no more seats.")
    
    #case "Out" and eating lunch
    elif dict_staff_id[staff[1]] == 1 and staff[0] == "Out":
        best_seat = sitting_pos_of_staff[staff[1]]
        del sitting_pos_of_staff[staff[1]]
        print(staff[1] + " leaves from the seat (%d, %d)." % (best_seat[0]+1, best_seat[1]+1))
        dict_staff_id[staff[1]] = 2 #assign had lunch
        vacants[best_seat[0]][best_seat[1]] = 1
        #4 corners
        if best_seat[0] == 0 and best_seat[1] == 0: #top left
            if N > 1 and check_permiss_vacant(best_seat[0]+1, best_seat[1]):
                vacants[best_seat[0]+1][best_seat[1]] = 1
            if M > 1 and check_permiss_vacant(best_seat[0], best_seat[1]+1):
                vacants[best_seat[0]][best_seat[1]+1] = 1
        elif best_seat[0] == N-1 and best_seat[1] == 0:   #bottom left
            if N > 1 and check_permiss_vacant(best_seat[0]-1, best_seat[1]):
                vacants[best_seat[0]-1][best_seat[1]] = 1
            if M > 1 and check_permiss_vacant(best_seat[0], best_seat[1]+1):
                vacants[best_seat[0]][best_seat[1]+1] = 1
        elif best_seat[0] == 0 and best_seat[1] == M-1:   #top right
            if N > 1 and check_permiss_vacant(best_seat[0]+1, best_seat[1]):
                vacants[best_seat[0]+1][best_seat[1]] = 1
            if M > 1 and check_permiss_vacant(best_seat[0], best_seat[1]-1):
                vacants[best_seat[0]][best_seat[1]-1] = 1
        elif best_seat[0] == N-1 and best_seat[1] == M-1:   #bottom right
            if N > 1 and check_permiss_vacant(best_seat[0]-1, best_seat[1]):
                vacants[best_seat[0]-1][best_seat[1]] = 1
            if M > 1 and check_permiss_vacant(best_seat[0], best_seat[1]-1):
                vacants[best_seat[0]][best_seat[1]-1] = 1
        #4 edges
        elif best_seat[0] == 0:
            if M > 1:
                if check_permiss_vacant(best_seat[0], best_seat[1]-1):
                    vacants[best_seat[0]][best_seat[1]-1] = 1
                if best_seat[1]+1!=M and check_permiss_vacant(best_seat[0], best_seat[1]+1):
                    vacants[best_seat[0]][best_seat[1]+1] = 1
            if N > 1 and check_permiss_vacant(best_seat[0]+1, best_seat[1]):
                vacants[best_seat[0]+1][best_seat[1]] = 1
        elif best_seat[0] == N-1:
            if M > 1:
                if check_permiss_vacant(best_seat[0], best_seat[1]-1):
                    vacants[best_seat[0]][best_seat[1]-1] = 1
                if best_seat[1]+1!=M and check_permiss_vacant(best_seat[0], best_seat[1]+1):
                    vacants[best_seat[0]][best_seat[1]+1] = 1
            if N > 1 and check_permiss_vacant(best_seat[0]-1, best_seat[1]):
                vacants[best_seat[0]-1][best_seat[1]] = 1
        elif best_seat[1] == 0:
            if N > 1:
                if check_permiss_vacant(best_seat[0]-1, best_seat[1]):
                    vacants[best_seat[0]-1][best_seat[1]] = 1
                if best_seat[0]+1!=N and check_permiss_vacant(best_seat[0]+1, best_seat[1]):
                    vacants[best_seat[0]+1][best_seat[1]] = 1
            if M > 1 and check_permiss_vacant(best_seat[0], best_seat[1]-1):
                vacants[best_seat[0]][best_seat[1]-1] = 1
        elif best_seat[1] == M-1:
            if N > 1:
                if check_permiss_vacant(best_seat[0]-1, best_seat[1]):
                    vacants[best_seat[0]-1][best_seat[1]] = 1
                if best_seat[0]+1!=N and check_permiss_vacant(best_seat[0]+1, best_seat[1]):
                    vacants[best_seat[0]+1][best_seat[1]] = 1
            if M > 1 and check_permiss_vacant(best_seat[0], best_seat[1]-1):
                vacants[best_seat[0]][best_seat[1]-1] = 1
        #inside
        else:
            if N > 1:
                if check_permiss_vacant(best_seat[0]-1, best_seat[1]):
                    vacants[best_seat[0]-1][best_seat[1]] = 1
                if best_seat[0]+1!=N and check_permiss_vacant(best_seat[0]+1, best_seat[1]):
                    vacants[best_seat[0]+1][best_seat[1]] = 1
            if M > 1:
                if check_permiss_vacant(best_seat[0], best_seat[1]-1):
                    vacants[best_seat[0]][best_seat[1]-1] = 1
                if best_seat[1]+1!=M and check_permiss_vacant(best_seat[0], best_seat[1]+1):
                    vacants[best_seat[0]][best_seat[1]+1] = 1

