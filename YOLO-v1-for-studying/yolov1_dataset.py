from distutils.log import debug
import ntpath
import pandas as pd
import cv2
import os
import numpy as np
from tensorflow import keras

#Parameters configuration
gcells = 7                  #grid cells
bboxes = 2                  #bouding box per cell
classes = 20                #total classes
batch_size = 4             #batch size when training

#Directory of image, label and csv file
IMG_DIR = "YOLO-v1-for-studying/data/images"
LABEL_DIR = "YOLO-v1-for-studying/data/labels"
CSV_DIR = "YOLO-v1-for-studying/data/100examples.csv"   #Examples

#For debugging, if True, showing 1 image with label
debug=False
rnd = np.random.randint(0,10)


#Singular data loading to extract image and label
def Image_Loading(img_path, label_path, S=gcells, B=bboxes, C=classes):
    #Image processing
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (448, 448))
    image = image/255.
    imaget = image
    #Label processing
    boxes = []
    with open(label_path) as f:
        for label in f.readlines():
            class_label, x, y, width, height = [float(x) if float(x) != int(float(x)) else int(x)
                                                for x in label.replace('\n','').split()]
            boxes.append([class_label, x, y, width, height])
    
    label_matrix = np.zeros([7,7,30])
    for box in boxes:
        class_label, x, y, width, height = box
        class_label = int(class_label)
        r,c = int(S*y), int(S*x)                            #r,c is the position of gridcell containing object center followoing rule (row, column)
        x_cell, y_cell = S*x-c, S*y-r                       #x_cell, y_cell is the position of object center in gridcell
        width_cell, height_cell = (width*S, height*S)       #width_cell, height_cell is bounding box posterior

        if label_matrix[r,c,24] == 0:                           #Just label the first object, 2nd object label is all zeros
                label_matrix[r,c,24] = 1                        #confidence score
                label_matrix[r,c,20:24] = [x_cell, y_cell, width_cell, height_cell]
                label_matrix[r,c,class_label] = 1               #one-hot coding for object class
        
    #Debugging image and label matching
        imaget = cv2.rectangle(imaget, (int((S*x-width_cell/2)*64), int((S*y-height_cell/2)*64)), (int((S*x+width_cell/2)*64), int((S*y+height_cell/2)*64)), (255,0,0), 2)
    if debug:
        cv2.imshow(ntpath.basename(img_path),imaget)
        while (cv2.waitKey() == 'q'):
            pass

    return image, label_matrix


#Loading each image path and label path
def VOCDataset_Path_Loading(csv_file=CSV_DIR, img_dir=IMG_DIR, label_dir=LABEL_DIR):
    annotations = pd.read_csv(csv_file)
    images_path = []
    labels_path = []
    for i in range(len(annotations)):
        #loading label parameters and saving to boxes
        img_path = os.path.join(img_dir, annotations.iloc[i, 0])
        label_path = os.path.join(label_dir, annotations.iloc[i, 1])
        images_path.append(img_path)
        labels_path.append(label_path)
    
    #Debugging
    if debug:
        print("Load image paths and label path successfully!")
        Image_Loading(img_path=images_path[rnd], label_path=labels_path[rnd])

    return images_path, labels_path


#Create Generator object to generate batch_size images for training
class My_Generator(keras.utils.Sequence):
    def __init__(self, images_path, labels_path, batch_size):           #3 inputs
        # super().__init__()
        self.images_path = images_path
        self.labels_path = labels_path
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.images_path)/(self.batch_size))).astype(np.int64)

    def __getitem__(self, index):
        image_batch = self.images_path[index*self.batch_size:(index+1)*self.batch_size]      #each batch contains [batch_size] image paths
        label_batch = self.labels_path[index*self.batch_size:(index+1)*self.batch_size]
        images = []
        labels = []

        for i in range(0, len(image_batch)):
            image, label_matrix = Image_Loading(img_path=image_batch[i], label_path=label_batch[i])
            images.append(image)
            labels.append(label_matrix)
        return np.array(images), np.array(labels)




#Generator for training data
# TRAIN_CSV_DIR = "YOLO-v1-for-studying/data/train.csv"
TRAIN_CSV_DIR = "YOLO-v1-for-studying/data/100examples.csv"
train_images_path, train_labels_path = VOCDataset_Path_Loading(csv_file=TRAIN_CSV_DIR)
My_training_generator = My_Generator(images_path=train_images_path, labels_path=train_labels_path, batch_size=batch_size)

# Generator for validation data
# VAL_CSV_DIR = "YOLO-v1-for-studying/data/test.csv"
VAL_CSV_DIR = "YOLO-v1-for-studying/data/8examples.csv"
val_images_path, val_labels_path = VOCDataset_Path_Loading(csv_file=TRAIN_CSV_DIR)
My_validation_generator = My_Generator(images_path=val_images_path, labels_path=val_labels_path, batch_size=batch_size)

#For Debugging
if debug:
    image_train, label_train = My_training_generator.__getitem__(0)
    print(image_train.shape)



# #Loading each image path and label path
# def VOCDataset_Loading(csv_file=CSV_DIR, img_dir=IMG_DIR, label_dir=LABEL_DIR):
#     annotations = pd.read_csv(csv_file)
#     loaded_images = []
#     loaded_labels = []
#     for i in range(len(annotations)):
#         #loading label parameters and saving to boxes
#         img_path = os.path.join(img_dir, annotations.iloc[i, 0])
#         label_path = os.path.join(label_dir, annotations.iloc[i, 1])
#         image, label_matrix = Image_Loading(img_path=img_path, label_path=label_path)
#         loaded_images.append(image)
#         loaded_images.append(label_matrix)
#         # cv2.imshow("abc",image)
#         # if cv2.waitKey() == 'q':
#         #     pass
#     return loaded_images, loaded_labels

# TRAIN_CSV_DIR = "YOLO-v1-for-studying/data/100examples.csv"
# training_images, training_labels = VOCDataset_Loading(TRAIN_CSV_DIR)
# VAL_CSV_DIR = "YOLO-v1-for-studying/data/8examples.csv"
# val_images, val_labels = VOCDataset_Loading(VAL_CSV_DIR)


if __name__ == "__main__":
    # VOCDataset_Loading()
    print("File dataset !")
