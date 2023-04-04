from YOLOv4_SR_network import edsr
import cv2
import numpy as np
import tensorflow as tf
from YOLOv4_config import *
from YOLOv4_dataset import Dataset
from YOLOv4_utils import *
import sys



model_generator = edsr()
model_generator.load_weights("YOLOv4-for-studying/model_data/EDSR_X2_SRGAN-148800_Visdrone.h5")

#['train', 'validate', 'test']
data_type = 'test'

save_dir = "./YOLOv4-for-studying/dataset/Visdrone_DATASET/SR/"

dataset = Dataset(data_type, EVAL_MODE=False if data_type in ['train', 'validate'] else True)

for i,annotation in enumerate(dataset.annotations):
    image_path, bboxes_annotations = annotation

    image_name = image_path.split("/")[-1].split(".")[0]
    save_path = save_dir + image_name + ".npy"

    image = cv2.imread(image_path)
    bboxes = np.array([list(map(float, box.split(','))) for box in bboxes_annotations], np.float32)
    image, _ = image_preprocess(np.copy(image), TRAIN_INPUT_SIZE, np.copy(bboxes))
    _, bboxes = image_preprocess(np.copy(image), np.array(TRAIN_INPUT_SIZE)*2, np.copy(bboxes))

    image_sr = model_generator(image[np.newaxis,...]*255.0, training=False)[0]
    image_sr = np.maximum(np.minimum(image_sr,255.0), 0.0)
    
    np.save(save_path, image_sr)

    # cv2.imshow("test", np.array(image_sr, np.uint8))
    # if cv2.waitKey() == 'q':
    #     pass
    
    sys.stdout.write("\rLoaded image : {}".format(i))

    

print("\nFinished generating all SR-images ... ")    