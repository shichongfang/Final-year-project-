import os
import sys
import cv2
import numpy as np

step0 = __import__('0_path')


# get train image id list
def get_train_id_list():
    train_id_list = []
    for image_file in os.listdir(step0.train_image_dir):
        image_id = image_file.split('_')[0]
        train_id_list.append(image_id)
    train_id_list = list(set(train_id_list))
    train_id_list.sort()
    
    return train_id_list

# get train image label list, each image is multilabeled
def get_train_label_list():
    label_dict = {}
    with open(step0.train_csv_file) as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            line = line.split(',')
            image_id = line[0]
            image_labels = [int(image_label) for image_label in line[1].split(' ')]
            label_dict[image_id] = image_labels
    
    train_id_list = get_train_id_list()
    train_label_list = []
    for image_id in train_id_list:
        train_label_list.append(label_dict[image_id])
    
    return train_label_list

# get test image id list
def get_test_id_list():
    test_id_list = []
    for image_file in os.listdir(step0.test_image_dir):
        image_id = image_file.split('_')[0]
        test_id_list.append(image_id)
    test_id_list = list(set(test_id_list))
    test_id_list.sort()
    
    return test_id_list

# read image via image id, using protein of interest (green) only
# preprocess the image to fit the network input
def read_image(image_id, split):
    image_file = image_id + '_green.png'
    if split == 'train':
        image_path = os.path.join(step0.train_image_dir, image_file)
    elif split == 'test':
        image_path = os.path.join(step0.test_image_dir, image_file)
    else:
        print 'Unknown split.'
        sys.exit(1)
    # read image using opencv
    image = cv2.imread(image_path)
    # convert to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # resize to fit network
    image = cv2.resize(image, (224, 224))
    # change channel order
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)
    # extend to a batch
    image = np.expand_dims(image, axis=0)
    
    return image


train_id_list = get_train_id_list()
train_label_list = get_train_label_list()
test_id_list = get_test_id_list()

if __name__ == '__main__':
    print 'Get {} train id and {} test id.'.format(len(train_id_list), len(test_id_list))
    print 'Done.'

