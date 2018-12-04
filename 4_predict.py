import os
import numpy as np
import cPickle as pickle
from sklearn.preprocessing import MultiLabelBinarizer

step0 = __import__('0_path')
step1 = __import__('1_image_preprocessing')


# predict test image using the trained support vector machine
def predict_svm():
    test_image_features_file = os.path.join(step0.output_feature_dir, 'test.npy')
    test_image_features = np.load(test_image_features_file)

    # read svm from file
    output_model_file = os.path.join(step0.output_model_dir, 'svm.pkl')
    with open(output_model_file, 'rb') as f:
        classifier = pickle.load(f)

    # predict for test images
    mb = MultiLabelBinarizer()
    mb.fit_transform(step1.train_label_list)
    test_label_list = classifier.predict(test_image_features)
    test_label_list = mb.inverse_transform(test_label_list)
    label_dict = {}
    for image_id, image_label in zip(step1.test_id_list, test_label_list):
        if not image_label:
            image_label = (0,)
        label_dict[image_id] = image_label

    # write predict results and generate submission file
    output = []
    with open(step0.sample_submission_csv_file) as f:
        lines = f.readlines()
        output.append(lines[0].strip() + '\n')
        for line in lines[1:]:
            line = line.strip()
            line = line.split(',')
            image_id = line[0]
            image_label = [str(label) for label in label_dict[image_id]]
            output_line = image_id + ',' + ' '.join(image_label) + '\n'
            output.append(output_line)
    with open(step0.output_prediction_file, 'w') as f:
        f.writelines(output)


if __name__ == '__main__':
    predict_svm()
    print 'Done.'

