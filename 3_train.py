import os
import numpy as np
import cPickle as pickle
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

step0 = __import__('0_path')
step1 = __import__('1_image_preprocessing')


# train a multilabel support vector machine for classification
def train_svm():
    train_image_features_file = os.path.join(step0.output_feature_dir, 'train.npy')
    train_image_features = np.load(train_image_features_file)

    classifier = OneVsRestClassifier(LinearSVC(multi_class='crammer_singer'))
    train_label_list = MultiLabelBinarizer().fit_transform(step1.train_label_list)
    classifier.fit(train_image_features, train_label_list)

    # save the model for prediction
    output_model_file = os.path.join(step0.output_model_dir, 'svm.pkl')
    with open(output_model_file, 'wb') as f:
        pickle.dump(classifier, f)


if __name__ == '__main__':
    train_svm()
    print 'Done.'

