import os
import sys
import mxnet as mx


# data file path
train_image_dir = '/Users/housei/Desktop/FYP/train' # os.path.join('data', 'train')
train_csv_file = '/Users/housei/PycharmProjects/FYP/train.csv' # os.path.join('data', 'train.csv')
test_image_dir = '/Users/housei/Desktop/FYP/test' # os.path.join('data', 'test')
sample_submission_csv_file = '/Users/housei/PycharmProjects/FYP/sample_submission.csv' # os.path.join('data', 'sample_submission.csv')

# output file path
output_feature_dir = '/Users/housei/PycharmProjects/FYP/feature' # 'data'
output_model_dir = '/Users/housei/PycharmProjects/FYP/model' # 'data'
output_prediction_file = '/Users/housei/PycharmProjects/FYP/submission.csv' # os.path.join('data', 'submission.csv')


# devices for cnnc
ctx = mx.cpu(0) # mx.cpu()
# ctx = mx.gpu(1)
# pre-trained resnet-101 model on imagenet
# reference: http://data.dmlc.ml/mxnet/models/imagenet/resnet/101-layers/
resnet_model_prefix = '/Users/housei/PycharmProjects/FYP/resnet-101' # os.path.join('data', 'resnet-101')


# check configuration path
for each_file in [train_csv_file, sample_submission_csv_file, resnet_model_prefix + '-0000.params', resnet_model_prefix + '-symbol.json']:
    if not os.path.isfile(each_file):
        print '{} does not exist.'.format(each_file)
        sys.exit(1)
for each_dir in [train_image_dir, test_image_dir, output_feature_dir, output_model_dir]:
    if not os.path.isdir(each_dir):
        print '{} does not exist.'.format(each_dir)
        sys.exit(1)

if __name__ == '__main__':
    print 'Done.'

