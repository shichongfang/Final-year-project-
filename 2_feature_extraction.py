import os
import sys
import collections
import numpy as np
import mxnet as mx

step0 = __import__('0_path')
step1 = __import__('1_image_preprocessing')


# read pre-trained resnet into models
def get_resnet_model():
    sym, arg_params, aux_params = mx.model.load_checkpoint(step0.resnet_model_prefix, 0)
    layers = sym.get_internals()
    feature_layer = layers['flatten0_output']
    mod = mx.mod.Module(symbol=feature_layer, context=step0.ctx, label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
    mod.set_params(arg_params, aux_params)
    
    return mod

# extract image feature for train images
def get_train_image_feature():
    mod = get_resnet_model()
    batch = collections.namedtuple('Batch', ['data'])
    # use the flatten feature after conv layers in resnet-101
    train_image_features = np.zeros((len(step1.train_id_list), 2048))
    for i in range(len(step1.train_id_list)):
        image_id = step1.train_id_list[i]
        image = step1.read_image(image_id, 'train')
        image = mx.nd.array(image, step0.ctx)
        image = batch([image])
        mod.forward(image)
        image_feature = mod.get_outputs()[0]
        image_feature = image_feature.asnumpy()[0]
        train_image_features[i] = image_feature
    
    return train_image_features

# extract image feature for test images
def get_test_image_feature():
    mod = get_resnet_model()
    batch = collections.namedtuple('Batch', ['data'])
    # use the flatten feature after conv layers in resnet-101
    test_image_features = np.zeros((len(step1.test_id_list), 2048))
    for i in range(len(step1.test_id_list)):
        image_id = step1.test_id_list[i]
        image = step1.read_image(image_id, 'test')
        image = mx.nd.array(image, step0.ctx)
        image = batch([image])
        mod.forward(image)
        image_feature = mod.get_outputs()[0]
        image_feature = image_feature.asnumpy()[0]
        test_image_features[i] = image_feature
    
    return test_image_features


if __name__ == '__main__':
    train_image_features = get_train_image_feature()
    np.save(os.path.join(step0.output_feature_dir, 'train.npy'), train_image_features)
    print 'Extracted features for train images.'
    test_image_features = get_test_image_feature()
    np.save(os.path.join(step0.output_feature_dir, 'test.npy'), test_image_features)
    print 'Extracted features for test images.'
    print 'Done.'

