'''
@Author: David Vu
Run the pretrained model to extract 128D face features
'''

import tensorflow as tf
from architecture import inception_resnet_v1 as resnet
import numpy as np


KEEP_PROB = 0.6
TRAIN_MODEL = True
DO_NOT_TRAIN_MODEL = False

class FaceFeature(object):

    def __init__(self, face_rec_graph, model_path = 'models/model-20170512-110547.ckpt-250000'):

        print("\n [INFO] Loading the pretrained model...")
        with face_rec_graph.graph.as_default():
            self.sess = tf.Session()

            # Default input shape -> [160, 160, 3]
            self.x = tf.placeholder('float', [None, 160, 160, 3])
            # Reload the base model variables
            self.embeddings = tf.nn.l2_normalize(
                resnet.inference(self.x, KEEP_PROB, phase_train=DO_NOT_TRAIN_MODEL)[0], 1, 1e-10)

            # Restore the pretrained model
            saver = tf.train.Saver()
            saver.restore(self.sess, model_path)
            print("\n [INFO] Sucessfully loaded model.")


    def get_features(self, img_dataset):
        images = load_data_list(img_dataset, 160)
        return self.sess.run(self.embeddings, feed_dict = {self.x : images})



def whitening_transformation(x):
    """
        1. Changes the covariance matrix to identity.
        2. Changes mean vector to zero.
    """
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)

    return y

def load_data_list(img_dataset, image_size, pre_whiten = True):
    """
        1. Performs a whitening transformation on each image.
        2. Create a batch of whole dataset.
    """

    images = np.zeros((len(img_dataset), image_size, image_size, 3))
    i = 0

    for img in img_dataset:
        if img is not None:
            if pre_whiten:
                img = whitening_transformation(img)
            images[i, :, :, :] = img
            i += 1

    return images
