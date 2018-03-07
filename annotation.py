import tensorflow as tf
import  numpy as np
import PIL.Image as Image
from skimage import io, transform
import TensorflowUtils as utils
import scipy.misc as misc
import matplotlib.pyplot as plt


def FCN(jpg_path, pb_file_path = "pb/FCN.pb"):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            input_x = sess.graph.get_tensor_by_name("input_image:0")
            print input_x
            output = sess.graph.get_tensor_by_name("inference/predict:0")
            print output
            probability = sess.graph.get_tensor_by_name("keep_probabilty:0")
            print probability
            
            img = io.imread(jpg_path)
            resize_image = misc.imresize(img,
                                         [224, 224], interp='nearest')
            resize_image = np.array([np.array(resize_image)])
            
            img_out = sess.run(output, feed_dict={input_x:resize_image, probability  :1.0})
            img_out = np.squeeze(img_out, axis = 3)

            # img_out = np.array(img_out)
            # print img_out
            # fig = plt.figure()
            # plt.imshow(img_out[0].astype(np.uint8))    

            utils.save_image(img_out[0].astype(np.uint8), "pb/", name="prediction")

            # plt.show()
            # print "img_out_softmax:",img_out_softmax
            # prediction_labels = np.argmax(img_out_softmax, axis=1)
            # print "label:",prediction_labels
