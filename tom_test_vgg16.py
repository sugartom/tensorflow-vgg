import numpy as np
import tensorflow as tf

import vgg16
import utils as utils_vgg

# Yitao-TLS-Begin
import os
import sys
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat

# tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            # 'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
# tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS
# Yitao-TLS-End




def get_batch(batch_size):
	img = utils_vgg.load_image("./test_data/tiger.jpeg").reshape((1, 224, 224, 3))
	batch = np.concatenate((img, img, img, img, img, img, img, img, img, img), 0)

	return batch


batch_size = 10
images = tf.placeholder("float", [batch_size, 224, 224, 3])
batch = get_batch(batch_size)


with tf.Session() as sess:
  feed_dict = {images: batch}

  vgg = vgg16.Vgg16()
  with tf.name_scope("content_vgg"):
    vgg.build(images)
    pred = vgg.prob

    np_pred = sess.run(pred, feed_dict=feed_dict)
    print(np_pred)
    utils_vgg.print_prob(np_pred[0], './synset.txt')


    # Yitao-TLS-Begin
    export_path_base = "vgg_model"
    export_path = os.path.join(
        compat.as_bytes(export_path_base),
        compat.as_bytes(str(FLAGS.model_version)))
    print 'Exporting trained model to', export_path
    builder = saved_model_builder.SavedModelBuilder(export_path)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(images)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(pred)

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'images': tensor_info_x},
        outputs={'scores': tensor_info_y},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature,
        },
        legacy_init_op=legacy_init_op)

    builder.save()

    print('Done exporting!')
    # Yitao-TLS-End