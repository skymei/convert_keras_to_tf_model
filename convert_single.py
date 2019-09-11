import logging
import os
import keras
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
from core_single.models import StyleTransferNetwork

logger = logging.getLogger(__name__)


class AndroidConvertSingleService(object):

    @classmethod
    def convert_single_android_model(cls, local_model_path):
        keras.backend.clear_session()
        keras.backend.set_learning_phase(0)

        StyleTransferNetwork.build(
            (256, 256),
            alpha=0.5,
            checkpoint_file=local_model_path
        )

        basename = os.path.basename(local_model_path)
        output_dir = os.path.dirname(local_model_path)

        # Freeze Graph
        cls._freeze_graph(basename, output_dir)
        # Optimize Graph
        cls._optimize_graph(basename, output_dir)

    @classmethod
    def _freeze_graph(cls, basename, output_dir):
        name, _ = os.path.splitext(basename)
        saver = tf.train.Saver()

        with keras.backend.get_session() as sess:
            checkpoint_filename = os.path.join(output_dir, '%s.ckpt' % name)
            output_graph_filename = os.path.join(output_dir, '%s_frozen.pb' % name)
            saver.save(sess, checkpoint_filename)

            tf.train.write_graph(
                sess.graph_def, output_dir, '%s_graph_def.pbtext' % name
            )

            freeze_graph.freeze_graph(
                input_graph=os.path.join(output_dir, '%s_graph_def.pbtext' % name),
                input_saver='',
                input_binary=False,
                input_checkpoint=checkpoint_filename,
                output_graph=output_graph_filename,
                output_node_names='deprocess_stylized_image_1/mul',
                restore_op_name="save/restore_all",
                filename_tensor_name="save/Const:0",
                clear_devices=True,
                initializer_nodes=None
            )
            logger.info('Saved frozen graph to: %s' % output_graph_filename)

    @classmethod
    def _optimize_graph(cls, basename, output_dir):
        name, _ = os.path.splitext(basename)
        frozen_graph_filename = os.path.join(output_dir, '%s_frozen.pb' % name)
        graph_def = cls.load_graph_def(frozen_graph_filename)

        optimized_graph = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def=graph_def,
            input_node_names=['input_1'],
            placeholder_type_enum=dtypes.float32.as_datatype_enum,
            output_node_names=['deprocess_stylized_image_1/mul'],
            toco_compatible=True
        )

        optimized_graph_filename = os.path.basename(
            frozen_graph_filename).replace('frozen', 'optimized')
        optimized_graph_filename = optimized_graph_filename
        tf.train.write_graph(
            optimized_graph, output_dir, optimized_graph_filename, as_text=False
        )
        logger.info('Saved optimized graph to: %s' %
                    os.path.join(output_dir, optimized_graph_filename))

    @classmethod
    def load_graph_def(cls, filename):
        input_graph_def = tf.GraphDef()
        with gfile.FastGFile(filename, 'rb') as file:
            data = file.read()
            input_graph_def.ParseFromString(data)
        return input_graph_def


AndroidConvertSingleService.convert_single_android_model('model_path')
