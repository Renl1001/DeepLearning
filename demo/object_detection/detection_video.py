# coding: utf-8
import numpy as np
import os
import sys
import tensorflow as tf

import cv2
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'  # MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'  # 
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# 设置模型的路径和参数
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90


def detect_in_video():
    # 打开视频
    cap = cv2.VideoCapture('test_images/video1.mp4')
    # 获取视频的fps和大小
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #  定义VideoWriter
    out = cv2.VideoWriter('test_images/save/video1_out.mp4',
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                          size)
    # 加载模型到内存中
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # 加载标签映射
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')

            # 读取一帧
            ret, frame = cap.read()
            while ret:
                color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image_np_expanded = np.expand_dims(color_frame, axis=0)

                # run
                (boxes, scores, classes,
                 num) = sess.run([
                     detection_boxes, detection_scores, detection_classes,
                     num_detections
                 ],
                                 feed_dict={image_tensor: image_np_expanded})

                # 可视化检测结果
                vis_util.visualize_boxes_and_labels_on_image_array(
                    color_frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    min_score_thresh=.20)

                cv2.imshow('frame', color_frame)
                output_rgb = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
                out.write(output_rgb)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                ret, frame = cap.read()
            out.release()
            cap.release()
            cv2.destroyAllWindows()


def main():
    detect_in_video()


if __name__ == '__main__':
    main()
