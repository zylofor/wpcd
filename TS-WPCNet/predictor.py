from datetime import datetime

import pika
import os
import json
import base64

import argparse
from predictor import VisualizationDemo
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
import multiprocessing as mp
from centernet.config import add_centernet_config
from detectron2.utils.video_visualizer import VideoVisualizer
import time
import numpy as np
import cv2
from urllib import request
from PIL import Image
from UDCS import UDCS
import math
import torch


NUM_CLASSES = 1


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_centernet_config(cfg)

    cfg.MODEL.CENTERNET.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    if cfg.MODEL.META_ARCHITECTURE in ['ProposalNetwork', 'CenterNetDetector']:
        cfg.MODEL.CENTERNET.INFERENCE_TH = args.confidence_threshold
        cfg.MODEL.CENTERNET.NMS_TH = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/CenterNet2_R50_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
        default="server/out_imgs"
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

class RequestHandler():


    def __init__(self, queue_name, user, pwd, ip, port):

        self.queue_name = queue_name
        self.credentials = pika.PlainCredentials(user, pwd)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(ip, port, '/', self.credentials, heartbeat=0))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue_name, durable=True)
        self.response = ResponseHandler(queue_name='land_inspection_response')

    def callback(self, channel, method, properties, body):
        print(datetime.now())
        print("receive", body.decode())
        if json.loads(body.decode())["imgName"] == "heartbeat":
            print("heartbeat")
        else:
            message, status = convert(json.loads(body.decode())["imgName"], json.loads(body.decode())["url"])

            result = json.dumps(message,cls=NpEncoder)

            self.response.startrespones(self.channel, result)
            if status:
                print(" [x] Done")
                # end_time = time.time()
                #
                # print("Total time is {}".format(end_time - start_time))
                print(datetime.now())

    def startconsume(self):

        print(' [*] Waiting for messages. To exit press CTRL+C')
        # self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=self.queue_name, on_message_callback=self.callback, auto_ack=True)
        self.channel.start_consuming()


class ResponseHandler:


    def __init__(self, queue_name, user='admin',
                 pwd='admin',
                 ip='127.0.0.1',
                 port=5672):

        self.queue_name = queue_name
        # self.credentials = pika.PlainCredentials(user, pwd)
        # self.connection = pika.BlockingConnection(
        #     pika.ConnectionParameters(ip, port, '/', self.credentials, heartbeat=0))
        # self.channel = self.connection.channel()
        # self.channel.queue_declare(queue=self.queue_name, durable=True)

    def startrespones(self, channel, message):

        self.channel = channel
        self.channel.basic_publish(
            exchange='',
            routing_key=self.queue_name,
            body=message,
            properties=pika.BasicProperties(
                delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE,
            ))
        print(" [x] Sent %r" % message)
        # self.connection.close()

def cutImage(imageName,box,plt_img,clusterData):
    roi = np.array(plt_img.crop(tuple(box)))
    clusterData.append(roi)
    return clusterData


def output_the_results_json(predictions,plt_img,imageName,resultFile):
    global w, h
    clusterData = []
    # TODO:Save the results infomation.
    for k, pred_box in enumerate(predictions['instances']._fields['pred_boxes']):
        box = np.array(pred_box.cpu())
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = math.ceil(box[0]), math.ceil(box[1]), math.ceil(box[2]), math.ceil(box[3])
        w, h = math.ceil(bottom_right_x - top_left_x), math.ceil(bottom_right_y - top_left_y)
        point_cx, point_cy = math.ceil(top_left_x + 0.5 * w), math.ceil(top_left_y + 0.5 * h)
        coordinateInfo = [point_cx, point_cy]

        if distCategories is True:
            xxyy = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
            clusterDataList = cutImage(imageName,xxyy,plt_img,clusterData)

        inferenceInfo = {
            'imageName': imageName,
            'imageSize': list(plt_img.size),
            'coordinateInfo': coordinateInfo,
            'workpieceClass': 'null',
            'bestClassifyCase': 'null'
        }
        resultFile.append(inferenceInfo)


    clusterResults, best_k = UDCS(clusterDataList)

    bestClusterResults = clusterResults[best_k - 2]


    for i, each_workpieceClass in enumerate(clusterResults.T):
        resultFile[i]["workpieceClass"] = list(each_workpieceClass)
        resultFile[i]["bestClassifyCase"] = int(best_k)

    return bestClusterResults,best_k


def convert(imageName, imageUrl):
    imageUrl = "http://" + imageUrl
    resp = request.urlopen(imageUrl)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    plt_img = Image.fromarray(img)

    try:
        resultFile = []

        output_dir = 'server/out_imgs'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        start_time = time.time()

        predictions, imageRGB = demo.run_on_image(img)

        if predictions is not None:
            bestClusterResults, best_k = output_the_results_json(predictions,plt_img,imageName,resultFile)
            instances = predictions["instances"].to(cpu_device)
            visualized_output = visualizer.draw_instance_predictions(imageRGB, instances, bestClusterResults, best_k)

        else:
            resultFile = None

        if 'instances' in predictions:
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    imageUrl, len(predictions["instances"]), time.time() - start_time
                )
            )
        else:
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    imageUrl, len(predictions["proposals"]), time.time() - start_time
                )
            )

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(imageUrl))
                visualized_output.save(out_filename)

        print(resultFile)

        return resultFile, 1

    except Exception as error:
        message = {
            "imageName": imageName,
            "output": 'nan',
            "status": 'error: ' + str(error)
        }
        return message, 0


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    cpu_device = torch.device("cpu")
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    dsize = 1024
    cfg = setup_cfg(args)
    resultsDir = "server/output_json"
    demo = VisualizationDemo(cfg)
    output_file = None
    distCategories = True

    visualizer = VideoVisualizer(
        MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        ),
        instance_mode=ColorMode.IMAGE)

    convRequester = RequestHandler(queue_name='land_inspection_request',
                                   user='admin',
                                   pwd='admin',
                                   ip='127.0.0.1',
                                   port=5672)
    convRequester.startconsume()
