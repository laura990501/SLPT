import argparse
import os.path

from Config import cfg
from Config import update_config

from utils import create_logger
from SLPT import Sparse_alignment_network
from Dataloader import WFLW_test_Dataset

import torch, cv2, math
import numpy as np
import pprint

import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import Face_Detector
import utils

def parse_args():
    parser = argparse.ArgumentParser(description='Video Demo')

    # face detector
    parser.add_argument('-m', '--trained_model', default='./Weight/Face_Detector/yunet_final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--video_source', default='./Video/Video4.mp4', type=str, help='the image file to be detected')
    parser.add_argument('--confidence_threshold', default=0.7, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--vis_thres', default=0.3, type=float, help='visualization_threshold')
    parser.add_argument('--base_layers', default=16, type=int, help='the number of the output of the first layer')
    parser.add_argument('--device', default='cuda:0', help='which device the program will run on. cuda:0, cuda:1, ...')

    # landmark detector
    parser.add_argument('--modelDir', help='model directory', type=str, default='./Weight')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str, default='WFLW_6_layer.pth')
    parser.add_argument('--logDir', help='log directory', type=str, default='./log')
    parser.add_argument('--dataDir', help='data directory', type=str, default='./')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default=None)

    args = parser.parse_args()

    return args


def draw_landmark(landmark, image):
    #使用 cv2.circle 函数在图像上绘制关键点。对于每个关键点 (x, y)，它使用半径为 3 的圆点（颜色为绿色）在图像上绘制

    for (x, y) in (landmark + 0.5).astype(np.int32):
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

    return image


def crop_img(img, bbox, transform):
    x1, y1, x2, y2 = (bbox[:4] + 0.5).astype(np.int32)

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + w // 2
    cy = y1 + h // 2
    center = np.array([cx, cy])

    scale = max(math.ceil(x2) - math.floor(x1),
                math.ceil(y2) - math.floor(y1)) / 200.0

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    input, trans = utils.crop_v2(img, center, scale * 1.15, (256, 256))

    input = transform(input).unsqueeze(0)

    return input, trans

def face_detection(img, model, im_width, im_height):
    img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_NEAREST)
    img = np.float32(img)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    scale = torch.Tensor([im_width, im_height, im_width, im_height,
                          im_width, im_height, im_width, im_height,
                          im_width, im_height, im_width, im_height,
                          im_width, im_height])
    scale = scale.to(device)

    # feed forward
    loc, conf, iou = model(img)

    # post processing
    priorbox = Face_Detector.PriorBox(Face_Detector.cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = Face_Detector.decode(loc.data.squeeze(0), prior_data, Face_Detector.cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    cls_scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    iou_scores = iou.squeeze(0).data.cpu().numpy()[:, 0]
    # clamp here for the compatibility for ONNX
    _idx = np.where(iou_scores < 0.)
    iou_scores[_idx] = 0.
    _idx = np.where(iou_scores > 1.)
    iou_scores[_idx] = 1.
    scores = np.sqrt(cls_scores * iou_scores)

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    selected_idx = np.array([0, 1, 2, 3, 14])
    keep = Face_Detector.nms(dets[:, selected_idx], args.nms_threshold)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]

    return dets

def find_max_box(box_array):
    potential_box = []
    for b in dets:
        if b[14] < args.vis_thres:
            continue
        potential_box.append(np.array([b[0], b[1], b[2], b[3], b[14]]).astype('int')) #把符合条件的左上、右下和置信度组成的数组添加到potential_box

    if len(potential_box) > 0: #找最大人脸
        x1, y1, x2, y2 = (potential_box[0][:4]).astype(np.int32)
        Max_box = (x2 - x1) * (y2 - y1)
        Max_index = 0
        for index in range(1, len(potential_box)):
            x1, y1, x2, y2 = (potential_box[index][:4]).astype(np.int32)
            temp_box = (x2 - x1) * (y2 - y1)
            if temp_box >= Max_box:
                Max_box = temp_box
                Max_index = index
        return box_array[Max_index]
    else:
        return None


if __name__ == '__main__':
    args = parse_args() # 解析命令行参数
    update_config(cfg, args) # 更新配置信息

    device = torch.device(args.device) # 设置设备

    torch.set_grad_enabled(False) # 禁用梯度计算

    # Cuda
    cudnn.benchmark = True # 用CUDA加速
    torch.backends.cudnn.enabled = True # 启用CUDA

    # load face detector
    net = Face_Detector.YuFaceDetectNet(phase='test', size=None)  # 初始化人脸检测器
    net = Face_Detector.load_model(net, args.trained_model, True) # 加载模型权重
    net.eval() # 设置为评估模式
    net = net.to(device) # 将模型移动到指定设备上
    print('Finished loading Face Detector!')

    # 加载人脸关键点检测器
    model = Sparse_alignment_network(cfg.WFLW.NUM_POINT, cfg.MODEL.OUT_DIM,
                                     cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                     cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                     cfg.TRANSFORMER.FEED_DIM, cfg.WFLW.INITIAL_PATH, cfg)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda() # 使用多GPU进行训练

    checkpoint_file = os.path.join(args.modelDir, args.checkpoint)
    checkpoint = torch.load(checkpoint_file)
    pretrained_dict = {k: v for k, v in checkpoint.items()
                       if k in model.module.state_dict().keys()}
    model.module.load_state_dict(pretrained_dict)
    model.eval()

    print('Finished loading face landmark detector')

    # Camera Begin
    cap = cv2.VideoCapture(args.video_source)

    im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    normalize = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    frame_count = 0  # 记录帧号
    while True:
        _, frame = cap.read()
        if frame is None:
            break

        dets = face_detection(frame.copy(), net, 320, 240)
        if(len(dets) == 0):
            continue

        for bbox in dets:
            bbox[0] = int(bbox[0] / 320.0 * im_width + 0.5)
            bbox[2] = int(bbox[2] / 320.0 * im_width + 0.5)
            bbox[1] = int(bbox[1] / 240.0 * im_height + 0.5)
            bbox[3] = int(bbox[3] / 240.0 * im_height + 0.5)
            alignment_input, trans = crop_img(frame.copy(), bbox, normalize)

            outputs_initial = model(alignment_input.cuda())
            output = outputs_initial[2][0, -1, :, :].cpu().numpy()

            landmark = utils.transform_pixel_v2(output * cfg.MODEL.IMG_SIZE, trans, inverse=True)

            # Find the leftmost, rightmost, topmost, and bottommost points
            leftmost = min(landmark[:, 0])
            rightmost = max(landmark[:, 0])
            topmost = min(landmark[:, 1])
            bottommost = max(landmark[:, 1])

            # Draw face rectangle via landmarks
            cv2.rectangle(frame, (int(leftmost), int(topmost)), (int(rightmost), int(bottommost)), (0, 0, 255), 2)
            # Draw naive rectangle
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 225, 255), 3)


            frame = draw_landmark(landmark, frame)

        result_path = os.path.join('result_36', f'result_{cap.get(cv2.CAP_PROP_POS_FRAMES)}.jpg')
        cv2.imwrite(result_path, frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break