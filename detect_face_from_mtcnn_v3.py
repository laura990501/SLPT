# -*- coding:utf-8  -*-
# Auther : Lyu jiayi
# Data : 2023/06/08
# Comment: This is a face detection framework based on MTCNN or RetinaFace.
# Modified: 2021/06/08 V1->V2: complete basic functions;test V2&V3 module; image path in txt changed to relative path
# Modified: 2021/06/09 V2->V3: write comments and make the code elegent.
# Modified: 2021/06/30 V3->V4: Add detection of facial box based on facial landmarks

import argparse
import os
import cv2
from mtcnn import MTCNN
import numpy as np
from Camera_landmarks_detect import detect_bbox


class FaceDetecter():
    def __init__(self, net_type="mtcnn", return_type="v1", scale=1.0):
        """
        Args:
            net_type: choice:[mtcnn,retinaface],default is mtcnn.
            return_type: choice:[v1,v2,v3], v1 is return left-top point and 宽高, v2 is return left-top and right-bottom points, v3 is return center point and 宽高.
            scale: float, default is 1.0. If larger than 1.0, enlarge the box with this ratio.
        """
        self.net_type = net_type
        self.return_type = return_type
        self.scale = scale

        # init network according the net_type
        if self.net_type == "mtcnn":
            self.net = MTCNN()
        else:
            # Initialize RetinaNet or other face detection models
            self.net=detect_bbox(img_path)
            

    def detect_img(self, img_path):
        """
        Args:
            img_path: str, img file path

        Return:
            box_info: list, box information according the return type
        """
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # detect face boxes
        boxes = self.net.detect_faces(img)

        # filter useless boxes with rules
        filtered_boxes = self.filter_boxes(boxes,img)

        # adjust box size according max width or height (maintain the wh ratio)
        adjusted_boxes = self.adjust_box_size(filtered_boxes, img.shape[1], img.shape[0])

        # return box information by return_type
        box_info = self.convert_to_return_type(adjusted_boxes)

        return box_info

    def filter_boxes(self, boxes, img):
        """
        Filter face box based on specific rules

        Args:
            boxes: img, boxes form detect_faces

        Return:
            filtered_boxes: list, 被选中的'box'、'confidence' 'keypoints'
        """
        filtered_boxes = []

        if len(boxes) != 0:

            box_areas = np.array([box['box'][2] * box['box'][3] for box in boxes]) / (img.shape[0] * img.shape[1])
            center_points = np.array(
                [(box['box'][0] + box['box'][2] // 2, box['box'][1] + box['box'][3] // 2) for box in boxes])
            center_differences = np.linalg.norm(center_points - np.array(img.shape[:2])[::-1] // 2, axis=1)
            confidences = np.array([box['confidence'] for box in boxes])
            idx_center_small = np.argsort(center_differences)
            idx_area_large = np.argsort(box_areas)[::-1]

            if idx_center_small[0] != idx_area_large[0]:
                if box_areas[idx_center_small[0]] > 0.15 and confidences[idx_center_small[0]] > 0.95:
                    max_largest_idx = idx_center_small[0]
                else:
                    max_largest_idx = idx_area_large[0]
            else:
                max_largest_idx = idx_center_small[0]

            filtered_boxes.append(boxes[max_largest_idx])
        else:
            height, width, _ = img.shape
            filtered_boxes.append({
                'box': [0, 0, width, height],  # 设置为图像边角的四个顶点坐标
                'confidence': 0.0,  # 设置置信度为0
                'keypoints': {}  # 设置空的关键点
            })

        return filtered_boxes


    def adjust_box_size(self, boxes, max_width, max_height):
        """
        resize the face box according to the scale

        Args:
            boxes: list, selected'box'、'confidence' 'keypoints'
            max_width, max_height

        Return:
            adjusted_boxes: list, enlarge box information by scale , and maintain its return type
                        'box'：tuple，represent the position and size of the face box (left-top_x,left-top_y,w,h)
                        'confidence'：confidence。
        """
        adjusted_boxes = []
        for box in boxes:
            box_info = box['box'] if 'box' in box else None
            if box_info is None or len(box_info) != 4:
                continue

            confidence = box['confidence']

            center_point = (int(box_info[0] + box_info[2] // 2), int(box_info[1] + box_info[3] // 2))

            edge_length = max(box_info[2], box_info[3])
            new_size = (int(edge_length * self.scale), int(edge_length * self.scale))
            right_bottom = (center_point[0] + new_size[0] // 2, center_point[1] + new_size[1] // 2)
            left_top = (center_point[0] - new_size[0] // 2, center_point[1] - new_size[1] // 2)
            right_bottom = (min(right_bottom[0], max_width), min(right_bottom[1], max_height))
            left_top = (max(left_top[0], 0), max(left_top[1], 0))
            adjusted_boxes.append(
                {'box': (left_top[0], left_top[1], right_bottom[0] - left_top[0], right_bottom[1] - left_top[1]),
                 'confidence': confidence})
        return adjusted_boxes


    def convert_to_return_type(self, adjusted_boxes):
        """
        return face box's information
        Args:
            adjusted_boxes: list, enlarge box information by scale , and maintain its return type，
                        'box'：tuple，reptresent the position and size of the face box (lefttop_x,lefttop_y,w,h)
                        'confidence'：confidence'
            return_type

        Return:
            retruns the representation that requires a face box based on x

        """
        box_info = []

        if self.return_type == "v1":
            # Convert to return type v1: left-top point and width, height
            box = adjusted_boxes[0]
            left_top = (box['box'][0], box['box'][1])
            width = box['box'][2]
            height = box['box'][3]
            # box_info = (left_top, width, height)
            box_info = (box['box'][0], box['box'][1], width, height)


        elif self.return_type == "v2":
            # Convert to return type v2: left-top and right-bottom points
            box = adjusted_boxes[0]
            left_top = (box['box'][0], box['box'][1])
            right_bottom = (box['box'][0] + box['box'][2], box['box'][1] + box['box'][3])
            # box_info = (left_top, right_bottom)
            box_info = (box['box'][0], box['box'][1],box['box'][0] + box['box'][2], box['box'][1] + box['box'][3])

        elif self.return_type == "v3":
            # Convert to return type v3: center point and width, height
            box = adjusted_boxes[0]
            center = (box['box'][0] + box['box'][2] // 2, box['box'][1] + box['box'][3] // 2)
            width = box['box'][2]
            height = box['box'][3]
            # box_info = (center, width, height)
            box_info = (box['box'][0] + box['box'][2] // 2, box['box'][1] + box['box'][3] // 2,box['box'][2], box['box'][3])

        print(box_info)

        return box_info

    def vis_lefttop_rightbottom(self, img_path, box_info):
        """
        left_top,right_bottom for visualization

        Args:
                   img_path: str, file path
                   box_info: list, box information according its return type

        returns:
                left_top,right_bottom for visualization
        """
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Visualize the result on the image
        vis_img = img.copy()
        if self.return_type == "v1":
            left, top, width, height = box_info  # Assuming box_info is in the format (left_top, width, height)
            right_bottom = (left + width, top + height)
            left_top = (left, top)

        elif self.return_type == "v2":

            left, top, right, bottom = box_info  # Assuming box_info is in the format (left_top, width, height)
            left_top = (left, top)
            right_bottom = (right, bottom)

        elif self.return_type == "v3":

            center_x, center_y, width, height = box_info
            left_top = (center_x - width // 2, center_y - height // 2)
            right_bottom = (center_x + width // 2, center_y + height // 2)

        return left_top, right_bottom


    def vis_result_folder(self, left_top, right_bottom,img_path):
        """
        visualize images for folder
        left_top, right_bottom from vis_lefttop_rightbottom

        """
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Visualize the result on the image
        vis_img = img.copy()
        cv2.rectangle(vis_img, left_top, right_bottom, (0, 255, 255), 2)
        # Save the visualization image
        output_dir = os.path.join(os.path.dirname(img_path), f"vis_{args.return_type}")
        os.makedirs(output_dir, exist_ok=True)
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"{img_name}")
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)  # Correct color space conversion
        cv2.imwrite(output_path, vis_img)


    def vis_result_singleimage(self, left_top, right_bottom,img_path):
        """
        visualize images for singleimage
        left_top, right_bottom from vis_lefttop_rightbottom

        """
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Visualize the result on the image
        vis_img = img.copy()
        cv2.rectangle(vis_img, left_top, right_bottom, (0, 255, 255), 2)
        # Save the visualization image
        output_dir = os.path.join(os.path.dirname(img_path), f"vis_{args.return_type}_singelimage")
        os.makedirs(output_dir, exist_ok=True)
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"{args.return_type}_{img_name}")
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)  # Correct color space conversion
        cv2.imwrite(output_path, vis_img)


    def detect_folder(self, img_folder):
        output_filename = f"output_{args.scale}_{args.net_type}_{args.return_type}.txt"  # 构造输出文件名
        output_path = os.path.join(img_folder, output_filename)  # 构造输出文件路径
        with open(output_path, 'w') as f:
            for filename in os.listdir(img_folder):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(img_folder, filename)
                    relative_path = os.path.relpath(img_path, img_folder)  # 获取相对路径
                    box_info = self.detect_img(img_path)
                    f.write(f"{relative_path} {' '.join([str(i) for i in box_info])}\n")
                    left_top, right_bottom=self.vis_lefttop_rightbottom(img_path,box_info)
                    self.vis_result_folder(left_top, right_bottom,img_path)

        return box_info


    def process_images(self, img_paths):
        """
        not used in this py
        can return face box for the folder

        Args:
            img_paths: list, list of image paths
        Returns:
            results: list, list of box information according its return type for whole folder
        """
        results = []
        for img_path in img_paths:
            box_info = self.detect_img(img_path)
            results.append(box_info)
            print(results)
        return results


def parse_args():
    parser = argparse.ArgumentParser(description='Face Detecting')
    parser.add_argument('--net_type', default="mtcnn", type=str, choices=["mtcnn", "retinaface","SLPT"], help='choose network')
    parser.add_argument('--return_type', default="v1", type=str, choices=["v1", "v2", "v3"],
                        help='choose return type')
    parser.add_argument('--scale', default=1.0, type=float, help='the scale of box size')
    parser.add_argument('--detect_folder', default=False, action='store_true', help="detect folder or img file")
    parser.add_argument('--path', required=True, type=str, help="img file path or folder path")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    detector = FaceDetecter(args.net_type, args.return_type, args.scale)
    if args.detect_folder:
        info=detector.detect_folder(args.path)
    else:
        info=detector.detect_img(args.path)
        left_top, right_bottom=detector.vis_lefttop_rightbottom(args.path, info)
        detector.vis_result_singleimage(left_top, right_bottom,args.path)



