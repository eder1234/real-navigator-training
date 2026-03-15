import os
import cv2
import numpy as np
import torch
#import pysift
import logging
logger = logging.getLogger(__name__)
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd

from models.matching import Matching
from models.utils import frame2tensor
from lightglue.utils import numpy_image_to_torch
from lightglue import viz2d

import sys
import os

# Add the directory containing the modules to the system path
#sys.path.append(os.path.abspath('/home/rodriguez/Documents/GitHub/accelerated_features'))
#from modulex.xfeat import XFeat

from matplotlib import pyplot as plt

class FeatureMatcher:
    def __init__(self, config, device='cpu', s_img=None, t_img=None, id=0, mode='star', threshold=0.5, feature=None):
        self.config = config
        self.device = device
        self.s_img = s_img
        self.t_img = t_img
        self.filtered_matched_points = None
        self.vm_id = 0
        self.id = id
        #self.feature = self.config['feature_matching']['descriptor'] # verify this
        self.feature = feature
        self.threshold = 0.5
        self.exctractor = None
        self.matcher = None
        self.image0, self.image1, self.m_kpts0, self.m_kpts1, self.matches01 = None, None, None, None, None
        self.mode = mode
        self.threshold = threshold
        self.set_feature(self.feature)

    def set_mode(self, mode):
        self.mode = mode

    def set_threshold(self, threshold):
        self.threshold = threshold
    
    def set_feature(self, feature):
        self.feature = feature
        if feature == 'LightGlue':
            self.extractor = SuperPoint(max_num_keypoints=400).eval().to(self.device) 
            self.matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(self.device)
            #self.extractor = SuperPoint(max_num_keypoints=200).eval().to(self.device) # default 2048
            #self.matcher = LightGlue(features="superpoint", filter_threshold=self.threshold).eval().to(self.device)

    def get_feature(self):
        return self.feature

    def set_target(self, t_img):
        self.t_img = t_img

    def set_current(self, s_img):
        self.s_img = s_img

    def compute_matches(self, s_img, t_img, vm_id=0):
        self.vm_id = vm_id
        #print(f"Using {self.feature} feature matching")
        #print(self.config['feature_matching']['descriptor'])
        if self.feature == 'XFeat':
            #print("Using XFeat feature matching")
            kp1, kp2 = self.matched_points_with_xfeat(s_img, t_img)
        if self.feature == 'LightGlue':
            #print("Using LightGlue feature matching")
            kp1, kp2 = self.filtered_matched_points_with_lightglue(s_img, t_img)
        if self.feature == 'SuperGlue':
            #print("Using SuperGlue feature matching")
            #threshold = self.config['thresholds']['confidence']
            kp1, kp2 = self.filtered_matched_points_with_superglue(s_img, t_img)
        if self.feature == 'AKAZE':
            #print("Using AKAZE feature matching")
            kp1, kp2 = self.matched_points_with_akaze(s_img, t_img)
        if self.feature == 'ORB':
            #print("Using ORB feature matching")
            kp1, kp2 = self.matched_points_with_orb(s_img, t_img)
        if self.feature == 'BRISK':
            #print("Using BRISK feature matching")
            kp1, kp2 = self.matched_points_with_brisk(s_img, t_img)
        if self.feature == 'SIFT':
            #print("Using SIFT feature matching")
            kp1, kp2 = self.matched_points_with_sift(s_img, t_img)
        #self.save_matched_points(s_img, t_img, kp1, kp2)
        return kp1, kp2
    
    '''
    def matched_points_with_xfeat(self, s_img, t_img):
        xfeat = XFeat()
        if self.mode == 'star':
            mkpts0, mkpts1 = xfeat.match_xfeat_star(s_img, t_img, top_k=8000)
        else:
            mkpts0, mkpts1 = xfeat.match_xfeat(s_img, t_img, top_k = 4096)
        return mkpts0, mkpts1
    '''
    def compute_rmse(self, p1, p2):
        return np.sqrt(np.mean((p1 - p2) ** 2))
    
    '''
    def matched_points_with_sift(self, s_img, t_img):
        MIN_MATCH_COUNT = 10
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        #print(f"Image data type: {s_img.dtype}, Image shape: {s_img.shape}")
        #print(f"Image data type: {t_img.dtype}, Image shape: {t_img.shape}")

        if len(s_img.shape) == 3:  # If the image is colored (assuming RGB)
            s_img = cv2.cvtColor(s_img, cv2.COLOR_RGB2GRAY)
        if len(t_img.shape) == 3:  # Same for the target image
            t_img = cv2.cvtColor(t_img, cv2.COLOR_RGB2GRAY)

        # Compute SIFT keypoints and descriptors
        kp1, des1 = pysift.computeKeypointsAndDescriptors(s_img)
        kp2, des2 = pysift.computeKeypointsAndDescriptors(t_img)

        # FLANN matcher
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = [m for m, n in matches if m.distance < self.threshold * n.distance]

        if len(good) > MIN_MATCH_COUNT:
            # Extract matched coordinates
            matched_kp1 = np.float32([kp1[m.queryIdx].pt for m in good])
            matched_kp2 = np.float32([kp2[m.trainIdx].pt for m in good])
            return matched_kp1, matched_kp2
        else:
            logger.warning(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
            return None, None
    '''
    def matched_points_with_akaze(self, s_img, t_img):
        # Initialize AKAZE detector
        akaze = cv2.AKAZE_create()
        # Find the keypoints and descriptors with AKAZE
        kp1, des1 = akaze.detectAndCompute(s_img, None)
        kp2, des2 = akaze.detectAndCompute(t_img, None)
        
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.threshold * n.distance:
                good_matches.append(m)
        
        # Extract location of good matches
        points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
        points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

        for i, match in enumerate(good_matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        return points1, points2

    def matched_points_with_orb(self, s_img, t_img):
        # Initialize ORB detector
        orb = cv2.ORB_create()
        # Find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(s_img, None)
        kp2, des2 = orb.detectAndCompute(t_img, None)
        
        # BFMatcher with default params and crossCheck=True for better matches
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        return points1, points2

    def matched_points_with_brisk(self, s_img, t_img):
        # Initialize BRISK detector
        brisk = cv2.BRISK_create()
        # Find the keypoints and descriptors with BRISK
        kp1, des1 = brisk.detectAndCompute(s_img, None)
        kp2, des2 = brisk.detectAndCompute(t_img, None)
        
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.threshold * n.distance:
                good_matches.append(m)
        
        # Extract location of good matches
        points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
        points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

        for i, match in enumerate(good_matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        return points1, points2

    '''
    def save_matched_points(self, s_img, t_img, kp1, kp2):
        if self.config['feature_matching']['descriptor'] == 'SuperGlue' and self.config['logs']['matched_points'] and len(kp1) > 0:
            matched_image = self.save_matched_points_with_superglue(s_img, t_img, kp1, kp2)
            match_img_path = os.path.join(self.config['paths']['LOGS_DIR'], f"matched_points/match_{self.vm_id:04d}_{len(kp1)}.png")
            cv2.imwrite(match_img_path, matched_image)       
        # Other features are not supported yet
    '''
    #############
    def save_matched_points(self, img_path):
        # Check if matched points logging is enabled in the configuration
        #if self.config['logs']['matched_points'] and len(kp1) > 0:
            #descriptor = self.config['feature_matching']['descriptor']
        matched_image = self.visualize_matched_points_by_color(self.s_img, self.t_img, self.kp1, self.kp2)#, descriptor)
        #match_img_path = os.path.join(self.config['log_matched_points'], f"{self.id:04d}_{self.feature}_match_{self.vm_id:04d}_{len(kp1)}.png")
        cv2.imwrite(img_path, matched_image)

    def save_matched_images_lightglue(self, path):
        axes = viz2d.plot_images([self.s_img, self.t_img])
        viz2d.plot_matches(self.kp1, self.kp2, color="lime", lw=0.2)
        if self.feature == 'LightGlue':
            viz2d.add_text(0, f'Stop after {self.matches01["stop"]} layers', fs=20)
        plt.savefig(path)

    def visualize_matched_points(self, s_img, t_img, kp1, kp2, descriptor):
        """ Draw lines connecting matched keypoints between image1 and image2 for any descriptor. """
        if len(s_img.shape) == 2:
            s_img = cv2.cvtColor(s_img, cv2.COLOR_GRAY2BGR)
        if len(t_img.shape) == 2:
            t_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2BGR)

        h1, w1 = s_img.shape[:2]
        h2, w2 = t_img.shape[:2]
        height = max(h1, h2)
        width = w1 + w2
        matched_image = np.zeros((height, width, 3), dtype=np.uint8)
        matched_image[:h1, :w1] = s_img
        matched_image[:h2, w1:w1+w2] = t_img

        for pt1, pt2 in zip(kp1, kp2):
            pt1 = (int(round(pt1[0])), int(round(pt1[1])))
            pt2 = (int(round(pt2[0] + w1)), int(round(pt2[1])))

            cv2.circle(matched_image, pt1, 3, (0, 255, 0), -1)
            cv2.circle(matched_image, pt2, 3, (0, 255, 0), -1)
            cv2.line(matched_image, pt1, pt2, (255, 0, 0), 1)

        # Resize for visualization if needed
        matched_image = cv2.resize(matched_image, (width, height))

        return matched_image

    def visualize_matched_points_by_color(self, s_img, t_img, kp1, kp2):
        """ Visualize matched keypoints in different colors without connecting lines. """
        #if len(s_img.shape) == 2:
        #    s_img = cv2.cvtColor(s_img, cv2.COLOR_GRAY2BGR)
        #if len(t_img.shape) == 2:
        #    t_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2BGR)

        h1, w1 = s_img.shape[:2]
        h2, w2 = t_img.shape[:2]
        height = max(h1, h2)
        width = w1 + w2
        matched_image = np.zeros((height, width, 3), dtype=np.uint8)
        matched_image[:h1, :w1] = s_img
        matched_image[:h2, w1:w1+w2] = t_img

        num_matches = len(kp1)
        colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num_matches)]

        for idx, (pt1, pt2) in enumerate(zip(kp1, kp2)):
            color = colors[idx]
            # Convert tensor elements to integers after rounding
            pt1 = (int(round(float(pt1[0]))), int(round(float(pt1[1]))))
            pt2 = (int(round(float(pt2[0]) + w1)), int(round(float(pt2[1]))))

            cv2.circle(matched_image, pt1, 3, color, -1)
            cv2.circle(matched_image, pt2, 3, color, -1)

        # Resize for visualization if needed
        matched_image = cv2.resize(matched_image, (width, height))

        return matched_image
    #############

    def save_matched_points_with_superglue(self, s_img, t_img, kp1, kp2):
        """ Draw lines connecting matched keypoints between image1 and image2. """
        if len(s_img.shape) == 2:
            s_img = cv2.cvtColor(s_img, cv2.COLOR_GRAY2BGR)
        if len(t_img.shape) == 2:
            t_img = cv2.cvtColor(t_img, cv2.COLOR_GRAY2BGR)

        h1, w1 = s_img.shape[:2]
        h2, w2 = t_img.shape[:2]
        height = max(h1, h2)
        width = w1 + w2
        matched_image = np.zeros((height, width, 3), dtype=np.uint8)
        matched_image[:h1, :w1] = s_img
        matched_image[:h2, w1:w1+w2] = t_img

        for pt1, pt2 in zip(kp1, kp2):
            pt1 = (int(round(pt1[0])), int(round(pt1[1])))
            pt2 = (int(round(pt2[0] + w1)), int(round(pt2[1])))

            cv2.circle(matched_image, pt1, 3, (0, 255, 0), -1)
            cv2.circle(matched_image, pt2, 3, (0, 255, 0), -1)
            cv2.line(matched_image, pt1,pt2, (255, 0, 0), 1)

        # Resize for visualization
        matched_image = cv2.resize(matched_image, (width , height))

        return matched_image

    def match_with_superglue(self, s_img, t_img):
        self.s_img = s_img
        self.t_img = t_img

        g_image1 = cv2.cvtColor(self.s_img, cv2.COLOR_BGR2GRAY)
        g_image2 = cv2.cvtColor(self.t_img, cv2.COLOR_BGR2GRAY)
        frame_tensor1 = frame2tensor(g_image1, self.device)
        frame_tensor2 = frame2tensor(g_image2, self.device)
        
        matching = Matching({'superpoint': {}, 'superglue': {'weights': 'indoor'}}).to(self.device).eval()
        with torch.no_grad():  # No need to track gradients
            pred = matching({'image0': frame_tensor1, 'image1': frame_tensor2})
        
        # Detach tensors before converting to NumPy arrays
        kpts0 = pred['keypoints0'][0].cpu().detach().numpy()
        kpts1 = pred['keypoints1'][0].cpu().detach().numpy()
        matches = pred['matches0'][0].cpu().detach().numpy()
        confidence = pred['matching_scores0'][0].cpu().detach().numpy()

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        conf = confidence[valid]

        return mkpts0, mkpts1, conf
    
    def select_high_confidence_points_with_superglue(self, mkp1, mkp2, confidences):
        kp1 = []
        kp2 = []

        for i, confidence in enumerate(confidences):
            if confidence > self.threshold:
                kp1.append(mkp1[i])
                kp2.append(mkp2[i])

        return np.asarray(kp1), np.asarray(kp2)
    
    def filtered_matched_points_with_superglue(self, s_img, t_img):
        mkpts0, mkpts1, conf = self.match_with_superglue(s_img, t_img)
        kp1, kp2 = self.select_high_confidence_points_with_superglue(mkpts0, mkpts1, conf)
        self.kp1 = kp1
        self.kp2 = kp2
        return self.kp1, self.kp2
    
    def filtered_matched_points_with_lightglue(self, s_img, t_img):
        self.s_img = s_img
        self.t_img = t_img
        feats0 = self.extractor.extract(numpy_image_to_torch(s_img).to(self.device))
        feats1 = self.extractor.extract(numpy_image_to_torch(t_img).to(self.device))
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension
        self.matches01 = matches01
        # Access the keypoints using the correct key, typically 'keypoints'
        kpt1 = feats0["keypoints"][matches01["matches"][..., 0]]
        kpt2 = feats1["keypoints"][matches01["matches"][..., 1]]

        #kp1 = [(int(kp[0].item()), int(kp[1].item())) for kp in kpt1]
        #kp2 = [(int(kp[0].item()), int(kp[1].item())) for kp in kpt2]
        kp1 = [[int(kp[0].item()), int(kp[1].item())] for kp in kpt1]
        kp2 = [[int(kp[0].item()), int(kp[1].item())] for kp in kpt2]
        # kp1 and kp2 should be numpy arrays perhaps

        self.kp1 = kp1
        self.kp2 = kp2
        return kp1, kp2
