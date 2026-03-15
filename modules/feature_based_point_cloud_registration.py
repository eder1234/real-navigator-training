import numpy as np
import quaternion
from modules.feature_matcher import FeatureMatcher
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

class FeatureBasedPointCloudRegistration:
    def __init__(self, config, device, id_run, feature_nav_conf, feature_mode, topological_map, manual_operation):
        # Initialize the feature matcher
        self.feature_nav = FeatureMatcher(config=config, device=device, id=id_run)
        self.feature_nav.set_threshold(0.1)
        self.feature_nav.set_feature(feature_nav_conf)
        self.feature_nav.set_mode(feature_mode)
        
        # Set operation modes
        self.topological_map = topological_map
        self.manual_operation = manual_operation
        
        # Initialize bot_lost flag
        self.bot_lost = True
    def compute_relative_poses(self, source_colors, source_depths, target_colors, target_depths, max_workers=None):
        """
        Compute relative poses for multiple source-target pairs.

        Parameters:
            source_colors (List[np.ndarray]): List of source color images.
            source_depths (List[np.ndarray]): List of source depth maps.
            target_colors (List[np.ndarray]): List of target color images.
            target_depths (List[np.ndarray]): List of target depth maps.
            max_workers (int, optional): Maximum number of worker threads/processes.

        Returns:
            pd.DataFrame: DataFrame containing pose estimation results for each pair.
        """
        assert len(source_colors) == len(source_depths) == len(target_colors) == len(target_depths), \
            "All input lists must have the same length."
        
        num_pairs = len(source_colors)
        results = []
        
        # Choose the executor based on the nature of compute_relative_pose
        # If compute_relative_pose is CPU-bound, use ProcessPoolExecutor
        # If it's I/O-bound or releases the GIL, use ThreadPoolExecutor
        # Here, we'll assume it's CPU-bound
        executor = ProcessPoolExecutor(max_workers=max_workers)
        
        # Prepare a list of futures
        futures = []
        for i in range(num_pairs):
            futures.append(
                executor.submit(
                    self.compute_relative_pose,
                    source_colors[i],
                    source_depths[i],
                    target_colors[i],
                    target_depths[i]
                )
            )
        
        # Use tqdm to display progress
        for future in tqdm(as_completed(futures), total=num_pairs, desc="Computing Relative Poses"):
            result = future.result()
            results.append(result)
        
        executor.shutdown()
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        return results_df

    def compute_relative_pose(self, source_color, source_depth, target_color, target_depth):
        # Feature matching between source and target images
        kp_source, kp_target = self.feature_nav.compute_matches(source_color, target_color)
        
        # Check for sufficient matches
        if len(kp_source) < 4 and (self.topological_map or not self.manual_operation):
            self.bot_lost = True
            est_quaternion = None
            rmse = None
            est_t_source_target = None
            return self.bot_lost, est_quaternion, rmse, est_t_source_target
        
        pc_source_h = self.generate_pc_in_cam_ref_frame(source_depth)
        pc_target_h = self.generate_pc_in_cam_ref_frame(target_depth)
        
        ipc_source = self.get_ipc_from_pc(pc_source_h, kp_source)
        ipc_target = self.get_ipc_from_pc(pc_target_h, kp_target)
        
        # Perform SVD registration to compute transformation
        rmse, transformed_ipc_source, est_T_source_target = self.execute_SVD_registration(ipc_source, ipc_target)
        
        # Extract rotation and translation components
        est_R_source_target = est_T_source_target[:3, :3]
        est_t_source_target = est_T_source_target[:3, 3]
        
        # Convert rotation matrix to quaternion
        est_quaternion = quaternion.from_rotation_matrix(est_R_source_target)
        
        # Update bot_lost flag
        self.bot_lost = False
        
        return self.bot_lost, est_quaternion, rmse, est_t_source_target, est_T_source_target

    def steps_from_source_to_target(self, source_color, source_depth, target_color, target_depth, delta_t=0.1, delta_r=1.0, min_matches=10):
        kp_source, kp_target = self.feature_nav.compute_matches(source_color, target_color)
        if len(kp_source) < min_matches:
            rmse = float('inf')
            steps = float('inf')
            t = np.nan
            R = np.nan
            return rmse, steps, t, R
        
        pc_source_h = self.generate_pc_in_cam_ref_frame(source_depth)
        pc_target_h = self.generate_pc_in_cam_ref_frame(target_depth)
    
        ipc_source = self.get_ipc_from_pc(pc_source_h, kp_source)
        ipc_target = self.get_ipc_from_pc(pc_target_h, kp_target)
    
        # IPC Registration
        rmse, _, est_T_source_target = self.execute_SVD_registration(ipc_source, ipc_target)
        t = est_T_source_target[:3, 3]
        R = est_T_source_target[:3, :3]
    
        # Convert rotation matrix to Euler angles
        _, pitch, _ = self.rotation_matrix_to_euler_angles(R)  # in degrees
        steps_r = abs(int(pitch / delta_r))
        steps_t = abs(int(np.linalg.norm(t) / delta_t))
        
        # Check if translation in the Z-direction is negative
        if t[2] < 0:
            steps = float('inf')
        else:
            steps = steps_r + steps_t
        return rmse, steps, t, R

    # Existing helper methods
    def generate_pc_in_cam_ref_frame(self, depth_img, T_cam_world=None):
        W = H = depth_img.shape[1]  # Assuming square images
        hfov = np.pi / 2
        K = np.array([
            [1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        
        # Generate mesh grid for image coordinates
        xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, H))
        depth = depth_img.reshape(1, H, W)
        xs = xs.reshape(1, H, W)
        ys = ys.reshape(1, H, W)
    
        # Unproject to 3D points
        xys = np.vstack((xs * depth, ys * depth, -depth, np.ones(depth.shape)))
        xys = xys.reshape(4, -1)
        pc_cam_h = np.matmul(np.linalg.inv(K), xys)
    
        return pc_cam_h

    def get_ipc_from_pc(self, pc_cam, kp_cam):
        H, W = 256, 256  # Adjust based on your image dimensions
        cam_key_id = [int(kp[1] * W + kp[0]) for kp in kp_cam]
        ipc_cam_h = pc_cam[:, cam_key_id]
        ipc_cam = ipc_cam_h[:3].T
        return ipc_cam

    def execute_SVD_registration(self, source_pc, target_pc):
        # Compute centroids
        centroid_source = np.mean(source_pc, axis=0)
        centroid_target = np.mean(target_pc, axis=0)
        
        # Center the point clouds
        source_centered = source_pc - centroid_source
        target_centered = target_pc - centroid_target
        
        # Cross-covariance matrix
        H = np.dot(source_centered.T, target_centered)
        
        # Singular Value Decomposition
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        # Ensure a proper rotation matrix (determinant = 1)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        # Compute translation
        t = centroid_target - np.dot(R, centroid_source)
        
        # Homogeneous transformation matrix
        T = np.identity(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        # Transform the source point cloud
        transformed_source_pc = self.transform_point_cloud(source_pc, T)
        
        # Compute RMSE
        distances = np.linalg.norm(transformed_source_pc - target_pc, axis=1)
        rmse = np.sqrt(np.mean(distances**2))
        
        return rmse, transformed_source_pc, T

    def transform_point_cloud(self, point_cloud, transformation_matrix):
        # Add a ones column for homogeneous coordinates
        homogeneous_points = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
        # Apply the transformation
        return np.dot(homogeneous_points, transformation_matrix.T)[:, :3]

    def rotation_matrix_to_euler_angles(self, R):
        # Convert rotation matrix to Euler angles (in degrees)
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        # Convert from radians to degrees
        x = np.degrees(x)
        y = np.degrees(y)
        z = np.degrees(z)
        return x, y, z

    def nav_eval(self, t, R, frontal_threshold=0.0, lateral_threshold=0.5, rotation_threshold=0.2):
        navigability=True
        #print("Type of t:", type(t))
        #print("Value of t:", t)
        if np.isnan(t).any():
            navigability=False
            return navigability
        if t[2] < frontal_threshold:
            navigability=False
            #return navigability
        lateral_displacement = np.sqrt(t[0]**2 + t[1]**2)
        rotation_angle = np.arccos((np.trace(R) - 1) / 2)
        if (lateral_displacement / t[2]) > lateral_threshold and abs(rotation_angle) < rotation_threshold:
            navigability=False
        return navigability