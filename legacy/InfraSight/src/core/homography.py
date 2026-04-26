import numpy as np
import cv2

class HomographyEngine:
    """
    Converts 2D pixel masks into 3D ground-plane physical areas (cm^2)
    using Camera Intrinsics and Extrinsics, assuming a flat road surface.
    """
    def __init__(self, height_cm, pitch_deg, focal_length_px=700, img_w=640, img_h=640):
        self.h = height_cm
        self.pitch = np.radians(pitch_deg)
        
        # Intrinsic Matrix (K)
        self.K = np.array([
            [focal_length_px, 0, img_w / 2],
            [0, focal_length_px, img_h / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        # Extrinsic Rotation Matrix (Pitching downwards)
        self.R = np.array([
            [1, 0, 0],
            [0, np.cos(self.pitch), -np.sin(self.pitch)],
            [0, np.sin(self.pitch), np.cos(self.pitch)]
        ], dtype=np.float32)

        # Translation Vector (Camera height on the Z/Y axis depending on convention)
        self.t = np.array([[0], [self.h], [0]], dtype=np.float32)

        self.H = self._compute_homography()
        self.H_inv = np.linalg.inv(self.H)

    def _compute_homography(self):
        """
        Calculates the Homography matrix H that maps ground plane coordinates (Z=0)
        to image plane pixels.
        """
        # H = K * [r1, r2, t] where r1, r2 are the first two columns of R
        Rt = np.column_stack((self.R[:, 0], self.R[:, 1], self.t))
        H = self.K @ Rt
        return H

    def calculate_physical_area(self, mask: np.ndarray) -> float:
        """
        Projects the boundary pixels of the YOLO mask onto the physical ground plane
        and calculates the area in cm^2.
        """
        # Find the contours (jagged edges) of your YOLO mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0

        largest_contour = max(contours, key=cv2.contourArea)
        
        physical_points = []
        for point in largest_contour:
            u, v = point[0]
            
            # Convert pixel (u,v) to homogeneous coordinate
            p_img = np.array([[u], [v], [1]], dtype=np.float32)
            
            # Project to ground plane: p_ground = H_inv * p_img
            p_ground = self.H_inv @ p_img
            
            # Normalize by the 3rd coordinate (scale factor)
            X = p_ground[0, 0] / p_ground[2, 0]
            Y = p_ground[1, 0] / p_ground[2, 0]
            
            physical_points.append([X, Y])

        physical_points = np.array(physical_points, dtype=np.float32)
        
        # Calculate physical area using OpenCV contour area on the unwarped points
        area_cm2 = cv2.contourArea(physical_points)
        
        return float(area_cm2)