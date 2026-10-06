import numpy as np
import cv2


def project_trajectory_on_image(image_bgr, local_abs_poses, K, T_cam_base):
    """
    Projects a sequence of 2D local base_link poses onto the 2D camera image plane.
    """
    img_out = image_bgr.copy()
    h, w = img_out.shape[:2]
    
    pts_2d = []
    for pose in local_abs_poses:
        # Homogeneous point in base_link (assuming robot moves on the Z=0 ground plane)
        p_base = np.array([pose[0], pose[1], 0.0, 1.0])
        p_cam = T_cam_base @ p_base
        
        # Gotcha Fix: Z-Clipping. Discard points behind or too close to the camera lens
        if p_cam[2] > 0.01:
            uv_w = K @ p_cam[:3]
            u, v = int(uv_w[0] / uv_w[2]), int(uv_w[1] / uv_w[2])
            
            # Only draw if the point lands inside the physical image sensor bounds
            if 0 <= u < w and 0 <= v < h:
                pts_2d.append((u, v))
                
    # Draw the trajectory
    for i in range(len(pts_2d)):
        cv2.circle(img_out, pts_2d[i], radius=6, color=(0, 255, 0), thickness=-1)
        if i > 0:
            cv2.line(img_out, pts_2d[i-1], pts_2d[i], color=(0, 200, 255), thickness=3)
            
    return img_out