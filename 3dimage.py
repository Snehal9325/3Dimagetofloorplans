import cv2
import numpy as np
import open3d as o3d

def preprocess_image(image_path):
   
    image = cv2.imread(image_path)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def reconstruct_3d(depth_image):
    height, width = depth_image.shape
    fx = fy = 1  
    cx, cy = width // 2, height // 2  
    
    points = []
    for v in range(height):
        for u in range(width):
            z = depth_image[v, u]
            if z == 0: continue  
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])

    point_cloud = np.array(points)
    return point_cloud

def extract_floor_plan(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.05)

    points = np.asarray(voxel_down_pcd.points)
    floor_plan = points[:, :2]

    return floor_plan

def visualize_floor_plan(floor_plan):
    import matplotlib.pyplot as plt

    plt.scatter(floor_plan[:, 0], floor_plan[:, 1], s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Floor Plan')
    plt.show()

def main():
    image_path =r'C:\Users\91932\Pictures\enterprise-AI.jpg'  
    preprocessed_image = preprocess_image(image_path)

    depth_image = preprocessed_image  
    point_cloud = reconstruct_3d(depth_image)

    floor_plan = extract_floor_plan(point_cloud)

    visualize_floor_plan(floor_plan)

if __name__ == "__main__":
    main()
