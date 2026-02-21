import cv2
from collections import deque
import skimage
import matplotlib.pyplot as plt
import os
from scipy.spatial import KDTree

from src.model.inference import load_model, predict_large_image
from PIL import Image
import skimage.morphology
import numpy as np
import ShowShortestPath
MODEL_PATH = "data/weights/unet.pth"
IMAGE_PATH = "assets/demo_satellite.tiff"
new_image=True
# Cache for the last-processed image/skeleton so subsequent re-annotations can reuse it
SKELETON_CACHE = {}

def convert_img_to_binary(image):
    # Placeholder for image processing logic
    # In a real implementation, this would convert the image to binary format
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_scaled = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_scaled = image

    _, binary = cv2.threshold(gray_scaled, 127, 255, cv2.THRESH_BINARY)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_v)

    return binary


def Remove_small_components(binary_image, min_size):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    new_image = np.zeros_like(binary_image)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            new_image[labels == i] = 1

    return new_image

def get_components(binary, start, visited):
    q=deque([start])
    components = []
    visited[start] = True
    while q:
        x, y = q.popleft()
        components.append((x, y))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < binary.shape[0] and 0 <= ny < binary.shape[1] and
                    not visited[nx, ny] and binary[ny,nx]):
                    visited[ny,nx] = True
                    q.append((ny,nx))
    return components

def skeletonize_image(binary_image):
    # Placeholder for skeletonization logic
    # In a real implementation, this would skeletonize the binary image
    skeleton = skimage.morphology.skeletonize(binary_image)

    return skeleton





def find_endpoints(skeleton):
    # Placeholder for endpoint detection logic
    # In a real implementation, this would find endpoints in the skeletonized image
    endpoints = []
    h, w = skeleton.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            if skeleton[i, j]:
                neighbors = skeleton[i-1:i+2, j-1:j+2]
                if np.sum(neighbors) == 2:  # Endpoint condition
                    endpoints.append((i, j))
    return np.array(endpoints)

def find_closest_points(endpoints, epsilon):
    # Placeholder for closest point detection logic
    # In a real implementation, this would find the closest points on the skeleton to the endpoints
    tree = KDTree(endpoints)
    pairs= tree.query_pairs(epsilon)
    return pairs


def get_direction(skeleton, point, step_size):
    y, x = point
    current = (y, x)
    path = [current]
    prev = None

    for _ in range(step_size):
        curr_y, curr_x = current

        neighbors = []

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue

                ny, nx = curr_y + dy, curr_x + dx

                if (0 <= ny < skeleton.shape[0] and
                        0 <= nx < skeleton.shape[1] and
                        skeleton[ny, nx]):

                    if (ny, nx) != prev:
                        neighbors.append((ny, nx))

        if not neighbors:
            break

        prev = current
        current = neighbors[0]
        path.append(current)

    if len(path) >= 2:
        y0, x0 = path[0]
        y1, x1 = path[-1]

        v = np.array([y1 - y0, x1 - x0], dtype=float)
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm

    return None
def connect_points(skel, p1, p2):
    rr, cc = skimage.draw.line(p1[0], p1[1], p2[0], p2[1])
    skel[rr, cc] = 1




def process_image(image, min_size=40, epsilon=30, step_size=3):

    binary_image = convert_img_to_binary(image)

    binary_image=Remove_small_components(binary_image, min_size)

    skeleton = skeletonize_image(binary_image)

    endpoints = find_endpoints(skeleton)

    pairs = find_closest_points(endpoints, epsilon)

    for p1_idx, p2_idx in pairs:
        pt1 = endpoints[p1_idx]
        pt2 = endpoints[p2_idx]
        dir1 = get_direction(skeleton, endpoints[p1_idx], step_size)
        dir2 = get_direction(skeleton, endpoints[p2_idx], step_size)

        if dir1 is None or dir2 is None:
            continue

        cos_similarity = np.dot(dir1, dir2)
        if cos_similarity < -0.1 :  # Threshold for similarity
            connect_points(skeleton, pt1, pt2)

    cv2.imshow('image', image)
    os.makedirs('output', exist_ok=True)
    #cv2.imwrite(os.path.join('output', os.path.basename(image_path)), skeleton.astype(np.uint8) * 255)

    return skeleton

def road_route_extraction(image_path,point1,point2,barriers=None, model_path="data/weights/unet.pth" ):
    print("point1",point1)
    print("point2",point2)

    model = load_model(model_path)
    org_image = Image.open(image_path).convert("RGB")
    mask = predict_large_image(model, org_image, patch_size=512)
    mask_uint8 = (mask * 255).astype(np.uint8)

    skeleton = process_image(mask_uint8)

    # Cache skeleton and original image for this image_path so future re-annotations can reuse it
    SKELETON_CACHE[image_path] = (skeleton, org_image)

    point1 = find_closest_skeleton_point(skeleton, point1)
    point2 = find_closest_skeleton_point(skeleton, point2)
    print("new point1", point1)
    print("new point2", point2)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(org_image)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(skeleton, cmap="gray")
    plt.title("Skeleton Output")

    plt.show()

    annotated = ShowShortestPath.get_satellite_navigation_map(org_image, skeleton, point1, point2, barriers)

    return annotated


def road_route_extraction_from_skeleton(point1, point2, barriers=None, image_path=None):
    """
    Use the last cached skeleton and original image to compute the annotated shortest path
    without running the model again.

    Args:
        point1, point2: (y, x) tuples (row, col) in image coordinates
        barriers: iterable of (y, x) tuples to block

    Returns:
        Annotated image (numpy array / PIL image) or None if no cached skeleton available.
    """
    # If image_path is provided, try to use its cached skeleton; otherwise fall back to any available cache
    if image_path:
        cached = SKELETON_CACHE.get(image_path)
        if cached is None:
            raise RuntimeError(f"No cached skeleton for image_path={image_path}. Run full extraction first.")
        last_skeleton, last_org_image = cached
    else:
        # if no specific image_path asked, take any cached entry (best-effort)
        if not SKELETON_CACHE:
            raise RuntimeError("No cached skeleton available. Run road_route_extraction first or provide an image_path.")
        # pick the most recently added cache entry
        last_image_path = next(reversed(SKELETON_CACHE))
        last_skeleton, last_org_image = SKELETON_CACHE[last_image_path]

    # Find closest skeleton points to the requested points
    p1 = find_closest_skeleton_point(last_skeleton, point1)
    p2 = find_closest_skeleton_point(last_skeleton, point2)

    # Delegate to the visualization/solver using the cached skeleton and original image
    annotated = ShowShortestPath.get_satellite_navigation_map(last_org_image, last_skeleton, p1, p2, barriers)
    return annotated

def find_closest_skeleton_point(skeleton, point):
    y, x = point
    skeleton_points = np.argwhere(skeleton)
    tree = KDTree(skeleton_points)
    _, idx = tree.query([y, x])
    closest_point = skeleton_points[idx]
    return (int(closest_point[0]), int(closest_point[1]))

if __name__ == "__main__":
    newpoint1=(643, 414)
    newpoint2=(965, 360)
    skeleton_path= "output/result_81c6b838568d45ea95f5c5d2589f778f_demo_satellite.png"
    skeleton_image = cv2.imread("output/0d5961a819634be996f4161f15806aa5_demo_satellite.png", cv2.IMREAD_GRAYSCALE)
    #print shape
    print("Skeleton shape:", skeleton_image.shape)
    print("point1",skeleton_image[newpoint1[0], newpoint1[1]])
    print("point2",skeleton_image[newpoint2[0], newpoint2[1]])

    image_path = 'assets/demo_satellite.tiff'
    point1=(676, 396)
    point2=(1068, 389)
    annotated = road_route_extraction(image_path, point1, point2, model_path=MODEL_PATH)
    plt.imshow(annotated)
    plt.title("Annotated Shortest Path")
    plt.axis('off')
    plt.show()
