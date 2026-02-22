import numpy as np
from skan import Skeleton, summarize
import networkx as nx
import cv2
import matplotlib.pyplot as plt


class ShortestPathSolver:
    def __init__(self):
        self.skeleton = None
        self.graph = None
        self.df = None

    def convert(self, matrix):
        """
        Translates the skeleton matrix into a weighted network graph.
        """
        self.skeleton = Skeleton(matrix)
        self.df = summarize(self.skeleton, separator='-')
        self.graph = nx.Graph()

        # שינוי קריטי 1: שימוש ב-idx כדי לשלוף את נתוני העיקול מ-skan
        for idx, row in self.df.iterrows():
            src_idx = int(row['node-id-src'])
            dst_idx = int(row['node-id-dst'])

            # משיכת כל הפיקסלים שמרכיבים את הכביש בין שני הצמתים
            branch_coords = self.skeleton.path_coordinates(idx)

            self.graph.add_edge(
                src_idx,
                dst_idx,
                weight=row['branch-distance'],
                coords=branch_coords  # שמירת הפיקסלים בתוך הקשת בגרף
            )

            self.graph.nodes[src_idx]['pos'] = self.skeleton.coordinates[src_idx]
            self.graph.nodes[dst_idx]['pos'] = self.skeleton.coordinates[dst_idx]

        return self.graph

    def block_road(self, barrier_coord, block_radius=5):
        if self.graph is None or barrier_coord is None:
            return

        # Support passing a single (y,x) or an iterable of (y,x) barrier coordinates
        pts = []
        try:
            # If it's an iterable of iterables (e.g., list of [y,x]), treat as multiple
            if isinstance(barrier_coord, (list, tuple, np.ndarray)) and len(barrier_coord) > 0 and isinstance(barrier_coord[0], (list, tuple, np.ndarray)):
                for b in barrier_coord:
                    pts.append(np.array(b, dtype=float))
            else:
                pts.append(np.array(barrier_coord, dtype=float))
        except Exception:
            print(f"Warning: malformed barrier coordinate(s) {barrier_coord}, skipping")
            return

        nodes_to_remove = set()
        edges_to_remove = set()

        for p_coords in pts:
            if p_coords.size != 2:
                print(f"Warning: barrier coordinate must be (y,x), got {p_coords}, skipping")
                continue

            # Collect nodes to remove
            for n, data in list(self.graph.nodes(data=True)):
                pos = data.get('pos', None)
                if pos is None:
                    continue
                pos_arr = np.array(pos, dtype=float)
                try:
                    if np.linalg.norm(p_coords - pos_arr) <= block_radius:
                        nodes_to_remove.add(n)
                except Exception:
                    # skip malformed positions
                    continue

            # Collect edges to remove
            for u, v, data in list(self.graph.edges(data=True)):
                try:
                    pos_u = np.array(self.graph.nodes[u]['pos'], dtype=float)
                    pos_v = np.array(self.graph.nodes[v]['pos'], dtype=float)
                except Exception:
                    continue
                try:
                    dist = self.point_to_line_dist(p_coords, pos_u, pos_v)
                    if dist <= block_radius:
                        edges_to_remove.add((u, v))
                except Exception:
                    continue

        # Remove edges first, then nodes
        for u, v in edges_to_remove:
            if self.graph.has_edge(u, v):
                try:
                    self.graph.remove_edge(u, v)
                except Exception:
                    pass

        for n in list(nodes_to_remove):
            if self.graph.has_node(n):
                try:
                    self.graph.remove_node(n)
                except Exception:
                    pass

    def find_shortest_path(self, start_coords, end_coords):
        if self.graph is None:
            return None

        temp_graph = self.graph.copy()

        start_node = self.add_temporary_node(temp_graph, np.array(start_coords), "start")
        end_node = self.add_temporary_node(temp_graph, np.array(end_coords), "end")

        try:
            path = nx.dijkstra_path(temp_graph, start_node, end_node, weight='weight')

            full_path_coords = []

            # שינוי קריטי 2: פריסת המסלול המלא כולל כל העיקולים
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]

                # בדיקה האם שמרנו את העיקולים של הקשת הזו
                if 'coords' in temp_graph[u][v]:
                    edge_coords = temp_graph[u][v]['coords']
                    pos_u = temp_graph.nodes[u]['pos']

                    # מוודאים שאנחנו "מציירים" בכיוון הנכון (מ-u ל-v) כדי למנוע זגזוגים
                    if np.linalg.norm(edge_coords[-1] - pos_u) < np.linalg.norm(edge_coords[0] - pos_u):
                        edge_coords = edge_coords[::-1]

                    full_path_coords.extend(edge_coords)
                else:
                    # אם זו קשת זמנית (מההתחלה/סיום לכביש הראשי), נשתמש בקו ישר
                    full_path_coords.append(temp_graph.nodes[u]['pos'])
                    full_path_coords.append(temp_graph.nodes[v]['pos'])

            return full_path_coords

        except nx.NetworkXNoPath:
            return None

    def add_temporary_node(self, graph, p_coords, label):
        best_edge = None
        min_dist = float('inf')
        new_node_id = f"temp_{label}"

        # מציאת הכביש הקרוב ביותר לנקודה
        for u, v, data in graph.edges(data=True):
            pos_u = graph.nodes[u]['pos']
            pos_v = graph.nodes[v]['pos']
            dist = self.point_to_line_dist(p_coords, pos_u, pos_v)
            if dist < min_dist:
                min_dist = dist
                best_edge = (u, v, data['weight'])

        if best_edge:
            u, v, weight = best_edge

            # 1. שומרים את הפיקסלים של הכביש המעוקל המקורי לפני שחותכים אותו
            orig_coords = graph[u][v].get('coords', None)

            if graph.has_edge(u, v):
                graph.remove_edge(u, v)

            graph.add_node(new_node_id, pos=p_coords)

            pos_u = graph.nodes[u]['pos']
            pos_v = graph.nodes[v]['pos']
            dist_u = np.linalg.norm(p_coords - pos_u)
            dist_v = np.linalg.norm(p_coords - pos_v)
            total = dist_u + dist_v
            ratio = dist_u / total if total > 0 else 0.5

            # 2. אם יש לכביש המקורי פיקסלים של עיקול, אנחנו מפצלים גם אותם לשני החלקים
            if orig_coords is not None and len(orig_coords) > 0:
                # מוודאים שהפיקסלים מסודרים מהצד של u לכיוון v
                if np.linalg.norm(orig_coords[-1] - pos_u) < np.linalg.norm(orig_coords[0] - pos_u):
                    orig_coords = orig_coords[::-1]

                # מוצאים את הפיקסל הכי קרוב לנקודת ההתחלה/סיום החדשה שלנו
                distances = np.linalg.norm(orig_coords - p_coords, axis=1)
                split_idx = np.argmin(distances)

                # מחלקים את רשימת הפיקסלים המעוקלים לשניים
                coords_u_to_new = orig_coords[:split_idx + 1]
                coords_new_to_v = orig_coords[split_idx:]

                # מוסיפים את המקטעים החדשים כולל מידע העיקול שלהם
                graph.add_edge(u, new_node_id, weight=ratio * weight, coords=coords_u_to_new)
                graph.add_edge(new_node_id, v, weight=(1 - ratio) * weight, coords=coords_new_to_v)
            else:
                # למקרה החריג שאין עיקול שמור
                graph.add_edge(u, new_node_id, weight=ratio * weight)
                graph.add_edge(new_node_id, v, weight=(1 - ratio) * weight)

        return new_node_id
    def point_to_line_dist(self, p, a, b):
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        p = np.array(p, dtype=float)
        if np.array_equal(a, b):
            return np.linalg.norm(p - a)
        l2 = np.sum((a - b) ** 2)
        t = max(0, min(1, np.dot(p - a, b - a) / l2))
        projection = a + t * (b - a)
        return np.linalg.norm(p - projection)

def get_satellite_navigation_map(satellite_img, skeleton_matrix, start, end, barriers=None, block_radius=5):
    """
    Computes a path avoiding barriers and draws it on the satellite image.

    Args:
        satellite_img: numpy array (H,W,3) or (H,W) original image (RGB or BGR as read by cv2)
        skeleton_matrix: binary or grayscale skeleton image (0/255 or 0/1)
        start, end: (y, x) tuples in image coordinates
        barriers: iterable of (y, x) coords to mark as blocked
        block_radius: radius in pixels for blocking

    Returns:
        output_image: colored image (BGR) with the path drawn, or None if no path found.
    """
    # If satellite_img is a PIL Image, convert to numpy array (RGB -> BGR for cv2)
    try:
        from PIL import Image as PilImage
        if isinstance(satellite_img, PilImage.Image):
            satellite_img = np.array(satellite_img)
            # Convert RGB (PIL) to BGR for OpenCV drawing
            try:
                satellite_img = cv2.cvtColor(satellite_img, cv2.COLOR_RGB2BGR)
            except Exception:
                pass
    except Exception:
        # PIL not available or conversion failed; proceed and let later checks handle types
        pass

    # Normalize skeleton to binary 0/1
    if skeleton_matrix is None:
        return None
    skeleton_copy = skeleton_matrix.copy()
    # If barriers provided, zero a small disk around each barrier location to robustly cut the skeleton.
    if barriers:
        # Ensure we have a uint8 image for cv2 drawing (0/255)
        try:
            sk_tmp = (skeleton_copy > 0).astype('uint8') * 255
        except Exception:
            sk_tmp = np.array(skeleton_copy, dtype='uint8')

        for b in (barriers or []):
            try:
                y = int(b[0]); x = int(b[1])
            except Exception:
                continue
            if 0 <= y < sk_tmp.shape[0] and 0 <= x < sk_tmp.shape[1]:
                # draw filled circle (note cv2 expects (x,y) center)
                try:
                    cv2.circle(sk_tmp, (x, y), block_radius, 0, thickness=-1)
                except Exception:
                    # fallback: clear single pixel
                    sk_tmp[y, x] = 0

        # Convert back to original-like skeleton copy
        skeleton_copy = sk_tmp

    sk = np.array(skeleton_copy)
    if sk.dtype != np.uint8:
        sk = (sk > 0).astype(np.uint8)
    else:
        # Accept 0/255 or 0/1
        sk = (sk > 0).astype(np.uint8)

    # If start/end are out of bounds, return None early
    H, W = sk.shape[:2]
    sy, sx = start
    ey, ex = end
    if not (0 <= sy < H and 0 <= sx < W and 0 <= ey < H and 0 <= ex < W):
        print('Start/end out of bounds')
        return None

    # Build solver and graph
    solver = ShortestPathSolver()
    solver.convert(sk)

    # Apply barriers
    # Apply barriers to the graph level once (solver.block_road accepts iterables)
    if barriers:
        try:
            solver.block_road(barriers, block_radius=block_radius)
        except Exception:
            # fallback: try per-barrier removal
            for b in barriers:
                try:
                    solver.block_road(b, block_radius=block_radius)
                except Exception:
                    pass

    # Calculate path
    path_coords = solver.find_shortest_path(start, end)
    if path_coords is None:
        print('No path found after applying barriers')
        return None

    # Prepare colored image for drawing - convert grayscale RGB/GRAY to BGR if needed
    output_image = None
    if len(satellite_img.shape) == 2:
        output_image = cv2.cvtColor(satellite_img, cv2.COLOR_GRAY2BGR)
    else:
        # If image is PIL-like (RGB), convert to BGR for cv2 drawing
        if satellite_img.shape[2] == 3:
            # Heuristic: image may be RGB (e.g., PIL). Convert to BGR
            output_image = satellite_img.copy()
            try:
                # If colors look like RGB (matplotlib/PIL), convert to BGR
                output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
            except Exception:
                # If conversion fails, just copy
                output_image = satellite_img.copy()
        else:
            output_image = satellite_img.copy()

    # Draw path segments as a thick red line on the satellite image
    for i in range(len(path_coords) - 1):
        # Convert (y, x) to (x, y) for OpenCV plotting
        pt1 = (int(path_coords[i][1]), int(path_coords[i][0]))
        pt2 = (int(path_coords[i + 1][1]), int(path_coords[i + 1][0]))
        cv2.line(output_image, pt1, pt2, (0, 0, 255), 3)

    # Return both the annotated image and the raw path coordinates (list of (y,x))
    return output_image, path_coords


if __name__ == "__main__":
    # Minimal main: read a skeleton and an original image, run the output function and show the result
    # Edit these paths to test different files
    skeleton_path = 'output/81c6b838568d45ea95f5c5d2589f778f_demo_satellite.tiff'
    orig_path = 'assets/demo_satellite.tiff'

    # Read skeleton (grayscale) and original (color)
    skeleton_matrix = cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE)
    satellite_img = cv2.imread(orig_path)

    if skeleton_matrix is None:
        print(f'Error: could not load skeleton from {skeleton_path}')

    if satellite_img is None:
        print(f'Error: could not load original image from {orig_path}')

    # Example start/end and barriers (y,x)
    p1 = (476, 636)
    p2 = (666, 478)
    block_points = []

    annotated = get_satellite_navigation_map(satellite_img, skeleton_matrix, p1, p2, block_points, block_radius=5)
    print('DEBUG: final orig_image shape after output =', None if annotated is None else getattr(annotated, 'shape', None))

    if annotated is not None:
        # Convert BGR to RGB for matplotlib
        try:
            disp = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        except Exception:
            disp = annotated
        plt.figure(figsize=(10, 10))
        plt.imshow(disp)
        plt.axis('off')
        plt.show()
    else:
        print('No annotated image to display.')
