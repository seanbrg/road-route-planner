import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
import sknw
import networkx as nx
import matplotlib.pyplot as plt


def build_graph_from_mask(mask, close_size=25, debug=False):
    """
    Takes a binary road mask, closes gaps, skeletonizes it, and returns a NetworkX graph.

    Args:
        mask (numpy array): 2D array (H, W) with 0s and 1s (or 0-255).
        close_size (int): Size of the gap-closing kernel.
                          15 pixels ~= 15 meters in Mass. Roads dataset.
                          Increase this if your roads are still broken.
        debug (bool): If True, plots the intermediate steps.

    Returns:
        G (nx.MultiGraph): The weighted road network graph.
    """

    # 1. Ensure Binary & uint8
    # If mask is float (0.0-1.0), convert to 0-255
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    # Threshold just in case to get crisp 0/255
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 2. The "Thickener" (Morphological Closing)
    # This bridges the shadow gaps.
    # MORPH_CLOSE = Dilate (expand white) -> Erode (shrink back)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    closed_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Optional: A slight extra dilation to smooth jagged edges
    # closed_mask = cv2.dilate(closed_mask, kernel, iterations=1)

    # 3. Skeletonize
    # skimage expects a boolean array (True/False)
    bool_mask = closed_mask > 0
    skeleton = skeletonize(bool_mask)

    # 4. Graph Construction (sknw)
    # iso=False keeps small isolated roads. Set True to remove them.
    G = sknw.build_sknw(skeleton, iso=False)

    # --- Visualization Block ---
    if debug:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(binary, cmap='gray')
        ax[0].set_title("1. Original Mask (Gaps)")

        ax[1].imshow(closed_mask, cmap='gray')
        ax[1].set_title(f"2. Closed Mask (Kernel {close_size})")

        # Overlay skeleton on closed mask
        ax[2].imshow(closed_mask, cmap='gray')

        # Draw edges from the graph to verify connectivity
        for (s, e) in G.edges():
            ps = G[s][e]['pts']
            ax[2].plot(ps[:, 1], ps[:, 0], 'r-', linewidth=1)

        ax[2].set_title("3. Final Skeleton Graph")
        plt.show()

    return G


if __name__ == "__main__":
    # 1. Load the mask (0 for grayscale)
    # cv2 reads this as 0-255 automatically
    test_mask = cv2.imread("../../assets/demo_output.tiff", 0)

    if test_mask is not None:
        print("Testing Skeletonizer...")

        # 2. Build the Graph
        # We assume build_graph_from_mask is defined as discussed previously
        graph = build_graph_from_mask(test_mask, close_size=35, debug=False)
        print(f"Graph built with {len(graph.nodes())} intersections and {len(graph.edges())} roads.")

        # 3. DRAW AND SAVE THE VISUALIZATION
        # We create a new figure specifically to save the red lines
        fig, ax = plt.subplots(figsize=(10, 10))

        # A. Draw the base mask (Background)
        ax.imshow(test_mask, cmap='gray')

        # B. Draw the edges (Roads)
        for (s, e) in graph.edges():
            ps = graph[s][e]['pts']
            ax.plot(ps[:, 1], ps[:, 0], 'r-', linewidth=1)  # Red lines

        # C. Draw the nodes (Intersections)
        nodes = graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        ax.plot(ps[:, 1], ps[:, 0], 'c.', markersize=3)  # Cyan dots

        # D. Clean up and Save
        ax.set_title("Extracted Road Graph")
        ax.axis('off')  # Hide axes for a clean image

        # Save the plot with the graph overlaid
        plt.savefig("../../assets/demo_graph_viz.png", dpi=150, bbox_inches='tight')
        plt.close()  # Close memory

        print("✅ Saved graph visualization to assets/demo_graph_viz.png")

    else:
        print("Error: assets/demo_output.tiff not found.")