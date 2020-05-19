# System
import argparse
import random
import sys
import numpy as np
from PIL import Image


# get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image-file-path', "-ifp", type=str, default='./images/sample.png')
parser.add_argument('--k', type=int, default=500)
args = parser.parse_args()

# set upper bound of limit of recursive
sys.setrecursionlimit(int(1e6))


class TreeNode:
    def __init__(self, index2d, belong_tree=None):
        self.index2d = index2d
        self.neighbors = list()  # [(neighbor_1_node, neighbor_1_edge_weight)...]
        self.belong_tree = belong_tree


class MinimumSpanningTree:
    def __init__(self, root_node, k):
        self.Int = 0
        self.k = k
        self.root_node = root_node
        self.num_of_nodes = 1
        self.root_node.belong_tree = self

    def iteration(self, node, tree_nodes, seen_nodes=None):
        if seen_nodes is None:
            seen_nodes = set()
        tree_nodes.append(node)
        seen_nodes.add(node)
        for neighbor_node, _ in node.neighbors:
            if neighbor_node not in seen_nodes:
                self.iteration(neighbor_node, tree_nodes, seen_nodes)

    def join(self, a_node, b_node, edge_weight):
        a_node.neighbors.append((b_node, edge_weight))
        b_node.neighbors.append((a_node, edge_weight))
        b_tree = b_node.belong_tree
        b_tree_nodes = list()
        b_tree.iteration(b_tree.root_node, b_tree_nodes)
        for node in b_tree_nodes:
            node.belong_tree = self
        self.Int = max(self.Int, edge_weight)
        self.num_of_nodes += b_tree.num_of_nodes

    def inner_threshold(self):
        return self.Int + (self.k / self.num_of_nodes)


def load_image_pixels():
    pixels_of_image = np.array(Image.open(args.image_file_path, 'r').convert('RGB'))
    return pixels_of_image


class GraphBasedImageSegment:
    def __init__(self):
        self.index2node = dict()
        self.index2MSTree = dict()
        self.indices2edgeDist = dict()
        self.MSTrees = set()

        self.k = None
        self.image_pixels = None

    def _build_graph_with_pixels(self):
        height, width, channels = self.image_pixels.shape

        # get nodes and minimum spanning trees
        for i in range(height):
            for j in range(width):
                index = (i, j)
                node = TreeNode(index)
                self.index2node[index] = node
                self.MSTrees.add(MinimumSpanningTree(node, self.k))
                cur_index, cur_pixel = (i, j), self.image_pixels[i, j]
                if i != height - 1:  # not the last row, to lower pixel
                    to_index, to_pixel = (i+1, j), self.image_pixels[i+1, j]
                    self.indices2edgeDist[(cur_index, to_index)] = self.euclidean_distance(cur_pixel, to_pixel)
                if j != width - 1:  # not the last col, to right pixel
                    to_index, to_pixel = (i, j+1), self.image_pixels[i, j+1]
                    self.indices2edgeDist[(cur_index, to_index)] = self.euclidean_distance(cur_pixel, to_pixel)
                if i != height - 1 and j != width - 1:  # not the last row or col, to lower right pixel
                    to_index, to_pixel = (i+1, j+1), self.image_pixels[i+1, j+1]
                    self.indices2edgeDist[(cur_index, to_index)] = self.euclidean_distance(cur_pixel, to_pixel)
                if i != height - 1 and j != 0:  # not the last col or the first col, to lower left pixel
                    to_index, to_pixel = (i+1, j-1), self.image_pixels[i+1, j-1]
                    self.indices2edgeDist[(cur_index, to_index)] = self.euclidean_distance(cur_pixel, to_pixel)

        # use value to sort edges from small to large
        self.indices2edgeDist = dict(sorted(self.indices2edgeDist.items(), key=lambda x: x[1]))

    def _merge_trees(self):
        for (a_index, b_index), edge_weight in self.indices2edgeDist.items():
            a_node, b_node = self.index2node[a_index], self.index2node[b_index]
            a_MSTree, b_MSTree = a_node.belong_tree, b_node.belong_tree
            if a_MSTree == b_MSTree:
                continue
            # if true, merge b_tree to a_tree, and remove b_tree from set
            if edge_weight <= min(a_MSTree.inner_threshold(), b_MSTree.inner_threshold()):
                a_MSTree.join(a_node, b_node, edge_weight)
                self.MSTrees.remove(b_MSTree)

    def _show_segment_image(self):
        num_of_segments = len(self.MSTrees)
        rand_colors = [self.rand_rgb() for _ in range(num_of_segments)]
        seg_image_pixels = np.zeros(self.image_pixels.shape)
        print(f'k: {self.k}\tcomponents: {len(self.MSTrees)}\t')
        for i, tree in enumerate(self.MSTrees):
            tree_nodes = list()
            tree.iteration(tree.root_node, tree_nodes)
            tree_nodes_indices = [node.index2d for node in tree_nodes]
            for node_index in tree_nodes_indices:
                seg_image_pixels[node_index] = rand_colors[i]
        ori_image = Image.fromarray(np.uint8(self.image_pixels))
        seg_image = Image.fromarray(np.uint8(seg_image_pixels))
        ori_image.show()
        seg_image.show()

    @ staticmethod
    def rand_rgb():
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return r, g, b

    def segment(self, image_pixels, k):
        self.image_pixels = image_pixels
        self.k = k
        self._build_graph_with_pixels()
        self._merge_trees()
        self._show_segment_image()

    @ staticmethod
    def euclidean_distance(a_pixel, b_pixel):
        return np.sqrt(np.sum(np.power(a_pixel - b_pixel, 2)))

    @ staticmethod
    def sort_pair_indices(a_index, b_index):
        return sorted([a_index, b_index], key=lambda x: (x[0], x[1]))


def main():
    image_pixels = load_image_pixels()
    GBIS = GraphBasedImageSegment()
    GBIS.segment(image_pixels, args.k)


if __name__ == '__main__':
    main()
