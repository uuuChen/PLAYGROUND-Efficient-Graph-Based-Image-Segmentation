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
parser.add_argument('--seg-by-rgb', action="store_true", default=False)
args = parser.parse_args()

# set upper bound of limit of recursive
sys.setrecursionlimit(int(1e6))


class TreeNode:
    def __init__(self, index2d, belong_tree=None):
        self.index2d = index2d
        self.neighbors = dict()  # key: value = (node, edge_weight)
        self.belong_tree = belong_tree


class MinimumSpanningTree:
    def __init__(self, root_node, k):
        self.Int = 0
        self.k = k
        self.root_node = root_node
        self.num_of_nodes = 1
        self.root_node.belong_tree = self

    def iteration(self, node, tree_nodes, nodes_neighbors=None, seen_nodes=None):
        if seen_nodes is None:
            seen_nodes = set()
        tree_nodes.append(node)
        seen_nodes.add(node)
        for neighbor_node in node.neighbors.keys():
            if neighbor_node not in seen_nodes:
                if nodes_neighbors is not None:
                    nodes_neighbors.append((node.index2d, neighbor_node.index2d))
                self.iteration(neighbor_node, tree_nodes, nodes_neighbors, seen_nodes)

    def join(self, a_node, b_node, edge_weight):
        a_node.neighbors[b_node] = edge_weight
        b_node.neighbors[a_node] = edge_weight
        b_tree = b_node.belong_tree
        b_tree_nodes = list()
        b_tree.iteration(b_tree.root_node, b_tree_nodes)
        for node in b_tree_nodes:
            node.belong_tree = self
        self.Int = max(self.Int, edge_weight)
        self.num_of_nodes += b_tree.num_of_nodes

    def inner_threshold(self):
        return self.Int + (self.k / self.num_of_nodes)

    def remove_edge(self, a_node, b_node):
        del a_node.neighbors[b_node]
        del b_node.neighbors[a_node]
        if not a_node.neighbors:
            a_node.belong_tree = None
        if not b_node.neighbors:
            b_node.belong_tree = None


def load_image_pixels():
    pixels_of_image = np.array(Image.open(args.image_file_path, 'r').convert('RGB'))
    return pixels_of_image


class GraphBasedImageSegment:
    def __init__(self):
        pass

    def _build_graph_with_pixels(self, image_pixels, k):
        graph = dict()
        index2node = dict()
        indices2edgeDist = dict()
        MSTrees = set()
        height, width, channels = image_pixels.shape

        # get nodes and minimum spanning trees
        for i in range(height):
            for j in range(width):
                index = (i, j)
                node = TreeNode(index)
                index2node[index] = node
                MSTrees.add(MinimumSpanningTree(node, k))
                cur_index, cur_pixel = (i, j), image_pixels[i, j]
                if i != height - 1:  # not the last row, to lower pixel
                    to_index, to_pixel = (i+1, j), image_pixels[i+1, j]
                    indices2edgeDist[(cur_index, to_index)] = self.euclidean_distance(cur_pixel, to_pixel)
                if j != width - 1:  # not the last col, to right pixel
                    to_index, to_pixel = (i, j+1), image_pixels[i, j+1]
                    indices2edgeDist[(cur_index, to_index)] = self.euclidean_distance(cur_pixel, to_pixel)
                if i != height - 1 and j != width - 1:  # not the last row or col, to lower right pixel
                    to_index, to_pixel = (i+1, j+1), image_pixels[i+1, j+1]
                    indices2edgeDist[(cur_index, to_index)] = self.euclidean_distance(cur_pixel, to_pixel)
                if i != height - 1 and j != 0:  # not the last  or the first col, to lower left pixel
                    to_index, to_pixel = (i+1, j-1), image_pixels[i+1, j-1]
                    indices2edgeDist[(cur_index, to_index)] = self.euclidean_distance(cur_pixel, to_pixel)

        # use value to sort edges from small to large
        indices2edgeDist = dict(sorted(indices2edgeDist.items(), key=lambda x: x[1]))

        # set graph item
        graph['index2node'] = index2node
        graph['indices2edgeDist'] = indices2edgeDist
        graph['MSTrees'] = MSTrees

        return graph

    def _merge_trees(self, graph):
        for (a_index, b_index), edge_weight in graph['indices2edgeDist'].items():
            a_node, b_node = graph['index2node'][a_index], graph['index2node'][b_index]
            a_MSTree, b_MSTree = a_node.belong_tree, b_node.belong_tree
            if a_MSTree == b_MSTree:
                continue
            # if true, merge b_tree to a_tree, and remove b_tree from set
            if edge_weight <= min(a_MSTree.inner_threshold(), b_MSTree.inner_threshold()):
                a_MSTree.join(a_node, b_node, edge_weight)
                graph['MSTrees'].remove(b_MSTree)
        return graph

    def _show_segmented_image(self, image_pixels, k, graphs, seg_by_rgb=False):
        print(f"k: {k}\tcomponents: {len(graphs[0]['MSTrees'])}\t")
        rand_colors = [self.rand_rgb() for _ in range(len(graphs[0]['MSTrees']))]
        seg_image_pixels = np.zeros((image_pixels.shape[0], image_pixels.shape[1], 3))  # generate rgb figure
        for i, tree in enumerate(graphs[0]['MSTrees']):
            # get tree nodes and tree root from first graph
            tree_nodes = list()
            nodes_neighbors = list()
            tree.iteration(tree.root_node, tree_nodes, nodes_neighbors=nodes_neighbors)

            # if seg_by_rgb is used, intersect tree nodes of the three graphs
            if seg_by_rgb:
                tree_nodes_indices = set()
                for from_index, to_index in nodes_neighbors:
                    r_from_node = graphs[0]['index2node'][from_index]
                    g_from_node = graphs[1]['index2node'][from_index]
                    b_from_node = graphs[2]['index2node'][from_index]

                    r_to_node = graphs[0]['index2node'][to_index]
                    g_to_node = graphs[1]['index2node'][to_index]
                    b_to_node = graphs[2]['index2node'][to_index]

                    if g_to_node in g_from_node.neighbors.keys() and b_to_node in b_from_node.neighbors.keys():
                        tree_nodes_indices.add(from_index)
                        tree_nodes_indices.add(to_index)
                    else:  # remove edge from r_tree
                        r_tree = r_from_node.belong_tree
                        r_tree.remove_edge(r_from_node, r_to_node)
            else:
                tree_nodes_indices = set([node.index2d for node in tree_nodes])

            # assign colors to pixels
            for index in tree_nodes_indices:
                seg_image_pixels[index] = rand_colors[i]

        ori_image = Image.fromarray(np.uint8(image_pixels))
        seg_image = Image.fromarray(np.uint8(seg_image_pixels))
        ori_image.show()
        seg_image.show()

    @ staticmethod
    def rand_rgb():
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return r, g, b

    def segment(self, image_pixels, k, seg_by_rgb=False):
        if not seg_by_rgb:
            graph = self._build_graph_with_pixels(image_pixels, k)
            graph = self._merge_trees(graph)
            self._show_segmented_image(image_pixels, k, [graph], seg_by_rgb=seg_by_rgb)
        else:
            graphs = list()
            for ch in range(image_pixels.shape[2]):  # run num_of_channels times
                channel_pixels = np.expand_dims(image_pixels[:, :, ch], axis=2)
                graph = self._build_graph_with_pixels(channel_pixels, k)
                graph = self._merge_trees(graph)
                graphs.append(graph)
            self._show_segmented_image(image_pixels, k, graphs, seg_by_rgb=seg_by_rgb)

    @ staticmethod
    def euclidean_distance(a_pixel, b_pixel):
        return np.sqrt(np.sum(np.power(a_pixel - b_pixel, 2)))

    @ staticmethod
    def sort_pair_indices(a_index, b_index):
        return sorted([a_index, b_index], key=lambda x: (x[0], x[1]))


def main():
    image_pixels = load_image_pixels()
    GBIS = GraphBasedImageSegment()
    GBIS.segment(image_pixels, args.k, args.seg_by_rgb)


if __name__ == '__main__':
    main()
