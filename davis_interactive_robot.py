##### The codes are from:
# https://github.com/albertomontesg/davis-interactive/blob/master/davisinteractive/metrics/jaccard.py
# https://github.com/albertomontesg/davis-interactive/blob/master/davisinteractive/utils/operations.py
# https://github.com/albertomontesg/davis-interactive/blob/master/davisinteractive/robot/interactive_robot.py
#######################

import logging
import time
import networkx as nx
import numpy as np

from scipy.special import comb
from skimage.filters import rank
from skimage.morphology import dilation, disk, erosion, medial_axis
from sklearn.neighbors import radius_neighbors_graph


logger = logging.getLogger(__name__)

# Source:
# https://github.com/albertomontesg/davis-interactive/blob/master/davisinteractive/metrics/jaccard.py
def batched_jaccard(y_true, y_pred, average_over_objects=True, nb_objects=None):
    """ Batch jaccard similarity for multiple instance segmentation.

    Jaccard similarity over two subsets of binary elements $A$ and $B$:

    $$
    \mathcal{J} = \\frac{A \\cap B}{A \\cup B}
    $$

    # Arguments
        y_true: Numpy Array. Array of shape (B x H x W) and type integer giving the
            ground truth of the object instance segmentation.
        y_pred: Numpy Array. Array of shape (B x H x W) and type integer giving the
            prediction of the object segmentation.
        average_over_objects: Boolean. Weather or not to average the jaccard over
            all the objects in the sequence. Default True.
        nb_objects: Integer. Number of objects in the ground truth mask. If
            `None` the value will be infered from `y_true`. Setting this value
            will speed up the computation.

    # Returns
        ndarray: Returns an array of shape (B) with the average jaccard for
            all instances at each frame if `average_over_objects=True`. If
            `average_over_objects=False` returns an array of shape (B x nObj)
            with nObj being the number of objects on `y_true`.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.ndim != 3:
        raise ValueError('y_true array must have 3 dimensions.')
    if y_pred.ndim != 3:
        raise ValueError('y_pred array must have 3 dimensions.')
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have the same shape. {} != {}'.format(y_true.shape, y_pred.shape))

    if nb_objects is None:
        objects_ids = np.unique(y_true[(y_true < 255) & (y_true > 0)])
        nb_objects = len(objects_ids)
    else:
        objects_ids = [i + 1 for i in range(nb_objects)]
        objects_ids = np.asarray(objects_ids, dtype=np.int64)
    if nb_objects == 0:
        raise ValueError('Number of objects in y_true should be higher than 0.')
    nb_frames = len(y_true)

    jaccard = np.empty((nb_frames, nb_objects), dtype=np.float64)

    for i, obj_id in enumerate(objects_ids):
        mask_true, mask_pred = y_true == obj_id, y_pred == obj_id

        union = (mask_true | mask_pred).sum(axis=(1, 2))
        intersection = (mask_true & mask_pred).sum(axis=(1, 2))

        for j in range(nb_frames):
            if np.isclose(union[j], 0):
                jaccard[j, i] = 1.
            else:
                jaccard[j, i] = intersection[j] / union[j]

    if average_over_objects:
        jaccard = jaccard.mean(axis=1)
    return jaccard

# Source:
# https://github.com/albertomontesg/davis-interactive/blob/master/davisinteractive/utils/operations.py
def bezier_curve(points, nb_points=1000):
    """ Given a list of points compute a bezier curve from it.

    # Arguments
        points: ndarray. Array of points with shape (N, 2) with N being the
            number of points and the second dimension representing the
            (x, y) coordinates.
        nb_points: Integer. Number of points to sample from the bezier curve.
            This value must be larger than the number of points given in
            `points`. Maximum value 10000.

    # Returns
        ndarray: Array of shape (1000, 2) with the bezier curve of the
            given path of points.

    """
    
    nb_points = min(nb_points, 1000)

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            '`points` should be two dimensional and have shape: (N, 2)')

    n_points = len(points)
    if n_points > nb_points:
        # We are downsampling points
        return points

    t = np.linspace(0., 1., nb_points).reshape(1, -1)

    # Compute the Bernstein polynomial of n, i as a function of t
    i = np.arange(n_points).reshape(-1, 1)
    n = n_points - 1
    polynomial_array = comb(n, i) * (t**(n - i)) * (1 - t)**i

    bezier_curve_points = polynomial_array.T.dot(points)

    return bezier_curve_points

# Source:
# https://github.com/albertomontesg/davis-interactive/blob/master/davisinteractive/robot/interactive_robot.py
class InteractiveScribblesRobot(object):
    """ Robot that generates realistic scribbles simulating human interaction.

    # Attributes
        kernel_size: Float. Fraction of the square root of the area used
            to compute the dilation and erosion before computing the
            skeleton of the error masks.
        max_kernel_radius: Float. Maximum kernel radius when applying
            dilation and erosion. Default 16 pixels.
        min_nb_nodes: Integer. Number of nodes necessary to keep a connected
            graph and convert it into a scribble.
        nb_points: Integer. Number of points to sample the bezier curve
            when converting the final paths into curves.
    """

    def __init__(self,
                 kernel_size=.15,
                 max_kernel_radius=16,
                 min_nb_nodes=4,
                 nb_points=1000):
        """ Robot constructor
        """
        if kernel_size >= 1. or kernel_size < 0:
            raise ValueError('kernel_size must be a value between [0, 1).')

        self.kernel_size = kernel_size
        self.max_kernel_radius = max_kernel_radius
        self.min_nb_nodes = min_nb_nodes
        self.nb_points = nb_points

    def _generate_scribble_mask(self, mask):
        """ Generate the skeleton from a mask
        Given an error mask, the medial axis is computed to obtain the
        skeleton of the objects. In order to obtain smoother skeleton and
        remove small objects, an erosion and dilation operations are performed.
        The kernel size used is proportional the squared of the area.

        # Arguments
            mask: Numpy Array. Error mask

        Returns:
            skel: Numpy Array. Skeleton mask
        """
        mask = np.asarray(mask, dtype=np.uint8)
        side = np.sqrt(np.sum(mask > 0))

        mask_ = mask
        # kernel_size = int(self.kernel_size * side)
        kernel_radius = self.kernel_size * side * .5
        kernel_radius = min(kernel_radius, self.max_kernel_radius)
        logging.info(
            'Erosion and dilation with kernel radius: {:.1f}'.format(
                kernel_radius), 2)
        compute = True
        while kernel_radius > 1. and compute:
            kernel = disk(kernel_radius)
            mask_ = rank.minimum(mask.copy(), kernel)
            mask_ = rank.maximum(mask_, kernel)
            compute = False
            if mask_.astype(np.bool_).sum() == 0:
                compute = True
                prev_kernel_radius = kernel_radius
                kernel_radius *= .9
                logging.info('Reducing kernel radius from {:.1f} '.format(
                    prev_kernel_radius) +
                                'pixels to {:.1f}'.format(kernel_radius), 1)

        mask_ = np.pad(
            mask_, ((1, 1), (1, 1)), mode='constant', constant_values=False)
        skel = medial_axis(mask_.astype(np.bool_))
        skel = skel[1:-1, 1:-1]
        return skel

    def _mask2graph(self, skeleton_mask):
        """ Transforms a skeleton mask into a graph

        Args:
            skeleton_mask (ndarray): Skeleton mask

        Returns:
            tuple(nx.Graph, ndarray): Returns a tuple where the first element
                is a Graph and the second element is an array of xy coordinates
                indicating the coordinates for each Graph node.

                If an empty mask is given, None is returned.
        """
        mask = np.asarray(skeleton_mask, dtype=np.bool_)
        if np.sum(mask) == 0:
            return None

        h, w = mask.shape
        x, y = np.arange(w), np.arange(h)
        X, Y = np.meshgrid(x, y)

        X, Y = X.ravel(), Y.ravel()
        M = mask.ravel()

        X, Y = X[M], Y[M]
        points = np.c_[X, Y]
        G = radius_neighbors_graph(points, np.sqrt(2), mode='distance')
        
        T = nx.from_scipy_sparse_array(G)#nx.from_scipy_sparse_matrix(G)

        return T, points

    def _acyclics_subgraphs(self, G):
        """ Divide a graph into connected components subgraphs
        Divide a graph into connected components subgraphs and remove its
        cycles removing the edge with higher weight inside the cycle. Also
        prune the graphs by number of nodes in case the graph has not enought
        nodes.

        Args:
            G (nx.Graph): Graph

        Returns:
            list(nx.Graph): Returns a list of graphs which are subgraphs of G
                with cycles removed.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError('G must be a nx.Graph instance')
        S = []  # List of subgraphs of G

        for c in nx.connected_components(G):
            g = G.subgraph(c).copy()

            # Remove all cycles that we may find
            has_cycles = True
            while has_cycles:
                try:
                    cycle = nx.find_cycle(g)
                    weights = np.asarray([G[u][v]['weight'] for u, v in cycle])
                    idx = weights.argmax()
                    # Remove the edge with highest weight at cycle
                    g.remove_edge(*cycle[idx])
                except nx.NetworkXNoCycle:
                    has_cycles = False

            if len(g) < self.min_nb_nodes:
                # Prune small subgraphs
                logging.info('Remove a small line with {} nodes'.format(
                    len(g)), 1)
                continue

            S.append(g)

        return S

    def _longest_path_in_tree(self, G):
        """ Given a tree graph, compute the longest path and return it
        Given an undirected tree graph, compute the longest path and return it.

        The approach use two shortest path transversals (shortest path in a
        tree is the same as longest path). This could be improve but would
        require implement it:
        https://cs.stackexchange.com/questions/11263/longest-path-in-an-undirected-tree-with-only-one-traversal

        Args:
            G (nx.Graph): Graph which should be an undirected tree graph

        Returns:
            list(int): Returns a list of indexes of the nodes belonging to the
                longest path.
        """
        if not isinstance(G, nx.Graph):
            raise TypeError('G must be a nx.Graph instance')
        if not nx.is_tree(G):
            raise ValueError('Graph G must be a tree (graph without cycles)')

        # Compute the furthest node to the random node v
        v = list(G.nodes())[0]
        distance = nx.single_source_shortest_path_length(G, v)
        vp = max(distance.items(), key=lambda x: x[1])[0]
        # From this furthest point v' find again the longest path from it
        distance = nx.single_source_shortest_path(G, vp)
        longest_path = max(distance.values(), key=len)
        # Return the longest path

        return list(longest_path)

    def interact(self,
                 sequence,
                 pred_masks,
                 gt_masks,
                 nb_objects=None,
                 frame=None):
        """ Interaction of the Scribble robot given a prediction.
        Given the sequence and a mask prediction, the robot will return a
        scribble in the region that fails the most.

        # Arguments
            sequence: String. Name of the sequence to interact with.
            pred_masks: Numpy Array. Array with the prediction masks. It must
				be an integer array with shape (B x H x W), with B being the number
				of frames of the sequence.
            gt_masks: Numpy Array. Array with the ground truth of the sequence.
				It must have the same data type and shape as `pred_masks`.
            nb_objects: Integer. Number of objects in the ground truth mask. If
                `None` the value will be infered from `y_true`. Setting this
                value will speed up the computation.
            frame: Integer. Frame to generate the scribble. If not given, the
                worst frame given by the jaccard will be used.

        # Returns
            dict: Return a scribble (default representation).
        """
        robot_start = time.time()

        predictions = np.asarray(pred_masks, dtype=np.int64)
        annotations = np.asarray(gt_masks, dtype=np.int64)

        nb_frames = len(annotations)
        if nb_objects is None:
            obj_ids = np.unique(annotations)
            obj_ids = obj_ids[(obj_ids > 0) & (obj_ids < 255)]
            nb_objects = len(obj_ids)

        obj_ids = [i for i in range(nb_objects + 1)]
        # Infer height and width of the sequence
        h, w = annotations.shape[1:3]
        img_shape = np.asarray([w, h], dtype=np.float64)

        if frame is None:
            jac = batched_jaccard(
                annotations, predictions, nb_objects=nb_objects)
            worst_frame = jac.argmin()
            logging.info(
                'For sequence {} the worst frames is #{} with Jaccard: {:.3f}'.
                format(sequence, worst_frame, jac.min()), 2)
        else:
            worst_frame = frame
        pred, gt = predictions[worst_frame], annotations[worst_frame]

        scribbles = [[] for _ in range(nb_frames)]

        for obj_id in obj_ids:
            logging.info(
                'Creating scribbles from error mask at object_id={}'.format(
                    obj_id), 2)
            start_time = time.time()
            error_mask = (gt == obj_id) & (pred != obj_id)
            if error_mask.sum() == 0:
                logging.info(
                    'Error mask of object ID {} is empty. Skip object ID.'.
                    format(obj_id))
                continue

            # Generate scribbles
            skel_mask = self._generate_scribble_mask(error_mask)
            skel_time = time.time() - start_time
            logging.info(
                'Time to compute the skeleton mask: {:.3f} ms'.format(
                    skel_time * 1000), 2)
            if skel_mask.sum() == 0:
                continue

            G, P = self._mask2graph(skel_mask)
            mask2graph_time = time.time() - start_time - skel_time
            logging.info(
                'Time to transform the skeleton mask into a graph: ' +
                '{:.3f} ms'.format(mask2graph_time * 1000), 2)

            t_start = time.time()
            S = self._acyclics_subgraphs(G)
            t = (time.time() - t_start) * 1000
            logging.info(
                'Time to split into connected components subgraphs ' +
                'and remove the cycles: {:.3f} ms'.format(t), 2)

            t_start = time.time()
            longest_paths_idx = [self._longest_path_in_tree(s) for s in S]
            longest_paths = [P[idx] for idx in longest_paths_idx]
            t = (time.time() - t_start) * 1000
            logging.info(
                'Time to compute the longest path on the trees: {:.3f} ms'.
                format(t), 2)

            t_start = time.time()
            scribbles_paths = [
                bezier_curve(p, self.nb_points) for p in longest_paths
            ]
            t = (time.time() - t_start) * 1000
            logging.info(
                'Time to compute the bezier curves: {:.3f} ms'.format(t), 2)

            end_time = time.time()
            logging.info(
                'Generating the scribble for object id {} '.format(obj_id) +
                'took {:.3f} ms'.format((end_time - start_time) * 1000), 2)
            # Generate scribbles data file
            for p in scribbles_paths:
                p /= img_shape
                path_data = {
                    'path': p.tolist(),
                    'object_id': int(obj_id),
                    'start_time': start_time,
                    'end_time': end_time
                }
                scribbles[worst_frame].append(path_data)

        scribbles_data = {'scribbles': scribbles, 'sequence': sequence}

        t = time.time() - robot_start
        logging.info(('The robot took {:.3f} s to generate all the '
                      'scribbles for {} objects. Sequence {}.').format(
                          t, nb_objects, sequence))
        return scribbles_data