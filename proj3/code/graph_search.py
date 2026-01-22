from heapq import heappush, heappop  # Recommended.
import numpy as np

from flightsim.world import World

from .occupancy_map import OccupancyMap # Recommended.

class Node:
    def __init__(self,idx, dis = np.inf, h = 0):
        self.parent = None
        self.dis = dis
        self.idx = idx
        self.h = h

    def __lt__(self, other):
        return self.dis+ self.h < other.dis +other.h
    
def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    if occ_map.is_occupied_index(start_index) or occ_map.is_occupied_index(goal_index):
        return None, 0
    
    node_map = np.empty(occ_map.map.shape,dtype=object)
    for index, _ in np.ndenumerate(node_map):
        h =  0 if not astar else (np.linalg.norm(occ_map.index_to_metric_center(index)-goal))
        node_map[index] = Node(idx = index, h= h)
    
    node_map[start_index].dis = 0

    heap = []
    heappush(heap,node_map[start_index])
    neighbor_offsets = genOffsets()#np.load("offsets.npz")['offsets']
    offset_cost = np.linalg.norm(neighbor_offsets * resolution,axis = 1)
    nodes_expanded = 0
    closed_set = set()
    while heap:
        curr = heappop(heap)
        if curr.idx in closed_set:
            continue
        elif curr.idx == goal_index:
            path = []
            while curr is not None:
                path.append(occ_map.index_to_metric_center(curr.idx))
                curr = curr.parent
            path.reverse()
            path[0] = start
            path[-1] = goal
            return np.array(path), nodes_expanded
        closed_set.add(curr.idx)
        nodes_expanded +=1
        neighbors = neighbor_offsets + np.array(curr.idx)
        dist = offset_cost + curr.dis
        for i in range(len(neighbors)):
            neighborIdx = tuple(neighbors[i])
            if occ_map.is_occupied_index(neighborIdx) or neighborIdx in closed_set:
                continue

            neighbor_node = node_map[neighborIdx]

            if dist[i] < neighbor_node.dis:
                neighbor_node.dis = dist[i]
                neighbor_node.parent = curr
                heappush(heap,neighbor_node)

    # Return a tuple (path, nodes_expanded)
    return None, nodes_expanded


def genOffsets():

    moves = []
    for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    moves.append((x, y, z))

    return moves