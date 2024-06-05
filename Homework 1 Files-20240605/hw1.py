# Skeleton file for HW1
# =====================================
# IMPORTANT: You are NOT allowed to modify the method signatures 
# (i.e. the arguments and return types each function takes). 
# We will pass your grade through an autograder which expects a specific format.
# =====================================


# Do not include any other files or an external package, unless it is one of
# [numpy, pandas, scipy, matplotlib, random]
# please contact us before sumission if you want another package approved.
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
# Implement the methods in this class as appropriate. Feel free to add other methods
# and attributes as needed. 
class UndirectedGraph:
    def __init__(self, number_of_nodes):
        """
        Initialize the graph with the specified number of nodes. Each node is identified by an integer index.
        Uses an adjacency matrix for storage where each cell (i, j) indicates the presence of an edge between nodes i and j.
        Args:
            number_of_nodes (int): The total number of nodes in the graph. Nodes are numbered from 0 to number_of_nodes - 1.
        """
        self.num_nodes = number_of_nodes
        self.adj_matrix = np.zeros((number_of_nodes, number_of_nodes), dtype=int)
    
    def add_edge(self, nodeA, nodeB):
        """
        Add an undirected edge to the graph between nodeA and nodeB.
        Args:
            nodeA (int): Index of the first node.
            nodeB (int): Index of the second node.
        """
        self.adj_matrix[nodeA][nodeB] = 1
        self.adj_matrix[nodeB][nodeA] = 1
    
    def edges_from(self, nodeA):
        """
        Return a list of all nodes connected to nodeA by an edge.
        Args:
            nodeA (int): Index of the node to retrieve edges from.
        Returns:
            list[int]: List of nodes that have an edge with nodeA.
        """
        return list(np.where(self.adj_matrix[nodeA] == 1)[0])
    
    def check_edge(self, nodeA, nodeB):
        """
        Check if there is an edge between nodeA and nodeB.
        Args:
            nodeA (int): Index of the first node.
            nodeB (int): Index of the second node.
        Returns:
            bool: True if there is an edge, False otherwise.
        """
        return self.adj_matrix[nodeA][nodeB] == 1
    
    def number_of_nodes(self):
        """
        Return the number of nodes in the graph.
        Returns:
            int: The number of nodes in the graph.
        """
        return self.num_nodes
    
    def print_graph(self):
        print("Adjacency Matrix:")
        print(self.adj_matrix)


# Problem 9(a)
def create_graph(n, p, seed=None):
    """
    Generate an undirected graph with n nodes. Each pair of nodes is connected with a probability p using NumPy's
    random number generation for deterministic randomness when seed is set.
    Args:
        n (int): Number of nodes in the graph.
        p (float): Probability of an edge between any two nodes.
        seed (int, optional): Seed for the random number generator for reproducibility.
    Returns:
        UndirectedGraph: The generated graph.
    """
    rng = np.random.default_rng(seed)  # Use NumPy's random generator with optional seeding
    graph = UndirectedGraph(n)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:  # Use rng to determine edge creation
                graph.add_edge(i, j)
    return graph
# Problem 9(b)
def shortest_path(G, i, j):
    """
    Finds the shortest path from node i to node j using a BFS algorithm.
    Args:
        G (UndirectedGraph): The graph.
        i (int): Start node.
        j (int): End node.
    Returns:
        int: The length of the shortest path, or -1 if no path exists.
    """
    if i == j:  # Check if start and end nodes are the same
        return 0
    
    # Initialize a queue for BFS
    queue = deque([(i, 0)])  # Each element is a tuple (current_node, current_distance)
    visited = set([i])  # Keep track of visited nodes to prevent revisiting
    
    while queue:
        current_node, current_distance = queue.popleft()
        
        # Iterate through each neighbor of the current node
        for neighbor in G.edges_from(current_node):
            if neighbor == j:
                return current_distance + 1  # Return the distance if end node is found
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, current_distance + 1))
    
    return -1  # Return -1 if no path is found

#probelm 9(c)
def avg_shortest_path(G, num_samples=1000, seed=None):
    """
    Estimate the average shortest path length by randomly sampling connected node pairs.
    Args:
        G (UndirectedGraph): The graph.
        num_samples (int): Number of samples to take.
        seed (optional, int): Seed for random number generation for reproducibility.
    Returns:
        float: The estimated average shortest path length.
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.adj_list.keys())
    num_nodes = len(nodes)
    total_path_length = 0
    sampled_pairs = 0

    while sampled_pairs < num_samples:
        i, j = rng.choice(num_nodes, size=2, replace=False)
        path_length = shortest_path(G, i, j)
        if path_length != -1:  # Ensure that nodes are connected
            total_path_length += path_length
            sampled_pairs += 1

    return total_path_length / num_samples if sampled_pairs > 0 else -1

# Problem 10(a)
def create_fb_graph(filename = "facebook_combined.txt"):
    ''' This method should return a undirected version of the facebook graph as an instance of the UndirectedGraph class.
    You may assume that the input graph has 4039 nodes.'''    
    # TODO: Implement this method 
    # for line in open(filename):
    #     pass
    pass

# Please include any additional code you use for analysis, or to generate graphs, here.
# Problem 9(c) if applicable.
# Problem 9(d)
# Problem 10(b)
# Problem 10(c) if applicable.
# Problem 10(d) if applicable.


# Example of usage and simulation
if __name__ == "__main__":
    # Create a graph instance with 5 nodes
    graph = UndirectedGraph(5)
    
    # Add some edges
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)
    graph.add_edge(3, 4)
    
    # Print the adjacency matrix to visualize the graph
    graph.print_graph()
