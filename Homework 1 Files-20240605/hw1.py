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

# Implement the methods in this class as appropriate. Feel free to add other methods
# and attributes as needed. 
import random

class UndirectedGraph:
    def __init__(self, number_of_nodes):
        """
        Initialize the graph with the specified number of nodes.
        Args:
            number_of_nodes (int): The total number of nodes in the graph. Nodes are numbered from 0 to number_of_nodes - 1.
        """
        self.num_nodes = number_of_nodes
        self.adj_list = {i: [] for i in range(number_of_nodes)}

    def add_edge(self, nodeA, nodeB):
        """
        Add an undirected edge to the graph between nodeA and nodeB.
        Args:
            nodeA (int): Index of the first node.
            nodeB (int): Index of the second node.
        """
        if nodeB not in self.adj_list[nodeA]:
            self.adj_list[nodeA].append(nodeB)
            self.adj_list[nodeB].append(nodeA)
    
    def edges_from(self, nodeA):
        """
        Return a list of all nodes connected to nodeA by an edge.
        Args:
            nodeA (int): Index of the node to retrieve edges from.
        Returns:
            list[int]: List of nodes that have an edge with nodeA.
        """
        return self.adj_list[nodeA]

    def number_of_nodes(self):
        """
        Return the number of nodes in the graph.
        Returns:
            int: The number of nodes in the graph.
        """
        return self.num_nodes

def create_graph(n, p):
    """
    Generate an undirected graph with n nodes. Each pair of nodes is connected with a probability p.
    Args:
        n (int): Number of nodes in the graph.
        p (float): Probability of an edge between any two nodes.
    Returns:
        UndirectedGraph: The generated graph.
    """
    graph = UndirectedGraph(n)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                graph.add_edge(i, j)
    return graph

# Problem 9(b)
def shortest_path(G,i,j):
    ''' Given an UndirectedGraph G and nodes i,j, output the length of the shortest path between i and j in G.
    If i and j are disconnected, output -1.'''
    # TODO: Implement this method
    pass

# Problem 9(c)
def avg_shortest_path(G, num_samples=1000):
    ''' Given an UndirectedGraph G, return an estimate of the average shortest path in G, where the average is taken
    over all pairs of CONNECTED nodes. The estimate should be taken by sampling num_samples random pairs of connected nodes, 
    and computing the average of their shortest paths. Return a decimal number.'''
    # TODO: Implement this method
    pass

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

