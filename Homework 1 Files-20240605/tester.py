from hw1 import *

def test_is_connected():
    # Test case 1: A fully connected graph
    graph = UndirectedGraph(4)
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(0, 3)
    graph.add_edge(1, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 3)
    assert is_connected(graph) == True, "Test case 1 failed"

    # Test case 2: A graph with one disconnected node
    graph = UndirectedGraph(4)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    # Node 3 is disconnected
    assert is_connected(graph) == False, "Test case 2 failed"

    # Test case 3: A graph with all nodes in a line (connected)
    graph = UndirectedGraph(4)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    assert is_connected(graph) == True, "Test case 3 failed"

    # Test case 4: A graph with two separate components
    graph = UndirectedGraph(6)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(3, 4)
    graph.add_edge(4, 5)
    assert is_connected(graph) == False, "Test case 4 failed"

    # Test case 5: A graph with a single node
    graph = UndirectedGraph(1)
    assert is_connected(graph) == True, "Test case 5 failed"

    # Test case 6: A graph with no edges and multiple nodes
    graph = UndirectedGraph(3)
    assert is_connected(graph) == False, "Test case 6 failed"

    # Test case 7: A large random connected graph
    random.seed(42)  # Setting the seed for reproducibility
    graph = create_graph(10, 0.5, seed=42)
    assert is_connected(graph) == True, "Test case 7 failed"

    # Test case 8: A large random disconnected graph
    graph = create_graph(10, 0.1, seed=42)
    assert is_connected(graph) == False, "Test case 8 failed"

    print("All test cases passed!")

# Run the tester
test_is_connected()