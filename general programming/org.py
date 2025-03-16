"""
Question 1: Print the Organizational Structure

You are given an array where each element is a string containing names. The first name in each string (at index 0) represents the boss, and the subsequent names are the direct reports of that boss. Your task is to print the entire organizational structure in a hierarchical format.
Example:

Input:
[
    "Alice Bob Carol",
    "Bob Dave Eve",
    "Carol Frank Grace"
]
Output:
Alice
  Bob
    Dave
    Eve
  Carol
    Frank
    Grace

Question 2: Print the List of Skip Level Reports

Given the name of a manager, print the list of skip level reports (i.e., the employees who report to the manager's direct reports)
Input: Alice
Output:
    Dave
    Eve
    Frank
    Grace


Time Complexity
Building the Structure: 
O(n⋅m)
where 
n is the number of strings (managers) and 
m is the average number of words per string.

Traversing the Structure (Printing or Finding Reports): 
O(V+E)
where 
V is the number of nodes and 
E is the number of edges.

Space Complexity
Graph/Tree Representation: 
O(V+E)
"""
class Node:
    def __init__(self, name = None):
        self.name = name
        self.children = []
    
    def addChild(self, child):
        self.children.append(child)
        return self
    
def node_exists(nodes, name):
    if name in nodes:
        return nodes[name]
    nodes[name] = Node(name)
    return nodes[name]

def array_to_graph(array):
    # track the nodes are already created 
    nodes = dict()
    root = None
    for string in array:
        names  = string.split(' ')
        node = node_exists(nodes, names[0])

        # set root node
        if root is None:
            root =  node

        for child in names[1:]:
            child_node = node_exists(nodes, child)
            node.addChild(child_node)

    return root, nodes

def preordertraversal(current, i = 0):
    print("\t" * i + current.name)
    for child in current.children:
        preordertraversal(child, i+1)

def traversal(array):
    root, nodes = array_to_graph(array)
    current = root

    preordertraversal(current, i = 0)

def search_name(current, name):
    if current.name == name:
        return current
    
    for child in current.children:
        result = search_name(child, name)
        if result:
            return result
    return None

def get_grandchildren(array, name):
    root, nodes = array_to_graph(array)
    current = root
    current = search_name(current, name)

    if current is None:
        return print("The employee is not found in the org chart")

    # leaf node dont have children or skip level reports
    if current.children==[]:
        return print("The employee is not a Manager")
    
    print(f"The skip level report of {name} are:")
    grand_children = []
    for child in current.children:
        for grand_child in child.children:
            grand_children.append(grand_child)

    if grand_children == []:
        for child in current.children:
            print(child.name)
    else:
        for grand_child in grand_children:
            print(grand_child.name)

def test_case1():
    array = [
        "Alice Bob Carol",
        "Bob Dave Eve",
        "Carol Frank Grace"
    ]
    traversal(array)
    name = "Alice"
    expected = """The skip level report of Alice are:
                Dave
                Eve
                Frank
                Grace"""
    assert expected == get_grandchildren(array, name)

def main():
    test_case1()


if __name__ == "__main__":
    main()