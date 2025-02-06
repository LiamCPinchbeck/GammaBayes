import warnings
import numpy as np
from gammabayes.priors.core import DiscreteLogPrior

class MTreeNode:
    """Represents a node in a mixture tree structure.

    Attributes:
        value: The value associated with the node.
        id: Unique identifier for the node.
        parent_node: The parent node of the current node.
        prior: Optional prior associated with the node.
        children: List of child nodes.
    """
    def __init__(self, value:list, id:int|str=None, parent_node=None, prior=None):
        """Initialize a tree node.

        Args:
            value: The value of the node.
            id (int | str, optional): Unique identifier for the node. Defaults to None.
            parent_node (MTreeNode, optional): The parent node of this node. Defaults to None.
            prior (DiscreteLogPrior, optional): The prior associated with the node. Defaults to None.
        """
        self.value = value
        self.children = []
        self.id = id  # Unique identifier for each node
        self.parent_node = parent_node
        self.prior = prior

        if id is None:
            if prior is None:
                self.id = None
            else:
                self.id = prior.name
            
        


    def add_child(self, child):
        """Add a child node to this node.

        Args:
            child (MTreeNode): The child node to add.

        Returns:
            MTreeNode: The added child node.
        """
        self.children.append(child)
        return child

    def __repr__(self):
        """Return a string representation of the node.

        Returns:
            str: String representation of the node.
        """
        return f"TreeNode(value={self.value}, id={self.id})"
    

    def deep_copy(self):
        """Create a deep copy of the node and its children.

        Returns:
            MTreeNode: A deep copy of the node.
        """
        copied_node = MTreeNode(value    = self.value, 
                                id  = self.id, 
                                parent_node = self.parent_node, 
                                prior = self.prior)
        copied_node.children = [child.deep_copy() for child in self.children]
        return copied_node




class MTree:
    """Represents a mixture tree structure.

    Attributes:
        root: The root node of the tree.
        nodes: Dictionary of nodes in the tree, indexed by their ids.
        leaves: List of leaf nodes in the tree.
        leaf_values: Dictionary of leaf node values.
    """
    def __init__(self, root:MTreeNode=None, equivalent_weight_node_mappings:list[list]=None):
        """Initialize the tree with an optional root node.

        Args:
            root (MTreeNode, optional): The root node of the tree. Defaults to None.
        """
        if root is None:
            self.root = MTreeNode(1.0, id="root")  # Initialize the root node
            self.nodes = {self.root.id: self.root}
        else:
            self.root = root
            self.nodes = {node.id: node for node in self.collect_nodes(root)}

        if not(equivalent_weight_node_mappings is None):
            self.equivalent_weight_node_mappings = equivalent_weight_node_mappings 
        else:
            self.equivalent_weight_node_mappings = []
         
        self.refresh_leaves()

    def collect_nodes(self, node:MTreeNode):
        """Collect all nodes starting from a given node.

        Args:
            node (MTreeNode): The starting node.

        Yields:
            MTreeNode: The nodes in the tree.
        """
        yield node
        for child in node.children:
            yield from self.collect_nodes(child)

    def refresh_leaves(self):
        """Refresh the list of leaf nodes and their values."""
        self.leaves = []
        self.leaf_values = {}
        self.collect_leaves(self.root)

    def collect_leaves(self, node:MTreeNode):
        """Collect leaf nodes starting from a given node.

        Args:
            node (MTreeNode): The starting node.
        """
        if not node.children:
            self.leaves.append(node)
            self.leaf_values[node.id] = self.compute_weight(node)
        for child in node.children:
            self.collect_leaves(child)

    def generate_id(self, parent:MTreeNode=None, prior=None):
        """Generate a unique id for a new node.

        Args:
            parent (MTreeNode, optional): The parent node. Defaults to None.
            prior (DiscreteLogPrior, optional): The prior associated with the node. Defaults to None.

        Returns:
            str: The generated id.
        """
        if parent is not None:
            idprefix = f"{parent.id}_" 
        elif prior is not None:
            idprefix = f"{prior.name}_"
        else:
            idprefix = f""

        id = 1
        while f"{idprefix}auto_id{id}" in self.nodes:
            id += 1
        return f"{idprefix}auto_id{id}"

    def add(self, value:float, parent:MTreeNode, prior:DiscreteLogPrior = None, id:int|str=None):
        """Add a new node to the tree.

        Args:
            value: The value of the new node.
            parent (MTreeNode): The parent node.
            prior (DiscreteLogPrior, optional): The prior associated with the node. Defaults to None.
            id (int | str, optional): The id of the new node. Defaults to None.

        Raises:
            ValueError: If the node id already exists.

        Returns:
            MTreeNode: The added node.
        """
        if id and id in self.nodes:
            raise ValueError(f"Node ID '{id}' already exists.")
        new_node_id = id if id is not None else self.generate_id(parent)
        new_node = MTreeNode(value, id=new_node_id, parent_node=parent, prior=prior)
        parent.add_child(new_node)
        self.nodes[new_node_id] = new_node
        self.refresh_leaves()
        return new_node

    def compute_weight(self, node:MTreeNode):
        """Compute the weight of a node based on its ancestors.

        Args:
            node (MTreeNode): The node for which to compute the weight.

        Returns:
            float: The computed weight.
        """
        product = node.value
        while node.parent_node:
            node = node.parent_node
            product *= node.value
        return product

    def create_tree(self, layout:list|dict, values:list=None, 
                            parent:MTreeNode=None, index=0, no_values=True,
                            equivalent_weight_node_mappings:list[list]=None):
        """Create a tree from a layout and optional values.

        Args:
            layout: The layout of the tree.
            values (list, optional): The values for the nodes. Defaults to None.
            parent (MTreeNode, optional): The parent node. Defaults to None.
            index (int, optional): The starting index for the values. Defaults to 0.

        Returns:
            int: The next index to be used.
        """
        if values is None:
            no_values = True
            num_in_layer = len(layout)

            values = [1/num_in_layer]*num_in_layer
            
        if isinstance(layout, MTree):
            # If the layout is another Tree, deep copy it
            self.root = layout.root.deep_copy()
            self.refresh_leaves()
            return
        
        if parent is None:
            parent = self.root

        for item in layout:
            if isinstance(item, dict):
                # Recursive case: item is a sublist representing children of the current node
                stem_node_name = list(item.keys())[0]
                new_node = self.add(values[index], parent, id=stem_node_name)

                index += 1

                if no_values:
                    num_in_layer = len(list(item.values())[0])
                    values[index:index] = [1/num_in_layer]*num_in_layer


                index = self.create_tree(layout=list(item.values())[0], values=values, parent=new_node, index=index, no_values=no_values)

            else:
                # Base case: item is an ID
                new_node = self.add(values[index], parent, id=item)
                index += 1

        if not(equivalent_weight_node_mappings is None):
            self.equivalent_weight_node_mappings = equivalent_weight_node_mappings

        # Just to make this clear that it will use the stored value
        else:
            self.equivalent_weight_node_mappings = self.equivalent_weight_node_mappings
        

        return index

    def _tree_str(self, node:MTreeNode=None, level:int=0, prefix:str="", precision:str='3g', prev_str:str="", print_ids:bool=False):
        """Generate a string representation of the tree.

        Args:
            node (MTreeNode, optional): The starting node. Defaults to None.
            level (int, optional): The current level in the tree. Defaults to 0.
            prefix (str, optional): The prefix for the current level. Defaults to "".
            precision (str, optional): The precision for displaying values. Defaults to '3g'.
            prev_str (str, optional): The previous string representation. Defaults to "".
            print_ids (bool, optional): Whether to print node ids. Defaults to False.

        Returns:
            str: The string representation of the tree.
        """
        if node is None:
            node = self.root
        connector = "|__ " if level > 0 else ""
        output_string = prev_str

        if print_ids:
            print_val = str(node.id)
        else:
            print_val = f"{node.value:.{precision}}"

        if level==0:
            output_string += f"{prefix}{connector}{print_val} __\n"
        else:
            output_string += f"{prefix}{connector}{print_val}\n"

        if node.children:
            child_prefix = prefix + ("|   " if level > 0 else "    ")
            for child in node.children:
                output_string = self._tree_str(child, level + 1, child_prefix, precision, prev_str=output_string, print_ids=print_ids)

        return output_string
        

    def print_tree(self, node:MTreeNode=None, level:int=0, prefix:str="", precision:str='3g', print_ids:bool=False):
        """Print the tree structure.

        Args:
            node (MTreeNode, optional): The starting node. Defaults to None.
            level (int, optional): The current level in the tree. Defaults to 0.
            prefix (str, optional): The prefix for the current level. Defaults to "".
            precision (str, optional): The precision for displaying values. Defaults to '3g'.
            print_ids (bool, optional): Whether to print node ids. Defaults to False.
        """
        print(self._tree_str(node=node, level=level, prefix=prefix, precision=precision, print_ids=print_ids))


    def __repr__(self):
        """Return a string representation of the tree.

        Returns:
            str: String representation of the tree.
        """
        output_string = "___ID Structure___:\n"
        output_string += self._tree_str(print_ids=True)
        output_string +='\n'
        output_string +='\n'
        output_string += "\n___Leaf Values___:\n"
        for leaf in self.leaf_values.items():
            output_string += str(leaf)+'\n'

        output_string += "\n__Nodes__:\n"
        for node in self.nodes.values():
            output_string += str(node)+'\n'


        return output_string

    def copy(self):
        """Create a deep copy of the tree.

        Returns:
            MTree: A deep copy of the tree.
        """
        return MTree(self.root.deep_copy())


    # Recursive function to delete all children nodes
    def _remove_children(self, node:MTreeNode):
        """Recursively remove all children of a node.

        Args:
            node (MTreeNode): The node whose children are to be removed.
        """


        # iterate over a copy of the list, as otherwise when iterating
            # the list itself could be changing
        for child in node.children[:]:  
            self._remove_children(child)  # recursively remove children
        
            del self.nodes[child.id]  # remove child from nodes dictionary

        # Clear the children list after removal
        node.children.clear()  


    def delete_node(self, node_id:str|int, _check_equivalencies=True):
        """Delete a node and its children from the tree.

        Args:
            node_id (str | int): The id of the node to delete.

        Returns:
            bool: True if the node was deleted, False otherwise.
        """
        if node_id not in self.nodes:
            warnings.warn(f"No node with ID '{node_id}' exists.")
            return False

        # Get the node to be deleted from the nodes dictionary
        node_to_delete = self.nodes[node_id]

        # If the node is not the root, remove it from its parent's children list
        if node_to_delete.parent_node:
            node_to_delete.parent_node.children = [
                child for child in node_to_delete.parent_node.children if child.id != node_id
            ]

        # Recursively remove this node and all its children from the tree
        self._remove_children(node_to_delete)

        # Finally, remove the node itself from the nodes dictionary
        del self.nodes[node_id]

        if _check_equivalencies:
            # Check if there are equivalent nodes, and if so remove them
            for equivalent_node_list in self.equivalent_weight_node_mappings:

                # Check if there are equivalent nodes
                if node_to_delete.id in equivalent_node_list:

                    print("Equivalency exists?: ", node_to_delete.id, equivalent_node_list)

                    # iterate thourgh list of ids
                    for equivalent_node_id in equivalent_node_list:
                        print("Node iden being checked: ", equivalent_node_id)
                        # If identifications exactly match then the removal should be taken care of 
                            # by the rest of the function. So skip this (actually the same) node
                        if equivalent_node_id!=node_to_delete.id:
                            print("Equivalent node to delete: ", equivalent_node_id)
                            # If not equal carry on recurrence
                            self.delete_node(equivalent_node_id, _check_equivalencies=False)
                break




        # Refresh the leaves and leaf values since the structure of the tree has changed
        self.refresh_leaves()
        return True


    def overwrite(self, values:list):
        """Overwrite the values of the tree nodes with a given list of values.

        Args:
            values (list): The new values for the nodes.
        """
        # Use a queue to handle the breadth-first traversal
        queue = list(self.root.children)  # Start with the root node
        value_index = 0  # Start from the first provided value

        while queue and value_index < len(values):
            current_node = queue.pop(0)
            # Update current node's value
            current_node.value = values[value_index]
            value_index += 1  # Move to the next value

            # Add children of the current nodes to the queue
            queue = list(current_node.children)+queue

        # After updating, recompute leaf values
        self.refresh_leaves()

