"""
To perform the symbolic regression we need to define the model.
The function is represented as a tree structure, where each node is an operation or a variable.
"""

import copy
from enum import Enum
from typing import Self

import numpy as np


class NodeType(Enum):
    VARIABLE = 0
    CONSTANT = 1
    ADD = 2
    SUB = 3
    MUL = 4
    DIV = 5
    SIN = 6
    COS = 7
    EXP = 8
    # LOG = 9


valid_children = {
    NodeType.VARIABLE: 0,
    NodeType.CONSTANT: 0,
    NodeType.ADD: 2,
    NodeType.SUB: 2,
    NodeType.MUL: 2,
    NodeType.DIV: 2,
    NodeType.SIN: 1,
    NodeType.COS: 1,
    NodeType.EXP: 1,
    # NodeType.LOG: 1
}


class Node:
    def __init__(self, type, value=None):
        self.type = type
        if type == NodeType.VARIABLE or type == NodeType.CONSTANT:
            assert value is not None, "Variable and constant nodes must have a value."
        self.value = value
        self.children = []
        self.parent = None
        self.depth = 1

    def f(self, x: np.ndarray):
        # Interpret the whole tree as function
        # Recursively evaluate the children
        match self.type:
            case NodeType.VARIABLE:
                return x[self.value]
            case NodeType.CONSTANT:
                return self.value
            case NodeType.ADD:
                return self.children[0].f(x) + self.children[1].f(x)
            case NodeType.SUB:
                return self.children[0].f(x) - self.children[1].f(x)
            case NodeType.MUL:
                return self.children[0].f(x) * self.children[1].f(x)
            case NodeType.DIV:
                return self.children[0].f(x) / self.children[1].f(x)
            case NodeType.SIN:
                return np.sin(self.children[0].f(x))
            case NodeType.COS:
                return np.cos(self.children[0].f(x))
            case NodeType.EXP:
                return np.exp(self.children[0].f(x))
            case NodeType.LOG:
                return np.log(self.children[0].f(x))

    def __str__(self):
        # Print the tree in a human-readable format
        # Recursively print the children
        first_child = self.children[0] if len(self.children) > 0 else "()"
        second_child = self.children[1] if len(self.children) > 1 else "()"

        match self.type:
            case NodeType.VARIABLE:
                return f"x[{self.value}]"
            case NodeType.CONSTANT:
                return f"{self.value}"
            case NodeType.ADD:
                return f"({first_child} + {second_child})"
            case NodeType.SUB:
                return f"({first_child} - {second_child})"
            case NodeType.MUL:
                return f"({first_child} * {second_child})"
            case NodeType.DIV:
                return f"({first_child} / {second_child})"
            case NodeType.SIN:
                return f"sin({first_child})"
            case NodeType.COS:
                return f"cos({first_child})"
            case NodeType.EXP:
                return f"exp({first_child})"
            case NodeType.LOG:
                return f"log({first_child})"

    @property
    def flatten(self):
        return list(self)

    def __iter__(self):
        nodes = self.children.copy() + [self]
        while nodes:
            node = nodes.pop()
            yield node
            nodes.extend(node.children)

    def clone(self):
        return copy.deepcopy(self)

    def __contains__(self, item: Self):
        return item in self.flatten

    def append(self, child):
        self.children.append(child)
        child.parent = self
        child.depth = self.depth + 1
