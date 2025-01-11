"""
To perform the symbolic regression we need to define the model.
The function is represented as a tree structure, where each node is an operation or a variable.
"""

import warnings
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
    ABS = 9
    LOG = 10


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
    NodeType.ABS: 1,
    NodeType.LOG: 1,
}

syms = {
    NodeType.ADD: "+",
    NodeType.SUB: "-",
    NodeType.MUL: "*",
    NodeType.DIV: "/",
    NodeType.SIN: "sin",
    NodeType.COS: "cos",
    NodeType.EXP: "exp",
    NodeType.ABS: "abs",
    NodeType.LOG: "log",
}
syms = {v: k for k, v in syms.items() if k in valid_children}


reverse_valid_children = {0: set(), 1: set(), 2: set()}
for k, v in valid_children.items():
    reverse_valid_children[v].add(k)


terminal_set = np.array([NodeType.VARIABLE, NodeType.CONSTANT])
function_set = np.array([k for k in valid_children.keys() if k not in terminal_set])


class Node:
    def __init__(self, type, value=None):
        self.type = type
        if type == NodeType.VARIABLE or type == NodeType.CONSTANT:
            assert value is not None, "Variable and constant nodes must have a value."
        self.value = value
        self._children = []
        self.parent = None
        self.depth = 1

    @property
    def children(self):
        return self._children

    def f(self, x: np.ndarray):
        # Interpret the whole tree as function
        # Recursively evaluate the children
        match self.type:
            case NodeType.VARIABLE:
                return x[self.value]
            case NodeType.CONSTANT:
                return self.value
            case NodeType.ADD:
                return self._children[0].f(x) + self._children[1].f(x)
            case NodeType.SUB:
                return self._children[0].f(x) - self._children[1].f(x)
            case NodeType.MUL:
                return self._children[0].f(x) * self._children[1].f(x)
            case NodeType.DIV:
                try:
                    return self._children[0].f(x) / self._children[1].f(x)
                except ZeroDivisionError:
                    return np.inf
            case NodeType.SIN:
                return np.sin(self._children[0].f(x))
            case NodeType.COS:
                return np.cos(self._children[0].f(x))
            case NodeType.EXP:
                return np.exp(self._children[0].f(x))
            case NodeType.ABS:
                return np.abs(self._children[0].f(x))
            case NodeType.LOG:
                return np.log(self._children[0].f(x))

    def __str__(self):
        # Print the tree in a human-readable format
        # Recursively print the children
        first_child = self._children[0] if len(self._children) > 0 else "()"
        second_child = self._children[1] if len(self._children) > 1 else "()"

        match self.type:
            case NodeType.VARIABLE:
                return f"x[{self.value}]"
            case NodeType.CONSTANT:
                return f"{self.value:.3f}"
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
            case NodeType.ABS:
                return f"abs({first_child})"
            case NodeType.LOG:
                return f"log({first_child})"

    @property
    def flatten(self):
        return list(self)

    def __iter__(self):
        nodes = [
            self,
        ]
        while nodes:
            node = nodes.pop()
            nodes.extend(node._children)
            yield node

    def clone(self):
        children = [child.clone() for child in self._children]
        new_node = Node(self.type, self.value)
        for child in children:
            new_node.append(child)
        return new_node

    def __contains__(self, item: Self):
        return item in self.flatten

    def append(self, child):
        self._children.append(child)
        child.parent = self
        child.depth = self.depth + 1
        nodes = self.children.copy()
        while nodes:
            node = nodes.pop()
            for child in node.children:
                child.depth = node.depth + 1
                nodes.append(child)

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, o: object) -> bool:
        return str(self) == str(o)

    def __repr__(self) -> str:
        return str(self)

    def simplify(self, zero: float = 1e-9, is_root=False) -> Self:
        # Return a simplified version of the node

        simplified_root = self.clone()
        simplified_root._children = [
            child.simplify(zero) for child in simplified_root._children
        ]

        # Simplify constant sub-trees
        if len(simplified_root._children) > 0 and all(
            child.type == NodeType.CONSTANT for child in simplified_root._children
        ):
            with np.errstate(divide="ignore", invalid="ignore"):
                simplified_root = Node(NodeType.CONSTANT, value=simplified_root.f(None))

        # Enforce order to avoid symmetrical trees
        # For commutative operations, enforce left.type > right.type

        enforce_order(simplified_root)

        # x - x -> 0
        # a - a -> 0
        if (
            simplified_root.type == NodeType.SUB
            and simplified_root.children[0].type
            in [NodeType.VARIABLE, NodeType.CONSTANT]
            and simplified_root.children[0].type == simplified_root.children[1].type
            and -zero
            <= simplified_root.children[0].value - simplified_root.children[1].value
            <= zero
        ):
            simplified_root = Node(NodeType.CONSTANT, value=0)
            enforce_order(simplified_root)

        # x / x -> 1
        # a / a -> 1
        if (
            simplified_root.type == NodeType.DIV
            and simplified_root._children[0].type
            in [NodeType.VARIABLE, NodeType.CONSTANT]
            and simplified_root._children[0].type == simplified_root._children[1].type
            and -zero
            <= simplified_root._children[0].value - simplified_root._children[1].value
            <= zero
        ):
            simplified_root = Node(NodeType.CONSTANT, value=1)
            enforce_order(simplified_root)

        # a + 0 -> a
        if (
            simplified_root.type == NodeType.ADD
            and simplified_root._children[1].type == NodeType.CONSTANT
            and -zero <= simplified_root._children[1].value <= zero
        ):
            simplified_root.type = simplified_root._children[0].type
            simplified_root.value = simplified_root._children[0].value
            simplified_root._children = simplified_root._children[0].children
            enforce_order(simplified_root)

        # 0 + x -> x
        if (
            simplified_root.type == NodeType.ADD
            and simplified_root._children[0].type == NodeType.CONSTANT
            and -zero <= simplified_root._children[0].value <= zero
        ):
            simplified_root.type = simplified_root._children[1].type
            simplified_root.value = simplified_root._children[1].value
            simplified_root._children = simplified_root._children[1].children
            enforce_order(simplified_root)

        # a - 0 -> a
        # x - 0 -> x
        if (
            simplified_root.type == NodeType.SUB
            and simplified_root._children[1].type == NodeType.CONSTANT
            and -zero <= simplified_root._children[1].value <= zero
        ):
            simplified_root.type = simplified_root._children[0].type
            simplified_root.value = simplified_root._children[0].value
            simplified_root._children = simplified_root._children[0].children
            enforce_order(simplified_root)

        # a * 1 -> a
        if (
            simplified_root.type == NodeType.MUL
            and simplified_root._children[1].type == NodeType.CONSTANT
            and -zero <= simplified_root._children[1].value - 1 <= zero
        ):
            simplified_root.type = simplified_root._children[0].type
            simplified_root.value = simplified_root._children[0].value
            simplified_root._children = simplified_root._children[0].children
            enforce_order(simplified_root)

        # a / 1 -> a
        if (
            simplified_root.type == NodeType.DIV
            and simplified_root._children[1].type == NodeType.CONSTANT
            and simplified_root._children[1].value == 1
        ):
            simplified_root.type = simplified_root._children[0].type
            simplified_root.value = simplified_root._children[0].value
            simplified_root._children = simplified_root._children[0].children
            enforce_order(simplified_root)

        # 0 / a -> 0
        if (
            simplified_root.type == NodeType.DIV
            and simplified_root._children[0].type == NodeType.CONSTANT
            and simplified_root._children[0].value == 0
        ):
            simplified_root = Node(NodeType.CONSTANT, value=0)
            enforce_order(simplified_root)

        # 0 * a -> 0
        if simplified_root.type == NodeType.MUL and (
            (
                simplified_root._children[0].type == NodeType.CONSTANT
                and -zero <= simplified_root._children[0].value <= zero
            )
            or (
                simplified_root._children[1].type == NodeType.CONSTANT
                and -zero <= simplified_root._children[1].value <= zero
            )
        ):
            simplified_root = Node(NodeType.CONSTANT, value=0)
            enforce_order(simplified_root)

        # a + (b + x) -> [a+b] + x
        # a + (b - x) -> [a+b] - x
        if (
            simplified_root.type == NodeType.ADD
            and simplified_root.children[0].type == NodeType.CONSTANT
            and simplified_root.children[1].type in [NodeType.ADD, NodeType.SUB]
            and simplified_root.children[1].children[0].type == NodeType.CONSTANT
        ):
            simplified_root.type = simplified_root.children[1].type
            children = [None, None]
            children[0] = simplified_root.children[1].children[0]
            children[1] = Node(
                NodeType.CONSTANT,
                simplified_root.children[0].value
                + simplified_root.children[1].children[0].value,
            )
            simplified_root._children = children
            enforce_order(simplified_root)

        # a * (b * x) -> [a*b] * x
        # a * (b / x) -> [a*b] / x
        if (
            simplified_root.type == NodeType.MUL
            and simplified_root.children[0].type == NodeType.CONSTANT
            and simplified_root.children[1].type in [NodeType.MUL, NodeType.DIV]
            and simplified_root.children[1].children[1].type == NodeType.CONSTANT
        ):
            simplified_root.type = simplified_root.children[1].type
            children = [None, None]
            children[0] = simplified_root.children[1].children[0]
            children[1] = Node(
                NodeType.CONSTANT,
                value=simplified_root.children[0].value
                * simplified_root.children[1].children[1].value,
            )
            simplified_root._children = children
            enforce_order(simplified_root)

        # a - (x * -b) -> a + (x*b)
        # a - (x / -b) -> a + (x/b)
        if (
            simplified_root.type == NodeType.SUB
            and simplified_root.children[1].type in [NodeType.MUL, NodeType.DIV]
            and simplified_root.children[1].children[1].type == NodeType.CONSTANT
            and simplified_root.children[1].children[1].value < 0
        ):
            simplified_root.type = NodeType.ADD
            simplified_root.children[1].children[1].value = (
                -simplified_root.children[1].children[1].value
            )
            enforce_order(simplified_root)

        # a*x + b*x -> (a+b)*x
        if (
            simplified_root.type == NodeType.ADD
            and simplified_root.children[0].type == NodeType.MUL
            and simplified_root.children[1].type == NodeType.MUL
            and simplified_root.children[0].children[0]
            == simplified_root.children[1].children[0]
            and simplified_root.children[0].children[0].type == NodeType.CONSTANT
            and simplified_root.children[0].children[1].type == NodeType.VARIABLE
            and simplified_root.children[1].children[1].type == NodeType.VARIABLE
        ):
            simplified_root.type = NodeType.MUL
            children = [None, None]
            children[0] = Node(
                NodeType.CONSTANT,
                value=simplified_root.children[0].children[0].value
                + simplified_root.children[1].children[0].value,
            )
            children[1] = simplified_root.children[0].children[1]
            simplified_root._children = children
            enforce_order(simplified_root)

        # a/(b/x) -> a/b*x
        if (
            simplified_root.type == NodeType.DIV
            and simplified_root.children[0].type == NodeType.CONSTANT
            and simplified_root.children[1].type == NodeType.DIV
            and simplified_root.children[1].children[0].type == NodeType.CONSTANT
            and simplified_root.children[1].children[1].type == NodeType.VARIABLE
        ):
            simplified_root.type = NodeType.MUL
            children = [None, None]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                children[1] = Node(
                    NodeType.CONSTANT,
                    simplified_root.children[0].value
                    / simplified_root.children[1].children[1].value,
                )
            children[0] = simplified_root.children[1].children[0]
            simplified_root._children = children
            enforce_order(simplified_root)

        # sometimes you have a chain of + nodes on the left and on the right very similar branches
        if (
            (simplified_root.type == NodeType.ADD)
            and (simplified_root.children[0].type == NodeType.ADD)
            and (simplified_root.children[1].type in reverse_valid_children[2])
            and (
                simplified_root.children[0].children[1].type
                in reverse_valid_children[2]
            )
            and (simplified_root.children[1].type in reverse_valid_children[2])
            and (simplified_root.children[1].children[0].type == NodeType.CONSTANT)
            and (
                simplified_root.children[1].children[1]
                == simplified_root.children[0].children[1].children[1]
            )
            and (
                simplified_root.children[0].children[1].children[0].type
                == NodeType.CONSTANT
            )
        ):
            children = [None, None]
            if simplified_root.children[1].type == NodeType.MUL:
                children[0] = simplified_root.children[0].children[0]
                children[1] = simplified_root.children[1]
                children[1].children[0].value += (
                    simplified_root.children[0].children[1].children[0].value
                )
                simplified_root._children = children
            elif simplified_root.children[1].type == NodeType.DIV:
                ...
            enforce_order(simplified_root)

        # abs(abs(a)) -> abs(a)
        if (
            simplified_root.type == NodeType.ABS
            and simplified_root.children[0].type == NodeType.ABS
        ):
            simplified_root = simplified_root.children[0]

        # log(exp(a)) or exp(log(a)) -> a
        if (
            simplified_root.type == NodeType.LOG
            and simplified_root.children[0].type == NodeType.EXP
        ) or (
            simplified_root.type == NodeType.EXP
            and simplified_root.children[0].type == NodeType.LOG
        ):
            simplified_root = simplified_root.children[0].children[0]
            enforce_order(simplified_root)

        # Reset depths and parents
        if is_root:
            simplified_root.depth = 1
            simplified_root.parent = None
        nodes = [simplified_root]
        while nodes:
            node = nodes.pop()
            for child in node._children:
                child.depth = node.depth + 1
                child.parent = node
                nodes.append(child)

        return simplified_root

    def draw(self, block: bool = True, ax=None):
        import matplotlib.pyplot as plt

        # Nodes are drawn top to bottom, right to left

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.yaxis.set_visible(False)
            ax.xaxis.set_visible(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(self)
        else:
            ax.set_title(self)

        # Iterate over the nodes and create a list ordered by depth
        draw_nodes = np.empty(shape=(max([n.depth for n in self.flatten]),), dtype=list)
        for node in self.flatten:
            if draw_nodes[node.depth - 1] is None:
                draw_nodes[node.depth - 1] = []
            draw_nodes[node.depth - 1].append(node)

        # Draw the nodes
        coords = {}
        for i, nodes in enumerate(draw_nodes):
            for j, node in enumerate(nodes):
                x = (j + 0.5) / len(nodes)
                y = 0.98 - i / len(draw_nodes)
                if node.type == NodeType.VARIABLE:
                    text = f"x[{node.value}]"
                elif node.type == NodeType.CONSTANT:
                    text = f"{node.value:.2f}"
                else:
                    text = syms[node.type]

                ax.text(
                    x,
                    y,
                    text,
                    ha="center",
                    va="center",
                    bbox=dict(
                        facecolor="white",
                        alpha=1,
                        boxstyle="circle,pad=0.8",
                        lw=0.7,
                    ),
                    fontweight="bold",
                )
                coords[id(node)] = (x, y)

        # Draw the edges
        for node in self.flatten:
            if node.parent is not None:
                x0, y0 = coords[id(node.parent)]
                x1, y1 = coords[id(node)]
                ax.plot([x0, x1], [y0, y1], color="black")

        if ax is None:
            plt.show(block=block)


def enforce_order(root):
    if root.type in [NodeType.ADD, NodeType.MUL]:
        left, right = root._children
        if left.type.value > right.type.value or (
            left.type == right.type
            and left.type in [NodeType.CONSTANT, NodeType.VARIABLE]
            and left.value < right.value
        ):
            root._children = [right, left]
