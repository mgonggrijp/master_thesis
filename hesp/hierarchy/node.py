class Node(object):
    """ Node object """

    def __init__(
            self, name, parent, children, ancestors, siblings, depth, sub_hierarchy, idx
    ):
        self.name = name
        self.parent = parent
        self.depth = depth
        self.children = children
        self.ancestors = ancestors
        self.siblings = siblings
        self.sub_hierarchy = sub_hierarchy
        self.idx = idx
