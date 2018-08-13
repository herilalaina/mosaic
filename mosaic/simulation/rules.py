
class BaseRule():
    def __init__(self, applied_to = []):
        self.applied_to = applied_to

    def test(self, list_nodes = []):
        raise NotImplemented()

class DependanceRule(BaseRule):
    def __init__(self, applied_to = [], parent = None):
        super(DependanceRule, self).__init__(applied_to=applied_to)
        self.parent = parent

    def test(self):
        raise NotImplemented()



class ChildRule(BaseRule):
    def __init__(self, applied_to = [], parent = None, value = []):
        super().__init__(applied_to = applied_to)
        self.parent = parent
        self.value = value

    def test(self, list_nodes = []):
        parent_value = None
        has_node = [False] * len(self.applied_to)

        for node_name, v in list_nodes:
            if node_name == self.parent:
                parent_value = v
            if node_name in self.applied_to:
                index = self.applied_to.index(node_name)
                has_node[index] = True

        return False if (parent_value not in self.value) and (True in has_node) else True

class ValueRule(BaseRule):
    def __init__(self, constraints = []):
        super().__init__(applied_to = [])
        for c, v in constraints:
            self.applied_to.append(c)
        self.constraints = constraints

    def test(self, list_nodes = []):
        has_node = []
        for node_name, v in list_nodes:
            if node_name in self.applied_to:
                index = self.applied_to.index(node_name)
                has_node.append((self.constraints[index][1] == v))

        return not (True in has_node and False in has_node)
