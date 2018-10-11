import random
import networkx as nx

from mosaic.simulation.rules import ChildRule

class BaseScenario():
    def __init__(self):
        pass

    def finished(self):
        raise NotImplemented()

    def queue_tasks(self):
        raise NotImplemented()

    def call(self):
        raise NotImplemented()

    def execute(self, task):
        raise NotImplemented()


class AbstractImportanceScenario(BaseScenario):
    def __init__(self, _dependency_graph, rules):
        super(AbstractImportanceScenario, self).__init__()
        self.rules = rules
        self.executed_task = []
        self.dependency_graph = nx.DiGraph()
        for source, target in _dependency_graph.items():
            if source not in self.dependency_graph:
                self.dependency_graph.add_node(source)
            for target in target:
                if target not in self.dependency_graph:
                    self.dependency_graph.add_node(target)
                self.dependency_graph.add_edge(source, target)


class ImportanceScenarioStatic(AbstractImportanceScenario):
    def __init__(self, dependency_graph, rules):
        super(ImportanceScenarioStatic, self).__init__(dependency_graph, rules)
        self.child_node = set()
        for r in self.rules:
            self.child_node.update(r.applied_to)

    def finished(self):
        if not self.executed_task:
            return False
        return self.dependency_graph.successors(self.executed_task[-1]) is None

    def queue_tasks(self):
        if not self.executed_task:
            return list(nx.topological_sort(self.dependency_graph))[0]
        else:
            return self.dependency_graph.successors(self.executed_task[-1])

    def _call(self):
        if self.finished():
            raise Exception("No task in queue.")
        if len(self.executed_task) > 0:
            task = random.choice(list(self.dependency_graph.successors(self.executed_task[-1])))
        else:
            task = list(nx.topological_sort(self.dependency_graph))[0]
        self.executed_task.append(task)
        return task

    def call(self):
        task = self._call()
        if task in self.child_node:
            return self.call()
        else:
            return task

    def execute(self, task):
        if not self.executed_task:
            if task != "root":
                raise Exception("No task in queue.")
        """else:
            list_tasks = self.dependency_graph.successors(self.executed_task[-1])
            if task not in list_tasks:
                raise Exception("No task in queue.")"""
        self.executed_task.append(task)
        return task

    def actualize_queue(self, parent, parent_value):
        for rule in self.rules:
            if isinstance(rule, ChildRule) and parent_value in rule.value and parent == rule.parent:
                for n in rule.applied_to:
                    self.child_node.remove(n)


class AbstractWorkflowScenario(BaseScenario):
    def __init__(self, queue, name, rules):
        super(AbstractWorkflowScenario, self).__init__()
        self.name = name
        self.executed_name_algo = True
        self.rules = rules

        self.queue = [node for node in queue if not self.child_task(node)]

    def child_task(self, node):
        for rule in self.rules:
            if isinstance(rule, ChildRule) and node in rule.applied_to:
                return True
        return False

    def finished(self):
        return len(self.queue) == 0

    def queue_tasks(self):
        return self.queue

    def add_to_current_queue(self, new_node):
        if isinstance(self, WorkflowListTask):
            self.queue.append(new_node)
        elif isinstance(self, WorkflowChoiceScenario):
            self.choosed_scenario.add_to_current_queue(new_node)
        elif isinstance(self, WorkflowComplexScenario):
            self.queue[0].append(new_node)

    def actualize_queue(self, parent, parent_value):
        for rule in self.rules:
            if isinstance(rule, ChildRule) and parent_value in rule.value and parent == rule.parent:
                for n in rule.applied_to:
                    self.add_to_current_queue(n)

    def call(self):
        if self.finished():
            raise Exception("No task in queue.")
        elif self.executed_name_algo:
            self.executed_name_algo = False
            return self.name
        return self._call()

    def execute(self, task):
        if self.finished():
            raise Exception("No task in queue.")
        elif self.executed_name_algo:
            if task != self.name:
                raise Exception("Name scenario must be called first")
            self.executed_name_algo = False
            return self.name
        return self._execute(task)


class WorkflowListTask(AbstractWorkflowScenario):
    def __init__(self, name=None, tasks = [], is_ordered = True, rules = []):
        super(WorkflowListTask, self).__init__(queue=tasks, name=name, rules = rules)
        self.is_ordered = is_ordered

    def _call(self):
        if self.is_ordered:
            return self.queue.pop(0)
        else:
            random.shuffle(self.queue)
            return self.queue.pop()

    def _execute(self, task):
        if self.is_ordered:
            if self.queue[0] == task:
                return self.queue.pop(0)
            else:
                raise Exception("Task {0} not in pipeline.")
        else:
            if task in self.queue:
                self.queue.remove(task)
                return task
            else:
                raise Exception("Task {0} not in pipeline.")

    def queue_tasks(self):
        if len(self.queue) == 0:
            return []
        if self.is_ordered:
            return [self.queue[0]]
        else:
            return self.queue

class WorkflowChoiceScenario(AbstractWorkflowScenario):
    def __init__(self, name = None, scenarios = [], nb_choice = 1, rules = []):
        super(WorkflowChoiceScenario, self).__init__(queue=scenarios, name=name, rules = rules)
        self.nb_choice = nb_choice
        self.choosed = False

    def _call(self):
        if not self.choosed:
            index = random.randint(0, len(self.queue) - 1)
            self.choosed_scenario = self.queue[index]
            self.choosed = True
        return self.choosed_scenario.call()

    def _execute(self, task):
        if not self.choosed:
            for s in self.queue:
                if s.name == task:
                    self.choosed_scenario = s
                    break
            self.choosed = True
        return self.choosed_scenario.execute(task)

    def queue_tasks(self):
        if len(self.queue) == 0:
            return []

        if self.executed_name_algo:
            return [self.name]
        elif not self.choosed:
            return [s.name for s in self.queue]
        else:
            return self.choosed_scenario.queue_tasks()

    def finished(self):
        if not self.choosed:
            return False
        return self.choosed_scenario.finished()

class WorkflowComplexScenario(AbstractWorkflowScenario):
    def __init__(self, name = None, scenarios = [], is_ordered = True, rules = []):
        super(WorkflowComplexScenario, self).__init__(queue = scenarios, name =  name, rules = rules)
        self.is_ordered = is_ordered

        if self.is_ordered == False:
            random.shuffle(self.queue)

    def _call(self):
        if self.is_ordered:
            task = self.queue[0].call()
            if self.queue[0].finished():
                self.queue.pop(0)
            return task
        else:
            task = self.queue[0].call()
            if self.queue[0].finished():
                self.queue.pop(0)
                random.shuffle(self.queue)
            return task

    def _execute(self, task):
        if self.is_ordered:
            task = self.queue[0].execute(task)
            if self.queue[0].finished():
                self.queue.pop(0)
            return task
        else:
            task = self.queue[0].execute(task)
            if self.queue[0].finished():
                self.queue.pop(0)
            return task

    def queue_tasks(self):

        if len(self.queue) == 0:
            return []

        if self.executed_name_algo:
            return [self.name]
        elif self.queue[0].executed_name_algo:
            if self.is_ordered:
                return [self.queue[0].name]
            else:
                return [s.name for s in self.queue]
        else:
            return self.queue[0].queue_tasks()
