import random

from mosaic.simulation.rules import ChildRule

class BaseScenario():
    def __init__(self, queue, name, rules):
        self.name = name
        self.executed_name_algo = True
        self.rules = rules

        self.queue = [node for node in queue if not self.child_task(node)]

    def child_task(self, node):
        for rule in self.rules:
            if isinstance(rule, ChildRule) and node in rule.applied_to:
                return True
        return False

    def actualize_queue(self, parent, parent_value):
        for rule in self.rules:
            if isinstance(rule, ChildRule) and parent_value in rule.value and parent == rules.parent:
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

    def finished(self):
        return len(self.queue) == 0

    def queue_tasks(self):
        return self.queue

class AbstractWorkflowScenario(BaseScenario):
    def __init__(self, queue, name, rules):
        super(AbstractWorkflowScenario, self).__init__(queue, name, rules)

    def add_to_current_queue(self, new_node):
        if isinstance(self, WorkflowListTask):
            self.queue.append(new_node)
        elif isinstance(self, WorkflowChoiceScenario):
            self.choosed_scenario.add_to_current_queue(new_node)
        elif isinstance(self, WorkflowComplexScenario):
            self.queue[0].append(new_node)


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

