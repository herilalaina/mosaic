import random


class BaseScenario():
    def __init__(self, queue, name):
        self.queue = queue
        self.name = name
        self.executed_name_algo = True

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

class ListTask(BaseScenario):
    def __init__(self, name=None, tasks = [], is_ordered = True):
        super().__init__(queue=tasks, name=name)
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

class ChoiceScenario(BaseScenario):
    def __init__(self, name = None, scenarios = [], nb_choice = 1):
        super().__init__(queue=scenarios, name=name)
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

class ComplexScenario(BaseScenario):
    def __init__(self, name = None, scenarios = [], is_ordered = True):
        super().__init__(queue = scenarios, name =  name)
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
            return [s.name for s in self.queue]
        else:
            return self.queue[0].queue_tasks()

"""
x1 = ListTask(is_ordered=False, name = "x1", tasks = ["x1_p1", "x1_p2", "x1_p4", "x1_p5", "x1_p6", "x1_p7"])
x2 = ListTask(is_ordered=True, name = "x2",  tasks = ["x2_p1", "x2_p2", "x2_p4", "x2_p5", "x2_p6", "x2_p7"])

x1.queue_tasks()
start = ChoiceScenario(name = "Model", scenarios=[x1, x2])
start.queue_tasks()
start.call()

for i in ["x1_p1", "x1_p2", "x1_p4", "x1_p5", "x1_p6", "x1_p7", "x2_p1", "x2_p2", "x2_p4", "x2_p5", "x2_p6", "x2_p7"]:
    print(start.execute())
"""
