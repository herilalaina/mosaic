import random

from mosaic.utils import random_uniform_on_log_space


class Parameter():
    def __init__(self, name = None, value_list = [], type_sampling = None,
                 type = None):
        self.name = name
        self.value_list = value_list
        self.type_sampling = type_sampling
        self.type = type

        if type_sampling not in ["uniform", "choice", "constant", "log_uniform"]:
            raise Exception("Can not handle {0} type".format(self.type))

    def get_info(self):
        return self.value_list, self.type_sampling

    def sample_new_value(self):
        if self.type_sampling == "choice":
            return random.choice(self.value_list)
        elif self.type_sampling == "uniform":
            if self.type == 'int':
                return random.randint(self.value_list[0], self.value_list[1])
            else:
                return random.uniform(self.value_list[0], self.value_list[1])
        elif self.type_sampling == "constant":
            return self.value_list
        elif self.type_sampling == "log_uniform":
            return random_uniform_on_log_space(self.value_list[0], self.value_list[1])