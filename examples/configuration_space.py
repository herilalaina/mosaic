# Import ConfigSpace and different types of parameters
from mosaic.external.ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition


from mosaic.mosaic import Search


# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()
# We define a few possible types of SVM-kernels and add them as "kernel" to our cs
kernel = CategoricalHyperparameter(
    "kernel", ["linear", "rbf", "poly", "sigmoid"], default_value="poly")
cs.add_hyperparameter(kernel)
# There are some hyperparameters shared by all kernels
C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0)
shrinking = CategoricalHyperparameter(
    "shrinking", ["true", "false"], default_value="true")
cs.add_hyperparameters([C, shrinking])
# Others are kernel-specific, so we can add conditions to limit the searchspace
degree = UniformIntegerHyperparameter(
    "degree", 1, 5, default_value=3)     # Only used by kernel poly
coef0 = UniformFloatHyperparameter(
    "coef0", 0.0, 10.0, default_value=0.0)  # poly, sigmoid
cs.add_hyperparameters([degree, coef0])
use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
use_coef0 = InCondition(child=coef0, parent=kernel, values=["poly", "sigmoid"])
cs.add_conditions([use_degree, use_coef0])
# This also works for parameters that are a mix of categorical and values from a range of numbers
# For example, gamma can be either "auto" or a fixed float
gamma = CategoricalHyperparameter(
    "gamma", ["auto", "value"], default_value="auto")  # only rbf, poly, sigmoid
gamma_value = UniformFloatHyperparameter(
    "gamma_value", 0.0001, 8, default_value=1)
cs.add_hyperparameters([gamma, gamma_value])
# We only activate gamma_value if gamma is set to "value"
cs.add_condition(InCondition(child=gamma_value,
                             parent=gamma, values=["value"]))
# And again we can restrict the use of gamma in general to the choice of the kernel
cs.add_condition(InCondition(child=gamma, parent=kernel,
                             values=["rbf", "poly", "sigmoid"]))
cs.add_condition(InCondition(child=degree, parent=kernel, values=["poly"]))
