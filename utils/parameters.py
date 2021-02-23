import itertools
from collections.abc import Mapping
from collections import OrderedDict


def _split_dict(param_dict):

    if not isinstance(param_dict, Mapping):
        return param_dict, itertools.repeat(())

    return param_dict.keys(), param_dict.values()


def _prepend_to_all(tuples, new_value):
    for t in tuples:
        yield (new_value, *t)


def _generate_param_value_grid(param_name, values, sub_dicts):
    for value, sub_dict in zip(values, sub_dicts):

        this_param_value = (param_name, value)
        other_params_values = generate_params_grid_iterator(sub_dict)

        prod = _prepend_to_all(other_params_values, this_param_value)

        yield prod


def _generate_param_grid(param_item):
    param_name, values_sub_dicts = param_item

    values, sub_dicts = _split_dict(values_sub_dicts)

    param_grid = (itertools.chain.from_iterable(
        _generate_param_value_grid(param_name, values, sub_dicts)
    ))

    return param_grid


def generate_params_grid_iterator(params_dict):

    # if params_dict is empty there are no combinations
    # so yield empty tuple
    if not params_dict:
        yield ()
        return

    # calculate param_grid for every parameter in dictionary
    params_grid = map(_generate_param_grid, params_dict.items())

    # all posible combinations of params in dict
    params_combinations = itertools.product(*params_grid)

    # chain params into one iterable
    for params_combination in params_combinations:
        yield itertools.chain.from_iterable(params_combination)


def generate_params_grid(param_dict):

    return map(dict, generate_params_grid_iterator(param_dict))


if __name__ == "__main__":
    # mf_parameters_dict = {
    #     "param1": [11, 12],
    #     "param2": {
    #         21: {"param2.1": [211, 212],
    #              "param2.2": [213, 214],
    #              },
    #         22: {"param2.1": [221, 222],
    #              "param2.2": [223, 224],
    #              }
    #     },
    #     "param3": {
    #         31: {"param3.1": {
    #             311: {"param3.2": [3111, 3112]},
    #             312: {"param3.2": [3121, 3122]},
    #         }},
    #         32: {"param3.1": {
    #             321: {"param3.2": [3211, 3212]},
    #             322: {"param3.2": [3221, 3222]},
    #         }},
    #     },
    # }
    mf_parameters_dict = OrderedDict({
        "a": OrderedDict({
            3: {"b": [10]},
            7: {"b": [10, 20, 40]},
            10: {"b": [10, 30, 70]},
            15: {"b": [10, 50, 70, 150]},
            25: {"b": [10, 50, 200, 400]},
            40: {"b": [50, 500, 1000]}
        })
    })

    params=generate_params_grid(mf_parameters_dict)

    for i, param in enumerate(params):
        print(i, param)