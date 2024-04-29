import numpy as np
from decimal import Decimal, ROUND_HALF_UP


def linearScale(value_list, new_min, new_max):
    old_min = min(value_list)
    old_max = max(value_list)
    if old_max == old_min:
        print("value_list is constant:", value_list)
        if old_max > new_max:
            return [new_max] * len(value_list)
        elif old_max < new_min:
            return [new_min] * len(value_list)
        else:
            return value_list
    if not old_max <= new_max or not old_min >= new_min:
        scaled_values = []
        for value in value_list:
            scaled_value = ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
            scaled_value = int(Decimal(scaled_value).quantize(0, ROUND_HALF_UP))
            scaled_values.append(scaled_value)

        return scaled_values
    return value_list


# Example usage
value_list = [-1, 2, 0, 5, 1, 1]
scaled_values = linearScale(value_list, -3, 4)
print(scaled_values)
# scaled_values = [int(Decimal(num).quantize(0, ROUND_HALF_UP)) for num in scaled_values]
# print(scaled_values)
