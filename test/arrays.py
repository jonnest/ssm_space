################################################################################
#                                                                              #
#   This file is part of the Statistical Shape Modeling library                #
#                                                                              #
#   2026 SSM Space Project                                                     #
#   Author: Jonathan Nestmann                                                  #
#                                                                              #
################################################################################
def create_presence_array(large_array, subarray):
    presence_array = [1 if num in subarray else 0 for num in large_array]
    return presence_array

# Example usage
large_array = [5, 8, 12, 15, 20, 25, 30]
subarray = [8, 15, 30]
result = create_presence_array(large_array, subarray)
print(result)  # Output: [0, 1, 0, 1, 0, 0, 1]