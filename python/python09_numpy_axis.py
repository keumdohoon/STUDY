import numpy as np
a = np.array([[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
                [[1, 2, 3], [1, 2, 3], [1, 2, 3]]])
print(a.shape) #(2. 3. 3)

# axis = 0
a_axis0 = np.sum(a, axis = 0)
print('a_axis0_sum:', a_axis0)

# a_axis0_sum: [[2 4 6]
#              [2 4 6]
#              [2 4 6]]

# axis = 1
a_axis1 = np.sum(a, axis = 1)
print('a_axis1_sum:', a_axis1)
    # a_axis1_sum: [[3 6 9]
    #              [3 6 9]]

a_axis2 = np.sum(a, axis = 2)
print('a_axis2_sum:', a_axis2)
    # a_axis2_sum: [[6 6 6]
    #               [6 6 6]]


# axis = 0
a_axis_3 = np.sum(a, axis = -3)
print('a_axis0_sum:', a_axis_3)

# a_axis0_sum: [[2 4 6]
#              [2 4 6]
#              [2 4 6]]

# axis = 1
a_axis_2 = np.sum(a, axis = -2)
print('a_axis1_sum:', a_axis_2)
    # a_axis1_sum: [[3 6 9]
    #              [3 6 9]]

a_axis_1 = np.sum(a, axis = -1)
print('a_axis2_sum:', a_axis_1)
    # a_axis2_sum: [[6 6 6]
    #               [6 6 6]]