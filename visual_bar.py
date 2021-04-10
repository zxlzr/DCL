# import matplotlib.pyplot as plt
# import numpy as np
#
# name_list = ['LN', 'No Norm', 'BN', 'DCL']
# data3 = [0.99, 0.94, 0.5, 0.4]
# data1 = [0.99, 0.94, 0.8, 0.6]
# x = list(range(len(data1)))
# total_width, n = 0.6, 2
# width = total_width / n
# plt.figure(figsize=(4.5, 4.5))
#
# plt.bar(np.array(x) - width, data1, hatch='/', color='white', edgecolor='blue', width=width, label='positive',
#         tick_label=name_list)
# plt.bar(np.array(x), data3, hatch='/', color='white', edgecolor='red', width=width, label='negative')
# # plt.ylim([0,110])
# # plt.xlabel('')
# plt.ylabel('Average Cosine Similarity')
# plt.legend(loc='upper right')
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

plt.ylim(0, 1)
name_list = ['LN', 'No Norm', 'BN', 'DCL']
num_list = [0.6, 0.6, 0.7, 0.6]
num_list1 = [0.1, 0.2, 0.3, 0.1]
total_width, n = 0.8, 2
x = [i - total_width // 2 for i in range(len(num_list))]

plt.ylabel('Average Cosine Similarity')
width = total_width / n
plt.xticks(np.arange(0, 4, 0.5))
plt.bar(x, num_list, width=width, label='positive', fc='y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, num_list1, width=width, label='negative', tick_label=name_list, fc='r')
plt.legend()
plt.show()
