import numpy as np


# BIOの順で格納。0行目は、正解がBで、予測が順にBIO。
# ref\pred | B | I | O
#    B     |   |   |
#    I     |   |   |
#    O     |   |   |

results = np.array([[20803, 523, 6507], [319, 3478, 1087], [6040, 1127, 923040]])
all = np.sum(results)
TP = np.sum(results[:2,:2]) / all
FN = np.sum(results[:2,2]) / all
FP = np.sum(results[2,:2]) / all
TN = results[2,2] / all

recall = TP / (TP + FN)
precision = TP / (TP + FP)
f1 = 2 * precision * recall / (precision + recall)

print(f'{recall=}')
print(f'{precision=}')
print(f'{f1=}')

