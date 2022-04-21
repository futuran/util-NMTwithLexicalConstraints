import numpy as np

results = np.array([[20791, 498, 6516], [354, 3417, 1109], [5973, 1061, 922576]])
all = np.sum(results)
TP = np.sum(results[:1,:1]) / all
FN = np.sum(results[:1,2]) / all
FP = np.sum(results[2,:1]) / all
TN = results[2,2] / all

recall = TP / (TP + FN)
precision = TP / (TP + FP)
f1 = 2 * precision * recall / (precision + recall)

print(f'{recall=}')
print(f'{precision=}')
print(f'{f1=}')
