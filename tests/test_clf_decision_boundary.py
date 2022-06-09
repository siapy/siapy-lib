import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

### Code from:
# https://stackoverflow.com/questions/51801118/sklearn-svm-gives-wrong-decision-boundary?rq=1

# def plot_svc_decision_boundary(svm_clf, xmin, xmax):
#     w = svm_clf.coef_[0]
#     b = svm_clf.intercept_[0]

#     # At the decision boundary, w0*x0 + w1*x1 + b = 0
#     # => x1 = -w0/w1 * x0 - b/w1
#     x0 = np.linspace(xmin, xmax, 200)
#     decision_boundary = -w[0]/w[1] * x0 - b/w[1]

#     margin = 1/w[1]
#     gutter_up = decision_boundary + margin
#     gutter_down = decision_boundary - margin

#     svs = svm_clf.support_vectors_
#     plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
#     plt.plot(x0, decision_boundary, "k-", linewidth=2)
#     plt.plot(x0, gutter_up, "k--", linewidth=2)
#     plt.plot(x0, gutter_down, "k--", linewidth=2)


# f = np.array([
#     [1,2],
#     [1,2],
#     [0,4],
#     [0,4],
# ])
# labels = [0,0,1,1]

# svm_clf = SVC(kernel='linear')
# svm_clf.fit(f, labels)

# plot_svc_decision_boundary(svm_clf, -1, 2.0)
# plt.xlabel('feature 1')
# plt.ylabel('feature 2')
# plt.scatter(f[0, 0], f[0, 1], marker='^', s=80)
# plt.scatter(f[1, 0], f[1, 1], marker='s', s=80)
# plt.show()

################################################################################

f = np.array([
    [400],
    [30],
    [25],
    [10],
    [5],
    [7],
])
labels = [0,0,0,1,1,1]

clf = SVC(kernel='linear')
# clf = LinearDiscriminantAnalysis()
clf.fit(f, labels)

# print(svm_clf.coef_)
# print(svm_clf.intercept_)

boundary = - clf.intercept_/clf.coef_
print(boundary)
