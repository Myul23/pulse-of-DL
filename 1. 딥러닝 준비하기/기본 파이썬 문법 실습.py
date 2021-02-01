# 근데 이 분 왜 python 2 쓰시지.

# persent print
# d: integer, f: float, s: string
x = 3
print("Integer: %01d, %02d, %03d, %04d, %05d" % (x, x, x, x, x))
x = 123.456
print("Float: %.0f, %.1f, %.2f, %1.2f, %2.2f" % (x, x, x, x, x))
x = "Hello world!"
print("String: [%s], [%3s], [%20s]" % (x, x, x))

# loop
dlmethods = ["ANN", "MLP", "CNN", "RNN", "DAE"]
for alg in dlmethods:
    print("[%s]" % alg)

# loop with index
for i, alg in enumerate(dlmethods):
    print("[%d / %s]" % (i, alg))

# if-else
for alg in dlmethods:
    if alg in ["ANN", "MLP", "CNN"]:
        print("[%s] is a feed-forward network." % alg)
    elif alg in ["RNN"]:
        print("[%s] is a recurrent network" % alg)
    else:
        print("[%s] is an supervised method." % alg)

# function
def sum(a, b):
    return a + b


# print("FUNCTION DEFINED.")

a = 10
b = 20
print("[%d] + [%d] is [%d]" % (a, b, sum(a, b)))

# string operation
head = "DEEP LEARNING"
body = "VERY "
tail = "EXCITING."
print(head + "is " + body + tail)
print(head + "is " + body * 10 + tail)

# List
a = []
print("a: [%s], TYPE IS [%s]" % (a, type(a)))

b = [1, 2, 3]
c = ["x", "y", "z"]
d = [b, c]
print(b, c, d, sep="\n")

a = ["a", "b", "c"]
for aval in a:
    print(aval)

b = []
for aidx in range(len(a)):
    b.append(a[aidx])
print(b)

# Dictionary
a = dict()
a["name"] = "Sungjoon"
a["job"] = "Student"
print(a)

b = {"name": "Sungjoon", "job": "Student"}
print(b)
print("NAME IS [%s]" % b["name"])
print("JOB IS [%s]" % b["job"])

# Class
class foo:
    def __init__(self, name):
        self.name = name
        print("HELLO, [%s]" % self.name)

    def boo(self, loud=False):
        if loud:
            print("BOO [%s]" % self.name.upper())
        else:
            print("boo [%s]" % self.name)


print("CLASS DEFINED.")


f = foo("Sungjoon")
f.boo()
f.boo(loud=True)


import numpy as np


def print_np(x):
    print("(1) Type is %s" % type(x), "(2) Shape is %s" % x.shape, "(3) Values are: \n%s" % (x), sep="\n")


# rank 1
x = np.array([1, 2, 3])
print_np(x)

x[0] = 5
print_np(x)

# rank 2
y = np.array([[1, 2, 3], [4, 5, 6]])
print_np(y)

# zeros
a = np.zeros((3, 2))
print_np(a)

# ones
b = np.ones((1, 2))
print_np(b)

# identify
c = np.eye(3, 3)
print_np(c)

# random uniform
d = np.random.random((2, 2))
print_np(d)

# random gaussian
e = np.random.randn(1, 5)
print_np(e)

# array indexing
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print_np(a)

b = a[:2, 1:3]
print_np(b)

# get row
row_r1 = a[1, :]  # return vector
row_r2 = a[1:2, :]  # return matrix
row_r3 = a[[1], :]  # return matrix

print_np(row_r1)
print_np(row_r2)
print_np(row_r3)

# different types
x = np.array([1, 2])
y = np.array([1.0, 2.0])

# type casting
z = np.array([1, 2], dtype=np.int64)

print_np(x)
print_np(y)
print_np(z)

# array math
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
print_np(x)
print_np(y)

# element-wise : 연산자 오버로딩
print(x + y, np.add(x, y), sep="\n")
print(x - y, np.substract(x, y), sep="\n")
print(x * y, np.multiply(x, y), sep="\n")
print(x / y, np.divide(x, y), sep="\n")

# matrix operations
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])
v = np.array([9, 10])
w = np.array([11, 12])

print_np(x)
print_np(y)
print_np(v)
print_np(w)

# dot
print(v.dot(w), np.dot(v, w))
print(x.dot(v), np.dot(x, v))
print(x.dot(y), np.dot(x, y))

# empty like
y = np.empty_like(x)
print_np(x)
print_np(y)
# PYTHON USED 'CALL BY REFERENCE'
# AND SHALLOW COPY (NO DEEP(or value) COPY)

# TILE: 추가로 주어진 행, 열로 반복
w = np.tile(v, (1, 3))
print_np(w)

w = np.tile(v, (2, 3))
print_np(w)

import matplotlib.pyplot as plt

# %matplotlib inline: Jupyter Notebook에서 그림을 그릴 때 필요한 명령어라고 합디다.
print("PLOT READY")

x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()

# plot with legend
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin)
plt.plto(x, y_cos)
plt.xlabel("x axis label")
plt.ylabel("y axis label")
plt.title("Sine and Cosine")
plt.legend(["Sine", "Cosine"])
plt.show()

# subplot
plt.subplot(1, 2, 1)
plt.plot(x, y_sin)
plt.title("Sine")

plt.subplot(1, 2, 2)
plt.plot(x, y_cos)
plt.title("Cosine")

plt.show()
