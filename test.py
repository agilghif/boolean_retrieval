from util import sort_intersect_list
from random import randint

a = []
b = []
cntA = 0
cntB = 0
for i in range(100):
    cntA += randint(1,10)
    cntB += randint(1,10)
    a. append(cntA)
    b.append(cntB)

print(a)
print(b)
print(sort_intersect_list(a, b))

with open('collections/1/4202502.txt') as f:
    print(f.read())