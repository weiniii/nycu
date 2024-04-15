import math

X = []
f = open('testfile.txt','r')
for r in f.readlines():
    r = r.strip('\n')
    X.append(list(map(int, r)))

f.close()

a , b = map(int, input('Input parameters a and b: ').split(' '))


times = len(X)

for t in range(times):
    N = len(X[t])
    w = sum(X[t])
    p = w / N
    s = "".join(list(map(str, X[0])))
    print(f"case {t + 1}:  {s}")
    print(f"Likelihood: {math.comb(N, w) * (p ** w) * ((1 - p) ** (N - w))}")
    print(f"Beta prior:      a = {a} b = {b}")
    
    a += w
    b += (N - w)
    
    print(f"Beta posterior:  a = {a} b = {b}")
    print("\n")
    
    