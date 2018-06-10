v0 = 1
v1 = 1
v2 = [None] * 10

i = 0
j = 2

while i < len(v2):
    if i >= 2:
        a = 2
        while a < v2.index(None):  # not v2[i] == None
            v2[a - 1] = v2[a - 1] + v2[a - 2]
            v2[a] = v2[0]
            a += 1
    else:
        v2[i] = 1
    print str(v2)
    i += 1
