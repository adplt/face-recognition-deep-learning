v0 = 1
v1 = 1
v2 = [0] * 10

i = 0
j = 2

while i < 5:
    if i == 0:
        v2[i] = 1
        print v2[i]
    elif i == 1:
        v2[i] = 1
        print str(v2[i - 1]) + ' ' + str(v2[i])
    else:
        print str(v2[0]) + ' '
        a = 2
        while a < len(v2):
            v2[a] = v2[a - 2] + v2[a - 1]
            # v2[a - 2] = v2[a - 1]
            # v2[a - 1] = v2[a]
            print str(v2[a]) + ' '
            a += 2
            # print 'panjang v2: ' + str(len(v2))
            # v2[a + 1] = 1
        # b = 0
        # while b < len(v2) + 1:
        #     print str(v2[b + 2]) + ' '
        #     b += 1
        print str(v2[len(v2) - 1])

    print '\n'
    i += 1
    # else:
    #     v2[i - 2] = v2[i - 4] + v2[i - 3]
    #     print str(v0) + ' ' + str(v2) + ' ' + str(v1) + '\n'
    #     i += 1
