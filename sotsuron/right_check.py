
import numpy as np

fatal = []
reach = []
bfdcount = 0
bfcount = 0
fu = np.unique(fatal.copy()).tolist() if fatal else [-1]
ru = np.unique(reach.copy()).tolist() if reach else [-2]
print(fu, ru)
           
bfdcount = (len(set(fu) & set(ru))) / 4

for i in range(len(reach)):
            r = reach[i]
            
            for g in fatal:
                if set(r).issubset(set(g)):
                    bfcount = 1
#print(fu, ru, len(set(fu) & set(ru)))つくる
if ru == [-2] and fu == [-1]:
    bfdcount = 1
    bfcount = 1

print(bfdcount, bfcount)