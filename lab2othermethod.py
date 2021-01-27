import csv
a = []
with open('EnjoySport.csv','r') as cfile:
    for row in csv.reader(cfile):
        a.append(row)
        print(row)
n = len(a[0]) - 1
print("Initially")
S = ["0"] * n
G = ["?"] * n
print(f"S0: {S}")
print("G0: {0}".format(G))
S = a[0][:-1] #first training example
tmp = []
for i,row in enumerate(a):
    if row[-1] == "Yes":
        for j,attrib in enumerate(row[:-1]):
            if attrib != S[j]:
                S[j] = "?"
            for k,g in enumerate(tmp):
                if g[j] != "?" and g[j] != S[j]:
                    del tmp[k]
    else:
        for j,attrib in enumerate(row[:-1]):
            if attrib != S[j] and S[j] != "?":
                G[j] = S[j]
                tmp.append(G)
                G = ["?"] * n
    print("----------------------------------------------------")
    print("For Training example {0} ".format(i+1,S))
    print("S{0}: {1}".format(i+1,S))
    if(tmp==[]):
        print("G{0}: {1}".format(i+1,G))
    else:
        print("G{0}: {1}".format(i+1,tmp))