#!/usr/bin/env python

import numpy as np
from kmodes import kmodes
import matplotlib.pyplot
import pylab
import csv

# stocks with their market caps, sectors and countries
#~ syms = np.genfromtxt('stocks.csv', dtype=str, delimiter=',')[:, 0]
#~ X = np.genfromtxt('stocks.csv', dtype=object, delimiter=',')[:, 1:]

a = open('ideal_ready_ads.csv','r')
lines = a.readlines()
rows = [line.split(",") for line in lines]
rows = rows[1:]
for row in rows:
	row[-1]=row[-1][:-1]
#rows = rows[1:]
print(rows[0][-1])
rows = np.array(rows)
syms = rows[:,0]
print(len(syms))
X = rows[:,1:-1]

#print(syms[0:5])
#print(X[0:5])
#print(syms)
#print(X)

kproto = kmodes.KModes(n_clusters=6, init='Cao', verbose=2)
clusters = kproto.fit_predict(X, categorical=[0,1,2,3])

newData = ["57139","835106","2","Air Travel#Business Travel"]
cluster = kproto.predict(newData)
print(cluster[0])

#~ for s, c in zip(syms, clusters):
    #~ print("Symbol: {}, cluster:{}".format(s, c))

bids = [i for i in range(0,len(syms))]


mod = list(list())
for i in range(0,len(clusters)):
	old = rows[i]
	old = np.append(old,clusters[i])
	mod.append(old)

with open("clusters_ads.csv","w") as f:
	spam = csv.writer(f)
	spam.writerows(mod)
#~ matplotlib.pyplot.scatter(bids,clusters,c=[])
#~ matplotlib.pyplot.show()
