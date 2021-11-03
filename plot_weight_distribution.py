# this script extracts the weight from
# sameas_laundromat_metalink_sum_weight.hdt
#

import networkx as nx
from SameAsEqGraph import *
from pyvis.network import Network
import community
import collections
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import requests
from collections import Counter
from rfc3987 import  parse
import urllib.parse
from hdt import HDTDocument, IdentifierPosition
from z3 import *
from rdflib.namespace import XSD
import csv
from extend_metalink import *


f = plt.figure()
f.set_figwidth(4)
f.set_figheight(1.5)
barWidth = 0.33
ax = plt.subplot(111)

count_total_edges_gold = 232347

count_weight_distribution_gold = {2: 118642, 1: 72041, 3: 30142, 5: 5747, 0: 3312, 4: 2376, 6: 54, 8: 21, 7: 12}
#
# count_weight_distribution_overall = {2 : 193970384,
# 1 : 199500469,
# 5 : 28912215,
# 3 : 68626509,
# 4 : 10677424,
# 7 : 125984,
# 6 : 34437,
# 8 : 34,
# 5407 : 1,
# 9 : 12,
# 10 : 3,
# 232 : 1
# }

count_weight_distribution_overall = {2 : 193970384,
1 : 199500469,
5 : 28912215,
3 : 68626509,
4 : 10677424,
7 : 125984,
6 : 34437,
8 : 34,
# 5407 : 1,
9 : 12,
10 : 3,
# 232 : 1
}

count_total_edges_overall = sum(count_weight_distribution_overall.values())
#
# x2 = [x for x in range(0, max(count_weight_distribution_gold.keys())+1)]
x2=count_weight_distribution_gold.keys()
y2 = [count_weight_distribution_gold[x]/count_total_edges_gold for x in x2]
ax.bar(x2, y2, color ='red', width=barWidth, label='Gold standard', align='center')


x3 =count_weight_distribution_overall.keys()
# x3 = [x for x in range(0, max(count_weight_distribution_overall.keys())+1)]
y3 = [count_weight_distribution_overall[x]/count_total_edges_overall for x in x3]
x3 = [x + barWidth*1 for x in x3]
ax.bar(x3, y3, color ='blue', width=barWidth, label='Overall', align='center')





ax.autoscale(tight=True)

ax.legend()

# plt.yscale('log')
plt.xlabel("Weight")
plt.ylabel("Frequency (in percentage)")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.legend()
plt.savefig('weght_districution.png', bbox_inches='tight', dpi = 300)
plt.show()
