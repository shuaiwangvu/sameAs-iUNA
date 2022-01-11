
import pandas as pd
import numpy as np
import datetime
import pickle
import time
import networkx as nx
import sys
import csv
from z3 import *
from bidict import bidict
import matplotlib.pyplot as plt
import tldextract
import json
import random
from collections import Counter
from hdt import HDTDocument, IdentifierPosition
import glob
from urllib.parse import urlparse
import gzip
# from extend_metalink import *
import requests
from requests.exceptions import Timeout


def read_file (file_name):
	pairs = []
	eq_file = open(file_name, 'r')
	reader = csv.DictReader(eq_file, delimiter='\t',)
	for row in reader:
		s = row["SUBJECT"]
		o = row["OBJECT"]
		# c = row["METALINK_ID"]
		pairs.append([s,o])
	return pairs


gs = [4170, 5723,6617,6927,9411,9756,11116,12745,14872,18688,25604,33122,37544,
39036, 42616,96073,97757,99932,236350,240577,337339,395175,712342,1133953,
1140988,4635725,9994282,14514123]

sameas = "http://www.w3.org/2002/07/owl#sameAs"

for id in gs:
	# print ('reading ', id)
	filename = str(id) +'_edges.tsv'
	pairs = read_file(filename)

	print (id, ' has ', len(pairs), ' pairs')

	nt_filename = str(id)+'_edges.nt'
	with open(nt_filename, 'w') as writer:
		for (s, o) in pairs:
			line = '<' + str(s) + '> '
			line += '<' + sameas + '> '
			line += '<' + str(o) + '> . \n'
			# print (line)
			writer.write(str(line))
