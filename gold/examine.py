# this script examins the annotated connected components
# print how many are annotated unknown, the prefix of IRIs of such entities.
# print the annotation and proportion of each.

import csv
from collections import Counter
import networkx as nx
import pandas as pd
import tldextract

def get_namespace_prefix (e):
	prefix, name, sign = get_name(e)
	return prefix


def get_name (e):
	name = ''
	prefix = ''
	sign = ''
	if e.rfind('/') == -1 : # the char '/' is not in the iri
		if e.split('#') != [e]: # but the char '#' is in the iri
			name = e.split('#')[-1]
			prefix = '#'.join(e.split('#')[:-1]) + '#'
			sign = '#'
		else:
			name = None
			sign = None
			prefix =  None
	else:
		name = e.split('/')[-1]
		prefix = '/'.join(e.split('/')[:-1]) + '/'
		sign = '/'

	return prefix, sign, name



def read_file (file_name):
	pairs = []
	eq_file = open(file_name, 'r')
	reader = csv.DictReader(eq_file, delimiter='\t',)
	for row in reader:
		s = row["Entity"]
		o = row["Annotation"]
		pairs.append((s,o))
	return pairs


gs = [4170, 5723,6617,6927,9411,9756,11116,12745,14872,18688,25604,33122,37544,
39036, 42616,96073,97757,99932,236350,240577,337339,395175,712342,1133953,
1140988,4635725,9994282,14514123]

single = []
multiple = []

sum_num_entities = 0
total_num_unknown = 0
prefix_ct = Counter()
prefix_ct_unknown = Counter()
for id in gs:
	# print ('reading ', id)
	filename = str(id) +'.tsv'
	pairs = read_file(filename)
	sum_num_entities += len (pairs)
	print ('\n***********************\n', id, ' has ', len (pairs), ' entities')
	ct = Counter ()

	for (e, a) in pairs:
		ct[a] += 1
		p = get_namespace_prefix(e)
		prefix_ct[p] += 1
		if a == 'unknown':
			prefix_ct_unknown[p] += 1

	for a in ct.keys():
		print ('\t', a, ' has\t', ct[a], " = {:10.2f}".format(ct[a]/len(pairs)))

	l = [a =='unknown' for (e, a) in pairs]
	print ('num of unknowns: ',sum (l))
	print ("in percentage, it is {:10.2f}".format(sum(l)/len (pairs)*100 ), '%')
	total_num_unknown += sum (l)


prefix_ct_unknown = {k: v for k, v in sorted(prefix_ct_unknown.items(), key=lambda item: item[1])}

prefix_ct_pct = {}

for p in prefix_ct_unknown.keys():
	if prefix_ct_unknown[p] > 1 and prefix_ct_unknown[p]/prefix_ct[p] > 0.3:
		print ('\t', p , ' has ', prefix_ct_unknown[p], ' unknowns')
		print ('\t\twith a total of ', prefix_ct[p], ', giving {:10.2f}'.format(prefix_ct_unknown[p]/prefix_ct[p]))
		prefix_ct_pct[p] = prefix_ct_unknown[p]/prefix_ct[p]

prefix_ct_pct = {k: v for k, v in sorted(prefix_ct_pct.items(), key=lambda item: item[1])}
print ('****')
for p in prefix_ct_pct.keys():
	if prefix_ct_pct[p] > 0.05:
		print ('\t', p , ' has {:10.2f}'.format(prefix_ct_pct[p]*100), ' pct')


print ('there are ', len (gs), ' files (connected components)')
print ('in total, you manually annotated ', sum_num_entities, ' entities')
print ('there are ', total_num_unknown, ' entities annotated unknown giving {:10.2f}'.format(total_num_unknown/sum_num_entities*100))
