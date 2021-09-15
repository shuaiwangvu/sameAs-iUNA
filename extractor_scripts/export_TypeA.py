
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
from tarjan import tarjan
from collections import Counter
from hdt import HDTDocument, IdentifierPosition

subPropertyOf = 'http://www.w3.org/2000/01/rdf-schema#subPropertyOf'
sameas = 'http://www.w3.org/2002/07/owl#sameAs'
owleqP = 'http://www.w3.org/2002/07/owl#equivalentProperty'


rdfs_label = "http://www.w3.org/2000/01/rdf-schema#label"
rdfs_comment = 'http://www.w3.org/2000/01/rdf-schema#comment'
rdfs_isDefinedBy = "http://www.w3.org/2000/01/rdf-schema#isDefinedBy"
# skos_inScheme = "http://www.w3.org/2004/02/skos/core#inScheme"
# mads_scheme = "http://www.loc.gov/mads/rdf/v1#isMemberOfMADSScheme"
# skos_topConceptOf = "http://www.w3.org/2004/02/skos/core#topConceptOf"

definition_in_relations = set([rdfs_isDefinedBy])
# source_relations = set([rdfs_isDefinedBy, skos_inScheme, mads_scheme, skos_topConceptOf])
label_relations = set([rdfs_label])
comment_relations = set([rdfs_comment])

PATH_LOD = "/scratch/wbeek/data/LOD-a-lot/data.hdt"
hdt_lod = HDTDocument(PATH_LOD)


# find all the transitive closure of subPropertyOf, sameAs:
def find_subPropertyOf_eqPropertyOf_closure(source_relations, sizebound):
	size = 0
	while size != len (source_relations):
		# update size record
		size = len (source_relations)

		# find all the new relations
		new_relations = set()
		for r in source_relations:
			triples, cardinality = hdt_lod.search_triples("", subPropertyOf, r)
			for s, _, _  in triples:
				new_relations.add(s)

			triples, cardinality = hdt_lod.search_triples("", owleqP, r)
			for s, _, _  in triples:
				new_relations.add(s)

			triples, cardinality = hdt_lod.search_triples(r, owleqP, "")
			for s, _, _  in triples:
				new_relations.add(s)

		source_relations = source_relations.union(new_relations)

	print ('After computing the closure (under subPropertyOf and equivalentProperty), we found ', len (source_relations), ' relations!')
	count = 0
	to_return = []
	for s in source_relations:
		s_triples, s_cardinality = hdt_lod.search_triples("", s, "")
		if s_cardinality >= sizebound:
			to_return.append(s)
			count += 1
			print ('\t',s)
			print ("\t\tWith", s_cardinality)
	print ("There are ", count, " properties (with more than", sizebound ," entries) above")
	return to_return



definition_in_relations = find_subPropertyOf_eqPropertyOf_closure(definition_in_relations, 100000)
print ("\n"*3)
label_relations = find_subPropertyOf_eqPropertyOf_closure(label_relations, 100000)
print ("\n"*3)
comment_relations = find_subPropertyOf_eqPropertyOf_closure(comment_relations, 100000)

#
start = time.time()

# total_unique_entities = 500000

triples, cardinality = hdt_lod.search_triples("", sameas, "")
count = 0
collect_entities = set()
for (s, _, o) in triples:
	# if count %1000 == 0:
	# 	print (count)
	count += 1

	collect_entities.add(s)
	collect_entities.add(o)
	# if len (collect_entities) >= total_unique_entities:
	# 	break
print ('found ', len (collect_entities), ' unique entities in ', count, ' sameas triples')

total_unique_entities = len (collect_entities)

end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("Time taken: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

count_A = 0
count_B = 0
count_C = 0
count_overall = 0
f = open("TypeA_export.log","w")

# file_name_extra = 'typeA_new.nt'
# file_extra =  open(file_name_extra, 'w', newline='')
# writer_extra = csv.writer(file_extra, delimiter=' ')
count_processed = 0
f_typeA = open ("typeA_f.nt", "w")
count = 0
for e in collect_entities:
	# if count >1000:
	# 	break
	if count_processed % 100000 == 0:
		f.write('\nProcessed: %d'%count_processed)
		f.write('\nExported: %d'%count)
		end = time.time()
		hours, rem = divmod(end-start, 3600)
		minutes, seconds = divmod(rem, 60)
		t = "\nTime taken: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
		f.write(t)
	for a in definition_in_relations:
		triples, cardinality = hdt_lod.search_triples(e, a, "")
		if cardinality > 0:
			for s,p,o in triples:
				# s = '<'+s+'>'
				# p = '<'+p+'>'
				# o = '<'+o+'>'
				f_typeA.write('<'+s+'> <'+ p +'> <'+ o +'> .\n')
				count += 1
				# export it as nt file.
	count_processed += 1

f.write('\n num of exported entries: %d'%count)
