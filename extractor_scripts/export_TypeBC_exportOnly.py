
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
import glob
from urllib.parse import urlparse
import gzip

subPropertyOf = 'http://www.w3.org/2000/01/rdf-schema#subPropertyOf'
sameas = 'http://www.w3.org/2002/07/owl#sameAs'
owleqP = 'http://www.w3.org/2002/07/owl#equivalentProperty'


rdfs_label = "http://www.w3.org/2000/01/rdf-schema#label"
rdfs_comment = 'http://www.w3.org/2000/01/rdf-schema#comment'
rdfs_isDefinedBy = "http://www.w3.org/2000/01/rdf-schema#isDefinedBy"
# skos_inScheme = "http://www.w3.org/2004/02/skos/core#inScheme"
# mads_scheme = "http://www.loc.gov/mads/rdf/v1#isMemberOfMADSScheme"
# skos_topConceptOf = "http://www.w3.org/2004/02/skos/core#topConceptOf"


meta_eqSet = "https://krr.triply.cc/krr/metalink/def/equivalenceSet"
meta_comm = "https://krr.triply.cc/krr/metalink/def/Community"
meta_identity_statement = "https://krr.triply.cc/krr/metalink/def/IdentityStatement"
rdf_statement = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement"

rdf_subject = "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"
rdf_object = "http://www.w3.org/1999/02/22-rdf-syntax-ns#object"
rdf_predicate = "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate"

rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# my extension:
# https://krr.triply.cc/krr/metalink/file_MD5/<file MD5>

my_file_IRI_prefix = "https://krr.triply.cc/krr/metalink/fileMD5/" # followed by the MD5 of the data
my_file = "https://krr.triply.cc/krr/metalink/def/file"
my_exist_in_file = "https://krr.triply.cc/krr/metalink/def/existsInFile" # a relation
my_has_label_in_file = "https://krr.triply.cc/krr/metalink/def/hasLabelInFile" # a relation
my_has_comment_in_file = "https://krr.triply.cc/krr/metalink/def/hasCommentInFile" # a relation
my_has_num_occurences_in_files = "https://krr.triply.cc/krr/metalink/def/numOccurences" #


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

# should export these relations
# labels
file =  open('typeB_relations_Sep17.csv', 'w', newline='')
writer = csv.writer(file,  delimiter='\t')
writer.writerow(["Relation", "NumOfTriples"])
for l in label_relations:
	_, cardinality = hdt_lod.search_triples("", l, "")
	writer.writerow([l, cardinality])
# comments
file =  open('typeC_relations_Sep17.csv', 'w', newline='')
writer = csv.writer(file,  delimiter='\t')
writer.writerow(["Relation", "NumOfTriples"])
for l in comment_relations:
	_, cardinality = hdt_lod.search_triples("", l, "")
	writer.writerow([l, cardinality])
