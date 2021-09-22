
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
from extend_metalink import *
import requests
from requests.exceptions import Timeout

PATH_LOD = "/scratch/wbeek/data/LOD-a-lot/data.hdt"
hdt_lod = HDTDocument(PATH_LOD)

PATH_META = "/home/jraad/ssd/data/identity/metalink/metalink-2/metalink-2.hdt"
hdt_metalink = HDTDocument(PATH_META)

rdfs_subclass = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
owl_equivalentClass = 'http://www.w3.org/2002/07/owl#equivalentClass'
dbo_interlan = "http://dbpedia.org/ontology/wikiPageInterLanguageLink"

rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

use_template = "http://dbpedia.org/property/wikiPageUsesTemplate"
dbr_disambig = "http://dbpedia.org/resource/Template:Disambig" #105043
dbr_disambiguation = "http://dbpedia.org/resource/Template:Disambiguation" # 54271

# find all the transitive closure of subPropertyOf, sameAs:
def find_subPropertyOf_eqPropertyOf_closure(source_relations, sizebound):
	size = 0
	while size != len (source_relations):
		# update size record
		size = len (source_relations)
		print ('so far, I found ', size)
		# find all the new relations
		new_relations = set()
		for r in source_relations:
			triples, cardinality = hdt_lod.search_triples("", rdfs_subclass, r)
			for s, _, _  in triples:
				new_relations.add(s)

			triples, cardinality = hdt_lod.search_triples("", owl_equivalentClass, r)
			for s, _, _  in triples:
				new_relations.add(s)

			triples, cardinality = hdt_lod.search_triples(r, owl_equivalentClass, "")
			for _, _, s  in triples:
				new_relations.add(s)

			triples, cardinality = hdt_lod.search_triples(r, dbo_interlan, "")
			for _, _, s  in triples:
				new_relations.add(s)

			triples, cardinality = hdt_lod.search_triples("", dbo_interlan, r)
			for s, _, _  in triples:
				new_relations.add(s)

		source_relations = source_relations.union(new_relations)

	print ('After computing the closure (under equicalenceClass and subClassOf), we found ', len (source_relations), ' relations!')
	count = 0
	to_return = []
	for s in source_relations:
		s_triples, s_cardinality = hdt_lod.search_triples("", use_template, s) # use_template
		if s_cardinality >= sizebound:
			to_return.append(s)
			count += 1
			print ('\t',s)
			print ("\t\tWith", s_cardinality)
	print ("There are ", count, " properties (with more than", sizebound ," entries) above")
	return to_return

# all_eq_classes_of_disambig = find_subPropertyOf_eqPropertyOf_closure(set([dbr_disambig]), 1000)

dbr_disambig = "http://dbpedia.org/resource/Template:Disambig" #105043
dbr_disambiguation = "http://dbpedia.org/resource/Template:Disambiguation" # 54271

with open( "sameas_disambiguation_entities.nt", 'w') as output:
	writer = csv.writer(output, delimiter=' ')

	print ('For dbr:Dis')
	count_dis = 0
	triples, s_cardinality = hdt_lod.search_triples("", use_template, dbr_disambig)
	for (s,_,_) in triples:
		_, cardinality = hdt_metalink.search_triples("", rdf_subject, s)
		_, cardinality2 = hdt_metalink.search_triples("", rdf_object, s)

		if cardinality2 >0 or cardinality >0 :
			count_dis += 1
			writer.writerow(['<'+s+'>', '<'+use_template+'>', '<' + dbr_disambig + '>', '.'])

	print (count_dis , 'out of ', s_cardinality, ' are in the identity graph')
	print ('{:10.2f}'.format(count_dis/s_cardinality))

	print ('For dbr:Disambiguation')
	count_dis = 0
	triples, s_cardinality = hdt_lod.search_triples("", use_template, dbr_disambiguation)
	for (s,_,_) in triples:
		_, cardinality = hdt_metalink.search_triples("", rdf_subject, s)
		_, cardinality2 = hdt_metalink.search_triples("", rdf_object, s)

		if cardinality2 >0 or cardinality >0 :
			count_dis += 1
			writer.writerow(['<'+s+'>', '<'+use_template+'>', '<' + dbr_disambiguation + '>', '.'])

	print (count_dis , 'out of ', s_cardinality, ' are in the identity graph')
	print ('{:10.2f}'.format(count_dis/s_cardinality))
