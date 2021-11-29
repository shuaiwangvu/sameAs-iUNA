

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


rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

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
		c = row["Comment"]
		pairs.append([s,o,c])
	return pairs


gs = [4170, 5723,6617,6927,9411,9756,11116,12745,14872,18688,25604,33122,37544,
39036, 42616,96073,97757,99932,236350,240577,337339,395175,712342,1133953,
1140988,4635725,9994282,14514123]

single = []
multiple = []

PATH_DIS = "../sameas_disambiguation_entities_Nov.hdt"
hdt_dis = HDTDocument(PATH_DIS)

PATH_LOD = "/scratch/wbeek/data/LOD-a-lot/data.hdt"
hdt_lod_a_lot = HDTDocument(PATH_LOD)

sum_num_entities = 0
total_num_unknown = 0

total_annotated_disambiguation = 0
total_typed_disambiguation = 0

total_collect_annotated_disambiguation = set()
total_collect_annotated_disambiguation_or_unknown = set()
total_collect_typed_disambiguation = set()
total_collect_entities = set()

use_template = "http://dbpedia.org/property/wikiPageUsesTemplate"
dbr_disambig = "http://dbpedia.org/resource/Template:Disambig" #105043
dbr_disambiguation = "http://dbpedia.org/resource/Template:Disambiguation" # 54271



pairs_dis = [
("http://it.dbpedia.org/property/wikiPageUsesTemplate","http://it.dbpedia.org/resource/Template:Disambigua"),
("http://es.dbpedia.org/property/wikiPageUsesTemplate","http://es.dbpedia.org/resource/Plantilla:Desambiguación"),
("http://ru.dbpedia.org/property/wikiPageUsesTemplate","http://ru.dbpedia.org/resource/Шаблон:Неоднозначность"),
("http://sr.dbpedia.org/property/wikiPageUsesTemplate","http://sr.dbpedia.org/resource/Шаблон:Вишезначна_одредница"),
("http://sco.dbpedia.org/property/wikiPageUsesTemplate","http://sco.dbpedia.org/resource/Template:Disambig"),
("http://fa.dbpedia.org/property/wikiPageUsesTemplate","http://fa.dbpedia.org/resource/الگو:ابهام_زدایی"),
("http://fr.dbpedia.org/property/wikiPageUsesTemplate","http://fr.dbpedia.org/resource/Modèle:Homonymie"),
("http://de.dbpedia.org/property/wikiPageUsesTemplate","http://de.dbpedia.org/resource/Vorlage:Begriffsklärung"),
("http://no.dbpedia.org/property/wikiPageUsesTemplate","http://no.dbpedia.org/resource/Mal:Pekerside"),
("http://lt.dbpedia.org/property/wikiPageUsesTemplate","http://lt.dbpedia.org/resource/Šablonas:Disambig"),
("http://sv.dbpedia.org/property/wikiPageUsesTemplate","http://sv.dbpedia.org/resource/Mall:Gren"),
("http://ja.dbpedia.org/property/wikiPageUsesTemplate","http://ja.dbpedia.org/resource/Template:Aimai"),
("http://pt.dbpedia.org/property/wikiPageUsesTemplate","http://pt.dbpedia.org/resource/Predefinição:Disambig"),
("http://pl.dbpedia.org/property/wikiPageUsesTemplate","http://pl.dbpedia.org/resource/Szablon:Ujednoznacznienie"),
("http://hy.dbpedia.org/property/wikiPageUsesTemplate","http://hy.dbpedia.org/resource/Կաղապար:Բի"),
("http://uk.dbpedia.org/property/wikiPageUsesTemplate","http://uk.dbpedia.org/resource/Шаблон:DisambigG"),
("http://ca.dbpedia.org/property/wikiPageUsesTemplate","http://ca.dbpedia.org/resource/Plantilla:Desambiguació")
]


prefix_ct = Counter()
prefix_ct_unknown = Counter()
annotation = {}
for id in gs:
	# print ('reading ', id)
	filename = str(id) +'.tsv'
	entries = read_file(filename)
	sum_num_entities += len (entries)
	print ('\n***********************\n', id, ' has ', len (entries), ' entities')
	count_unknown = 0
	collect_annotated_disambiguation = set()
	collect_typed_disambiguation = set()
	for row in entries:
		e = row[0]
		a = row[1]
		c = row[2] # comment
		annotation [e] = a
		total_collect_entities.add(e)

		if a == 'unknown':
			count_unknown += 1
		if c == 'disambiguation':
			collect_annotated_disambiguation.add(e)
		if c == 'disambiguation' or a == 'unknown':
			total_collect_annotated_disambiguation_or_unknown.add(e)


		triples, cardinality = hdt_lod_a_lot.search_triples(e, use_template, dbr_disambig)
		if cardinality > 0:
			collect_typed_disambiguation.add(e)

		triples, cardinality = hdt_lod_a_lot.search_triples(e, use_template, dbr_disambiguation)
		if cardinality > 0:
			collect_typed_disambiguation.add(e)

		pairs_dis
		for (p, o) in pairs_dis:
			triples, cardinality = hdt_lod_a_lot.search_triples(e, p, o)
			if cardinality > 0:
				collect_typed_disambiguation.add(e)

	print ('it has ', len(collect_annotated_disambiguation), ' annotated disambiguation entities')
	print ('it has ', len(collect_typed_disambiguation), ' typed disambiguation entities')

	total_num_unknown += count_unknown
	total_collect_annotated_disambiguation = total_collect_annotated_disambiguation.union(collect_annotated_disambiguation)
	total_collect_typed_disambiguation = total_collect_typed_disambiguation.union(collect_typed_disambiguation)

	anno_only = collect_annotated_disambiguation.difference(collect_typed_disambiguation)
	print('\tannotated only ', len(anno_only))

	# for t in anno_only:
	# 	print ('anno only ',t, ' with annotation ', annotation[t])

	type_only = collect_typed_disambiguation.difference(collect_annotated_disambiguation)
	# for t in type_only:
	# 	print (t, ' with annotation ', annotation[t])

	print('\ttyped only ', len(type_only))
	for t in type_only:
		print ('typed only', t, ' with annotation ', annotation[t])

	print ('\tmutual :', len (collect_typed_disambiguation.intersection(collect_annotated_disambiguation)))

print ('there are ', sum_num_entities , ' entities in 28 files')
print ('there are in total ', total_num_unknown, ' unknown entities (by annotation)')

total_annotated_disambiguation = len(total_collect_annotated_disambiguation)
total_typed_disambiguation = len(total_collect_typed_disambiguation)


print ('there are in total ', total_annotated_disambiguation, ' disambiguation entities (by annotation)')
print ('there are in total ', total_typed_disambiguation, ' disambiguation entities (by type)')

print ('typed dis', len (total_collect_typed_disambiguation))

all_typed_only = total_collect_typed_disambiguation.difference(total_collect_annotated_disambiguation)
print('in total ',len (all_typed_only), ' typed only')

all_annotated_only = total_collect_annotated_disambiguation.difference(total_collect_typed_disambiguation)
print('in total ',len (all_annotated_only), ' annotated only')

all_mutual =total_collect_annotated_disambiguation.intersection(total_collect_typed_disambiguation)
print ('mutual: ', len (all_mutual))
print ('*'*30)

# total_collect_annotated_disambiguation_or_unknown

print ('typed dis', len (total_collect_typed_disambiguation))
all_typed_only = total_collect_typed_disambiguation.difference(total_collect_annotated_disambiguation_or_unknown)
print('in total ',len (all_typed_only), ' typed only')
# print (all_typed_only)

all_annotated_only = total_collect_annotated_disambiguation_or_unknown.difference(total_collect_typed_disambiguation)
print('in total ',len (all_annotated_only), ' annotated only (or unknown)')

all_mutual =total_collect_annotated_disambiguation_or_unknown.intersection(total_collect_typed_disambiguation)
print ('mutual: ', len (all_mutual))
print ('*'*30)


print ('multilingual: ', len(pairs_dis))

count_type = Counter()
for n in total_collect_annotated_disambiguation:
	# find out the type of this node
	triples, cardinality = hdt_lod_a_lot.search_triples(n, rdf_type, "")
	for (_,_, t) in triples:
		count_type[t] += 1

print ('count type of these disambiguation nodes')
for t in count_type:
	print (t, ' has ', count_type[t])


# ct_type = Counter()
# rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

#
# # I want to test the following:
# # 1. what are their (most common) types?
# print ('among all the ', len (total_collect_annotated_disambiguation), 'annotated disambiguation entities')
# for n in total_collect_annotated_disambiguation:
# 	triples, cardinality = hdt_lod_a_lot.search_triples(n, rdf_type, "")
# 	for (_,_,t) in triples:
# 		ct_type [t] += 1
#
# for c in ct_type:
# 	print (c, ' -<type>- ', ct_type[c])
#
# use_template = "http://dbpedia.org/property/wikiPageUsesTemplate"
# dbr_disambig = "http://dbpedia.org/resource/Template:Disambig" #105043
# dbr_disambiguation = "http://dbpedia.org/resource/Template:Disambiguation" # 54271
#
# print('*'*20)
# template_used = Counter()
# # 2. what are the templates they used?
# for n in total_collect_annotated_disambiguation:
# 	triples, cardinality = hdt_lod_a_lot.search_triples(n, use_template, "")
# 	for (_,_,t) in triples:
# 		template_used [t] += 1
#
# for t in template_used:
# 	print (t, ' -<templates>- ', template_used[t])
#
# print('*'*20)
# predicate_used = Counter()
# # 2. what are the templates they used?
# for n in total_collect_annotated_disambiguation:
# 	triples, cardinality = hdt_lod_a_lot.search_triples(n, "", "")
# 	for (_,p,_) in triples:
# 		predicate_used [p] += 1
#
# for t in predicate_used:
# 	print (t, ' -<predicate>- ', predicate_used[t])
#
# print ('*****<<<wikiPageUsesTemplate>>>>*****')
# # wikiPageUsesTemplate
# for n in total_collect_annotated_disambiguation:
# 	print ('for n = ', n)
# 	triples, cardinality = hdt_lod_a_lot.search_triples(n, "", "")
# 	for (_,p,o) in triples:
# 		if 'wikiPageUsesTemplate' in p:
# 			print ('\tp = ', p)
# 			print ('\to = ', o)

#
# # among all the  161  entities
# # http://www.wikidata.org/ontology#Item  -<type>-  10
# # http://dbpedia.org/resource/German_surnames  -<type>-  1
# # http://dbpedia.org/ontology/GivenName  -<type>-  1
# # http://dbpedia.org/ontology/Name  -<type>-  1
# # http://dbpedia.org/ontology/Place  -<type>-  2
# # http://dbpedia.org/ontology/PopulatedPlace  -<type>-  2
# # http://dbpedia.org/ontology/Settlement  -<type>-  2
# # http://schema.org/Place  -<type>-  2
# # http://www.w3.org/2002/07/owl#Thing  -<type>-  2
# # http://dbpedia.org/class/yago/ChineseSingers  -<type>-  1
# # http://dbpedia.org/class/yago/Tibetans  -<type>-  1
# # http://dbpedia.org/resource/Chinese_singers  -<type>-  1
# # ********************
# # http://dbpedia.org/resource/Template:Geodis  -<templates>-  3
# # http://dbpedia.org/resource/Template:Disambig  -<templates>-  9
# # http://dbpedia.org/resource/Template:R_from_title_without_diacritics  -<templates>-  1
# # http://dbpedia.org/resource/Template:R_from_misspelling  -<templates>-  1
# # http://dbpedia.org/resource/Template:R_with_possibilities  -<templates>-  1
# # http://dbpedia.org/resource/Template:Disambiguation  -<templates>-  6
# # http://dbpedia.org/resource/Template:SMS  -<templates>-  1
# # http://dbpedia.org/resource/Template:Sclass-  -<templates>-  1
# # http://dbpedia.org/resource/Template:Ship  -<templates>-  1
# # http://dbpedia.org/resource/Template:USS  -<templates>-  1
# # http://dbpedia.org/resource/Template:surname  -<templates>-  1
# # http://dbpedia.org/resource/Template:TOC_right  -<templates>-  2
# # http://dbpedia.org/resource/Template:Infobox_Nome  -<templates>-  1
# # http://dbpedia.org/resource/Template:Surname  -<templates>-  1
# # http://dbpedia.org/resource/Template:Wiktionarypar  -<templates>-  1
# # http://dbpedia.org/resource/Template:wiktionarypar  -<templates>-  1
# # http://dbpedia.org/resource/Template:Wiktionary  -<templates>-  2
# # ********************
# # http://dbpedia.org/ontology/abstract  -<predicate>-  130
# # http://dbpedia.org/ontology/wikiPageID  -<predicate>-  103
# # http://dbpedia.org/ontology/wikiPageInterLanguageLink  -<predicate>-  2029
# # http://dbpedia.org/ontology/wikiPageOutLinkCount  -<predicate>-  87
# # http://dbpedia.org/ontology/wikiPageRevisionID  -<predicate>-  191
# # http://dbpedia.org/ontology/wikiPageWikiLink  -<predicate>-  2289
# # http://lt.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  9
# # http://www.w3.org/2000/01/rdf-schema#comment  -<predicate>-  197
# # http://www.w3.org/2000/01/rdf-schema#label  -<predicate>-  835
# # http://www.w3.org/2002/07/owl#sameAs  -<predicate>-  2864
# # http://www.w3.org/ns/prov#wasDerivedFrom  -<predicate>-  191
# # http://xmlns.com/foaf/0.1/isPrimaryTopicOf  -<predicate>-  100
# # http://dbpedia.org/ontology/wikiPageDisambiguates  -<predicate>-  781
# # http://dbpedia.org/ontology/wikiPageInLinkCount  -<predicate>-  70
# # http://pl.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  4
# # http://ja.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  2
# # http://schema.org/description  -<predicate>-  265
# # http://www.w3.org/1999/02/22-rdf-syntax-ns#type  -<predicate>-  26
# # http://www.w3.org/2004/02/skos/core#altLabel  -<predicate>-  57
# # http://www.wikidata.org/entity/P31s  -<predicate>-  10
# # http://bg.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  1
# # http://dbpedia.org/ontology/wikiPageLength  -<predicate>-  21
# # http://dbpedia.org/ontology/wikiPageOutDegree  -<predicate>-  21
# # http://dbpedia.org/ontology/wikiPageRedirects  -<predicate>-  7
# # http://dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  34
# # http://cs.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  3
# # http://hu.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  1
# # http://ru.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  9
# # http://www.w3.org/2000/01/rdf-schema#seeAlso  -<predicate>-  139
# # http://nl.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  13
# # http://es.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  6
# # http://sv.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  2
# # http://sh.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  1
# # http://dbpedia.org/property/hasPhotoCollection  -<predicate>-  13
# # http://dbpedia.org/property/wikilink  -<predicate>-  1965
# # http://xmlns.com/foaf/0.1/page  -<predicate>-  119
# # http://de.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  16
# # http://purl.org/dc/terms/subject  -<predicate>-  60
# # http://af.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  1
# # http://dbpedia.org/property/redirect  -<predicate>-  11
# # http://br.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  4
# # http://it.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  9
# # http://pt.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  8
# # http://dbpedia.org/ontology/description  -<predicate>-  205
# # http://hy.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  1
# # http://ro.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  2
# # http://lmo.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  1
# # http://ca.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  3
# # http://ko.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  7
# # http://fr.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  20
# # http://dbpedia.org/property/abstract  -<predicate>-  70
# # http://dbpedia.org/property/surnameProperty  -<predicate>-  2
# # http://www.w3.org/2004/02/skos/core#subject  -<predicate>-  6
# # http://fa.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  3
# # http://kk.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  4
# # http://dbpedia.org/ontology/alias  -<predicate>-  23
# # http://www.wikidata.org/entity/P948s  -<predicate>-  1
# # http://dbpedia.org/property/disambiguates  -<predicate>-  168
# # http://id.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  2
# # http://no.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  1
# # http://xmlns.com/foaf/0.1/depiction  -<predicate>-  152
# # http://vi.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  1
# # http://sr.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  1
# # http://dbpedia.org/ontology/wikiPageExternalLink  -<predicate>-  7
# # http://simple.dbpedia.org/property/almaMater  -<predicate>-  1
# # http://simple.dbpedia.org/property/appointed  -<predicate>-  1
# # http://simple.dbpedia.org/property/birthDate  -<predicate>-  1
# # http://simple.dbpedia.org/property/birthPlace  -<predicate>-  3
# # http://simple.dbpedia.org/property/children  -<predicate>-  6
# # http://simple.dbpedia.org/property/country  -<predicate>-  1
# # http://simple.dbpedia.org/property/deathDate  -<predicate>-  1
# # http://simple.dbpedia.org/property/deathPlace  -<predicate>-  3
# # http://simple.dbpedia.org/property/footnotes  -<predicate>-  1
# # http://simple.dbpedia.org/property/ministerFrom  -<predicate>-  1
# # http://simple.dbpedia.org/property/name  -<predicate>-  1
# # http://simple.dbpedia.org/property/nationality  -<predicate>-  1
# # http://simple.dbpedia.org/property/office  -<predicate>-  8
# # http://simple.dbpedia.org/property/order  -<predicate>-  4
# # http://simple.dbpedia.org/property/party  -<predicate>-  2
# # http://simple.dbpedia.org/property/predecessor  -<predicate>-  3
# # http://simple.dbpedia.org/property/president  -<predicate>-  2
# # http://simple.dbpedia.org/property/profession  -<predicate>-  1
# # http://simple.dbpedia.org/property/religion  -<predicate>-  1
# # http://simple.dbpedia.org/property/restingplace  -<predicate>-  2
# # http://simple.dbpedia.org/property/signature  -<predicate>-  1
# # http://simple.dbpedia.org/property/signatureAlt  -<predicate>-  1
# # http://simple.dbpedia.org/property/size  -<predicate>-  2
# # http://simple.dbpedia.org/property/spouse  -<predicate>-  1
# # http://simple.dbpedia.org/property/successor  -<predicate>-  6
# # http://simple.dbpedia.org/property/termEnd  -<predicate>-  7
# # http://simple.dbpedia.org/property/termStart  -<predicate>-  7
# # http://simple.dbpedia.org/property/vicepresident  -<predicate>-  1
# # http://simple.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  11
# # http://simple.dbpedia.org/resource/Template:Infobox_President  -<predicate>-  21
# # http://ar.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  1
# # http://tl.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  1
# # http://uz.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  3
# # http://uk.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  3
# # http://dbpedia.org/ontology/subtitle  -<predicate>-  1
# # http://dbpedia.org/ontology/thumbnail  -<predicate>-  2
# # http://dbpedia.org/property/g_percent_C3_percent_AAnero  -<predicate>-  1
# # http://dbpedia.org/property/gênero  -<predicate>-  1
# # http://dbpedia.org/property/imagem  -<predicate>-  1
# # http://dbpedia.org/property/legenda  -<predicate>-  1
# # http://dbpedia.org/property/nome  -<predicate>-  1
# # http://xmlns.com/foaf/0.1/name  -<predicate>-  1
# # http://sco.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  1
# # http://dbpedia.org/property/geburtsort  -<predicate>-  1
# # http://dbpedia.org/property/commonLanguages  -<predicate>-  1
# # http://dbpedia.org/property/commonName  -<predicate>-  1
# # http://dbpedia.org/property/continent  -<predicate>-  1
# # http://dbpedia.org/property/conventionalLongName  -<predicate>-  1
# # http://dbpedia.org/property/eventEnd  -<predicate>-  1
# # http://dbpedia.org/property/eventStart  -<predicate>-  1
# # http://dbpedia.org/property/flagP  -<predicate>-  1
# # http://dbpedia.org/property/gouarnamantType  -<predicate>-  1
# # http://dbpedia.org/property/imageMap  -<predicate>-  1
# # http://dbpedia.org/property/kerbenn  -<predicate>-  1
# # http://dbpedia.org/property/p  -<predicate>-  1
# # http://dbpedia.org/property/s  -<predicate>-  1
# # http://dbpedia.org/property/wiktionaryparProperty  -<predicate>-  2
# # http://dbpedia.org/property/yearEnd  -<predicate>-  1
# # http://dbpedia.org/property/yearStart  -<predicate>-  1
# # http://et.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  1
# # http://fi.dbpedia.org/property/wikiPageUsesTemplate  -<predicate>-  1
#
# d = "http://dbpedia.org/ontology/wikiPageDisambiguates"
# f = "http://dbpedia.org/property/disambiguates"
#
# # Test 3: if there is an d/f edge between two nodes, then they are different
# collect_d_edges = set()
# num_d_refl = 0
# for n in total_collect_entities:
# 	triples, cardinality = hdt_lod_a_lot.search_triples(n, d, "")
# 	for (_,_,m) in triples:
# 		if m in total_collect_entities:
# 			if n!=m:
# 				collect_d_edges.add((n, m))
# 			else:
# 				num_d_refl += 1
#
# print ('there are in total ', len(collect_d_edges), ' d edges')
# print ('and an additional (stupid) ', num_d_refl, ' edges')
#
# count_d_diff = 0
# count_d_same = 0
# for (n,m) in collect_d_edges:
# 	if annotation[n] != annotation[m] and annotation[n] != 'unknown' and annotation[m] != 'unknown':
# 		count_d_diff += 1
# 	if annotation[n] == annotation[m] and annotation[n] != 'unknown' and annotation[m] != 'unknown':
# 		count_d_same += 1
# 		print ('d-', annotation[n] ,': ',n, '<-sameas->', m)
#
# print ('count_d_diff', count_d_diff)
# print ('count_d_same', count_d_same)
#
#
# collect_f_edges = set()
# num_f_refl = 0
# for n in total_collect_entities:
# 	triples, cardinality = hdt_lod_a_lot.search_triples(n, f, "")
# 	for (_,_,m) in triples:
# 		if m in total_collect_entities:
# 			if n!=m:
# 				collect_f_edges.add((n, m))
# 			else:
# 				num_f_refl += 1
#
# print ('there are in total ', len(collect_f_edges), ' f edges')
# print ('and an additional (stupid) ', num_f_refl, ' edges')
#
# count_f_diff = 0
# count_f_same = 0
# for (n,m) in collect_f_edges:
# 	if annotation[n] != annotation[m] and annotation[n] != 'unknown' and annotation[m] != 'unknown':
# 		count_f_diff += 1
# 	if annotation[n] == annotation[m] and annotation[n] != 'unknown' and annotation[m] != 'unknown':
# 		count_f_same += 1
# 		print ('f-', annotation[n], ': ', n, '<-sameas->', m)
#
# print ('count_f_diff', count_f_diff)
# print ('count_f_same', count_f_same)



# **

# print('d'*20)
# collect_d = set()
# # 2. How many has d and f?
# print ('out of all ', len (total_collect_entities), ' entities')
# for n in total_collect_entities:
# 	triples, cardinality = hdt_lod_a_lot.search_triples(n, d, "")
# 	if cardinality>0:
# 		collect_d.add(n)
# print ('there are d:', len (collect_d), ' found')
# print ('intersection with annotated', len(collect_d.intersection(total_collect_annotated_disambiguation)))
# print ('intersection with annotated or unknown', len(collect_d.intersection(total_collect_annotated_disambiguation_or_unknown)))
# # total_collect_annotated_disambiguation_or_unknown
#
# for d in collect_d:
# 	print (d, 'has annotation', annotation[d])
#
# print('f'*20)
# collect_f = set()
# # 2. How many has d and f?
# print ('out of all ', len (total_collect_entities), ' entities')
# for n in total_collect_entities:
# 	triples, cardinality = hdt_lod_a_lot.search_triples(n, f, "")
# 	if cardinality>0:
# 		collect_f.add(n)
# print ('there are f:', len (collect_f), ' found')
# print ('intersection with annotated', len(collect_f.intersection(total_collect_annotated_disambiguation)))
# print ('intersection with annotated or unknown', len(collect_f.intersection(total_collect_annotated_disambiguation_or_unknown)))
# # total_collect_annotated_disambiguation_or_unknown
# for f in collect_f:
# 	print (f, 'has annotation', annotation[f])
