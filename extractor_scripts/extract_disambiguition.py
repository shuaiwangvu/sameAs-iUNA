
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


# all_eq_classes_of_disambig = find_subPropertyOf_eqPropertyOf_closure(set([dbr_disambig]), 1000)
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


with open( "sameas_disambiguation_entities_Nov.nt", 'w') as output:
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


	count_dis = 0
	count_all_language = 0
	for (ut, disamb) in pairs_dis:
		triples, s_cardinality = hdt_lod.search_triples("", ut, disamb)
		count_all_language += s_cardinality
		for (s,_,_) in triples:
			_, cardinality = hdt_metalink.search_triples("", rdf_subject, s)
			_, cardinality2 = hdt_metalink.search_triples("", rdf_object, s)

			if cardinality2 >0 or cardinality >0 :
				count_dis += 1
				writer.writerow(['<'+s+'>', '<'+ut+'>', '<' + disamb + '>', '.'])

	print (count_dis , 'out of ', count_all_language, ' are in the identity graph')
	print ('{:10.2f}'.format(count_dis/count_all_language))
