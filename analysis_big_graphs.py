# this file takes the annotated connected components and generate
# the edges of the graph (connected component)
#
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
from rfc3987 import  parse
import urllib.parse
import gzip
from extend_metalink import *
import requests
from requests.exceptions import Timeout
from SameAsEqGraph import get_simp_IRI, get_namespace, get_name
import pymetis


PATH_LOD = "/scratch/wbeek/data/LOD-a-lot/data.hdt"
hdt_lod_a_lot = HDTDocument(PATH_LOD)

PATH_META = "/home/jraad/ssd/data/identity/metalink/metalink-2/metalink-2.hdt"
hdt_metalink = HDTDocument(PATH_META)

PATH_DIS = "sameas_disambiguation_entities.hdt"
hdt_dis = HDTDocument(PATH_DIS)


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



def load_big_graphs_info (file_name):
	# Index   Size    Entities_without_literals
	nodes_file = open(file_name, 'r')
	reader = csv.DictReader(nodes_file, delimiter='\t',)
	collect_index = {}
	for row in reader:
		s = int(row["Index"])
		n = int(row["Entities_without_literals"])
		collect_index[s] = n
	return collect_index

# load the files
def load_graph (nodes_filename):
	g = nx.Graph()
	nodes_file = open(nodes_filename, 'r')
	reader = csv.DictReader(nodes_file, delimiter='\t',)
	for row in reader:
		s = row["Entity"]
		# a = row["Annotation"]
		# c = row["Comment"]
		g.add_node(s)
	return g


def obtain_edges(g):
	for n in g.nodes():
		(triples, cardi) = hdt_lod_a_lot.search_triples(n, sameas, "")
		for (_,_,o) in triples:
			if o in g.nodes():
				if n != o:
					g.add_edge(n, o)
		(triples, cardi) = hdt_lod_a_lot.search_triples("", sameas, n)
		for (s,_,_) in triples:
			if s in g.nodes():
				if s != n:
					g.add_edge(s, n)
	return g



def find_statement_id(subject, object):

	triples, cardinality = hdt_metalink.search_triples("", rdf_subject, subject)
	collect_statement_id_regarding_subject = set()

	for (s,p,o) in triples:
		collect_statement_id_regarding_subject.add(str(s))

	triples, cardinality = hdt_metalink.search_triples("", rdf_object, object)

	collect_statement_id_regarding_object = set()

	for (s,p,o) in triples:
		collect_statement_id_regarding_object.add(str(s))

	inter_section = collect_statement_id_regarding_object.intersection(collect_statement_id_regarding_subject)

	# do it the reverse way: (object, predicate, subject)
	triples, cardinality = hdt_metalink.search_triples("", rdf_object, subject)
	collect_statement_id_regarding_subject = set()

	for (s,p,o) in triples:
		collect_statement_id_regarding_subject.add(str(s))

	triples, cardinality = hdt_metalink.search_triples("", rdf_subject, object)

	collect_statement_id_regarding_object = set()

	for (s,p,o) in triples:
		collect_statement_id_regarding_object.add(str(s))

	inter_section2 = collect_statement_id_regarding_object.intersection(collect_statement_id_regarding_subject)

	if len (inter_section) >= 1:
		return list(inter_section)[0] #
	elif len (inter_section2) >= 1:
		# print ('\nfound one in reverse!: \n', subject, '\t', object)
		return list(inter_section2)[0] #:
	else:
		return None



def export_graph_edges (file_name, graph):
	file =  open(file_name, 'w', newline='')
	writer = csv.writer(file, delimiter='\t')
	writer.writerow([ "SUBJECT", "OBJECT", "METALINK_ID"])
	for (l, r) in graph.edges:
		if graph.edges[l, r]['metalink_id'] == None:
			writer.writerow([l, r, 'None'])
		else:
			writer.writerow([l, r, graph.edges[l, r]['metalink_id']])



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


pairs_prefix = [
('http://ko.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/'),
('http://simple.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/'),
('http://dbpedia.org/resource/', 'http://sw.opencyc.org/2008/06/10/concept/'),
('http://data.nytimes.com/', 'http://dbpedia.org/resource/'),
('http://simple.dbpedia.org/resource/', 'http://yi.dbpedia.org,/resource/'),
('http://ko.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/'),
('http://ml.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/'),
('http://yi.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/')]

# 10 edges at least, 10% error rate
paris_prefix_more = [
('http://da.dbpedia.org/resource/', 'http://ka.dbpedia.org/resource/') ,
('http://dbpedia.org/resource/', 'http://de.dbpedia.org/resource/') ,
('http://eu.dbpedia.org/resource/', 'http://th.dbpedia.org/resource/') ,
('http://de.dbpedia.org/resource/', 'http://pt.dbpedia.org/resource/') ,
('http://dbpedia.org/resource/', 'http://nl.dbpedia.org/resource/') ,
('http://kn.dbpedia.org/resource/', 'http://no.dbpedia.org/resource/') ,
('http://jv.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://oc.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://mr.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://fy.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://hr.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ku.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://fy.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://gd.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://sq.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://sq.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://qu.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://uz.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://br.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://an.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://nn.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ka.dbpedia.org/resource/', 'http://ka.dbpedia.org/resource/') ,
('http://kn.dbpedia.org/resource/', 'http://th.dbpedia.org/resource/') ,
('http://ko.dbpedia.org/resource/', 'http://tg.dbpedia.org/resource/') ,
('http://io.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://af.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ia.dbpedia.org/resource/', 'http://pnb.dbpedia.org/resource/') ,
('http://be-x-old.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://kk.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://is.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://jv.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://pnb.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://war.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://sw.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ia.dbpedia.org/resource/', 'http://la.dbpedia.org/resource/') ,
('http://cy.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://az.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://hu.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://yi.dbpedia.org/resource/', 'http://zh-min-nan.dbpedia.org/resource/') ,
('http://ko.dbpedia.org/resource/', 'http://sw.dbpedia.org/resource/') ,
('http://br.dbpedia.org/resource/', 'http://dbpedia.org/resource/') ,
('http://ku.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ka.dbpedia.org/resource/', 'http://th.dbpedia.org/resource/') ,
('http://lb.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://vec.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://dbpedia.org/resource/', 'http://nds.dbpedia.org/resource/') ,
('http://hy.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://simple.dbpedia.org/resource/', 'http://tg.dbpedia.org/resource/') ,
('http://arz.dbpedia.org/resource/', 'http://ko.dbpedia.org/resource/') ,
('http://be-x-old.dbpedia.org/resource/', 'http://ia.dbpedia.org/resource/') ,
('http://mr.dbpedia.org/resource/', 'http://ne.dbpedia.org/resource/') ,
('http://lmo.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ga.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://arz.dbpedia.org/resource/', 'http://simple.dbpedia.org/resource/') ,
('http://kk.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ga.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://vec.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://da.dbpedia.org/resource/', 'http://dbpedia.org/resource/') ,
('http://ceb.dbpedia.org/resource/', 'http://dbpedia.org/resource/') ,
('http://fr.dbpedia.org/resource/', 'http://nds.dbpedia.org/resource/') ,
('http://be-x-old.dbpedia.org/resource/', 'http://be-x-old.dbpedia.org/resource/') ,
('http://als.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ia.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://bn.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ur.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://sco.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ht.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://scn.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ia.dbpedia.org/resource/', 'http://lb.dbpedia.org/resource/') ,
('http://ast.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://bat-smg.dbpedia.org/resource/', 'http://da.dbpedia.org/resource/') ,
('http://gd.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://de.dbpedia.org/resource/', 'http://ia.dbpedia.org/resource/') ,
('http://diq.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://my.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://nds.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://als.dbpedia.org/resource/', 'http://vo.dbpedia.org/resource/') ,
('http://mr.dbpedia.org/resource/', 'http://new.dbpedia.org/resource/') ,
('http://ta.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://dbpedia.org/resource/', 'http://www4.wiwiss.fu-berlin.de/flickrwrappr/photos/') ,
('http://dbpedia.org/resource/', 'http://sw.opencyc.org/concept/') ,
('http://ko.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://wordnet.rkbexplorer.com/id/', 'http://www.w3.org/2006/03/wn/wn20/instances/') ,
('http://simple.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://dbpedia.org/resource/', 'http://sw.opencyc.org/2008/06/10/concept/') ,
('http://ko.dbpedia.org/resource/', 'http://ne.dbpedia.org/resource/') ,
('http://ast.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://lmo.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ia.dbpedia.org/resource/', 'http://nds.dbpedia.org/resource/') ,
('http://ia.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://tg.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://als.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://hi.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://sco.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://tt.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://am.dbpedia.org/resource/', 'http://simple.dbpedia.org/resource/') ,
('http://am.dbpedia.org/resource/', 'http://ko.dbpedia.org/resource/') ,
('http://arz.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://yo.dbpedia.org/resource/', 'http://zh-yue.dbpedia.org/resource/') ,
('http://pms.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://data.nytimes.com/', 'http://dbpedia.org/resource/') ,
('http://ba.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ceb.dbpedia.org/resource/', 'http://fr.dbpedia.org/resource/') ,
('http://ceb.dbpedia.org/resource/', 'http://it.dbpedia.org/resource/') ,
('http://vo.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://am.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://simple.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://mg.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ba.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://mzn.dbpedia.org/resource/', 'http://simple.dbpedia.org/resource/') ,
('http://ceb.dbpedia.org/resource/', 'http://es.dbpedia.org/resource/') ,
('http://ko.dbpedia.org/resource/', 'http://mzn.dbpedia.org/resource/') ,
('http://ceb.dbpedia.org/resource/', 'http://de.dbpedia.org/resource/') ,
('http://bat-smg.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ne.dbpedia.org/resource/', 'http://simple.dbpedia.org/resource/') ,
('http://ceb.dbpedia.org/resource/', 'http://www.wikidata.org/entity/') ,
('http://ko.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ml.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://yi.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/')
]

pair_prefix_10_pct = [
('http://tr.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://da.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://eu.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://gd.dbpedia.org/resource/', 'http://simple.dbpedia.org/resource/') ,
('http://www.wikidata.org/entity/', 'http://yi.dbpedia.org/resource/') ,
('http://nds.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://sl.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://gd.dbpedia.org/resource/', 'http://ko.dbpedia.org/resource/') ,
('http://es.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ru.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://hr.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://fa.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://az.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://sh.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://an.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://mk.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://et.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://vi.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://it.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ia.dbpedia.org/resource/', 'http://vo.dbpedia.org/resource/') ,
('http://pt.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://io.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ko.dbpedia.org/resource/', 'http://my.dbpedia.org/resource/') ,
('http://he.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ro.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://sk.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ms.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://eo.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://is.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://commons.dbpedia.org/resource/', 'http://sr.dbpedia.org/resource/') ,
('http://th.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://el.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://sr.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://id.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://bs.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://lt.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://my.dbpedia.org/resource/', 'http://simple.dbpedia.org/resource/') ,
('http://lv.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://pl.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://be.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://la.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://lv.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://gl.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://uz.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://be.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://af.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://cs.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://nn.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://de.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://fr.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ia.dbpedia.org/resource/', 'http://is.dbpedia.org/resource/') ,
('http://sl.dbpedia.org/resource/', 'http://sl.dbpedia.org/resource/') ,
('http://et.dbpedia.org/resource/', 'http://ia.dbpedia.org/resource/') ,
('http://ka.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://nl.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://hu.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ja.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://lb.dbpedia.org/resource/', 'http://vo.dbpedia.org/resource/') ,
('http://cy.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://dbpedia.org/resource/', 'http://linkedgeodata.org/triplify/node/') ,
('http://nds.dbpedia.org/resource/', 'http://vo.dbpedia.org/resource/') ,
('http://pnb.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://uk.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://fi.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://oc.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ia.dbpedia.org/resource/', 'http://ka.dbpedia.org/resource/') ,
('http://is.dbpedia.org/resource/', 'http://is.dbpedia.org/resource/') ,
('http://ar.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://bg.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://sv.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://yi.dbpedia.org/resource/', 'http://zh.dbpedia.org/resource/') ,
('http://ca.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://be-x-old.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://no.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://tr.dbpedia.org/resource/', 'http://tr.dbpedia.org/resource/') ,
('http://da.dbpedia.org/resource/', 'http://ka.dbpedia.org/resource/') ,
('http://dbpedia.org/resource/', 'http://de.dbpedia.org/resource/') ,
('http://eu.dbpedia.org/resource/', 'http://th.dbpedia.org/resource/') ,
('http://de.dbpedia.org/resource/', 'http://pt.dbpedia.org/resource/') ,
('http://dbpedia.org/resource/', 'http://nl.dbpedia.org/resource/') ,
('http://fy.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://cy.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://io.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ka.dbpedia.org/resource/', 'http://ka.dbpedia.org/resource/') ,
('http://az.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://jv.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://yi.dbpedia.org/resource/', 'http://zh-min-nan.dbpedia.org/resource/') ,
('http://gd.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://war.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ko.dbpedia.org/resource/', 'http://tg.dbpedia.org/resource/') ,
('http://oc.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://an.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://nn.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://kn.dbpedia.org/resource/', 'http://th.dbpedia.org/resource/') ,
('http://sw.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://uz.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://be-x-old.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://fy.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://br.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://qu.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://jv.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://af.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://is.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://sq.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://mr.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://hu.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ku.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://sq.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://kn.dbpedia.org/resource/', 'http://no.dbpedia.org/resource/') ,
('http://hr.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ia.dbpedia.org/resource/', 'http://la.dbpedia.org/resource/') ,
('http://pnb.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ia.dbpedia.org/resource/', 'http://pnb.dbpedia.org/resource/') ,
('http://kk.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ko.dbpedia.org/resource/', 'http://sw.dbpedia.org/resource/') ,
('http://br.dbpedia.org/resource/', 'http://dbpedia.org/resource/') ,
('http://simple.dbpedia.org/resource/', 'http://tg.dbpedia.org/resource/') ,
('http://arz.dbpedia.org/resource/', 'http://simple.dbpedia.org/resource/') ,
('http://dbpedia.org/resource/', 'http://nds.dbpedia.org/resource/') ,
('http://ka.dbpedia.org/resource/', 'http://th.dbpedia.org/resource/') ,
('http://ga.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://lb.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://be-x-old.dbpedia.org/resource/', 'http://ia.dbpedia.org/resource/') ,
('http://lmo.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ku.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ga.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://arz.dbpedia.org/resource/', 'http://ko.dbpedia.org/resource/') ,
('http://mr.dbpedia.org/resource/', 'http://ne.dbpedia.org/resource/') ,
('http://hy.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ne.dbpedia.org/resource/', 'http://ne.dbpedia.org/resource/') ,
('http://vec.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://kk.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://vec.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://da.dbpedia.org/resource/', 'http://dbpedia.org/resource/') ,
('http://fr.dbpedia.org/resource/', 'http://nds.dbpedia.org/resource/') ,
('http://ceb.dbpedia.org/resource/', 'http://dbpedia.org/resource/') ,
('http://be-x-old.dbpedia.org/resource/', 'http://be-x-old.dbpedia.org/resource/') ,
('http://bat-smg.dbpedia.org/resource/', 'http://da.dbpedia.org/resource/') ,
('http://als.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://scn.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://als.dbpedia.org/resource/', 'http://vo.dbpedia.org/resource/') ,
('http://bn.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ur.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ast.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://mr.dbpedia.org/resource/', 'http://new.dbpedia.org/resource/') ,
('http://my.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ht.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://de.dbpedia.org/resource/', 'http://ia.dbpedia.org/resource/') ,
('http://ta.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://sco.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://nds.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ia.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://gd.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://diq.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ia.dbpedia.org/resource/', 'http://lb.dbpedia.org/resource/') ,
('http://dbpedia.org/resource/', 'http://www4.wiwiss.fu-berlin.de/flickrwrappr/photos/') ,
('http://dbpedia.org/resource/', 'http://sw.opencyc.org/concept/') ,
('http://simple.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ko.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://wordnet.rkbexplorer.com/id/', 'http://www.w3.org/2006/03/wn/wn20/instances/') ,
('http://dbpedia.org/resource/', 'http://sw.opencyc.org/2008/06/10/concept/') ,
('http://tt.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://am.dbpedia.org/resource/', 'http://simple.dbpedia.org/resource/') ,
('http://als.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ia.dbpedia.org/resource/', 'http://nds.dbpedia.org/resource/') ,
('http://lmo.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ia.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://yo.dbpedia.org/resource/', 'http://zh-yue.dbpedia.org/resource/') ,
('http://pms.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ast.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://arz.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ko.dbpedia.org/resource/', 'http://ne.dbpedia.org/resource/') ,
('http://sco.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://vo.dbpedia.org/resource/', 'http://vo.dbpedia.org/resource/') ,
('http://tg.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://am.dbpedia.org/resource/', 'http://ko.dbpedia.org/resource/') ,
('http://hi.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://data.nytimes.com/', 'http://dbpedia.org/resource/') ,
('http://ceb.dbpedia.org/resource/', 'http://it.dbpedia.org/resource/') ,
('http://ba.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://am.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ko.dbpedia.org/resource/', 'http://mzn.dbpedia.org/resource/') ,
('http://ba.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://sw.opencyc.org/2008/06/10/concept/', 'http://www.w3.org/2006/03/wn/wn20/instances/') ,
('http://bat-smg.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://simple.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://mg.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://vo.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ceb.dbpedia.org/resource/', 'http://fr.dbpedia.org/resource/') ,
('http://ne.dbpedia.org/resource/', 'http://simple.dbpedia.org/resource/') ,
('http://ceb.dbpedia.org/resource/', 'http://www.wikidata.org/entity/') ,
('http://ceb.dbpedia.org/resource/', 'http://es.dbpedia.org/resource/') ,
('http://mzn.dbpedia.org/resource/', 'http://simple.dbpedia.org/resource/') ,
('http://ceb.dbpedia.org/resource/', 'http://de.dbpedia.org/resource/') ,
('http://ko.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://ml.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ceb.dbpedia.org/resource/', 'http://simple.dbpedia.org/resource/') ,
('http://yi.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://data.nobelprize.org/resource/city/', 'http://dbpedia.org/resource/') ,
('http://fa.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ceb.dbpedia.org/resource/', 'http://ko.dbpedia.org/resource/') ,
('http://new.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://dbpedia.org/resource/', 'http://www.ontosearch.com/2008/01/identification/') ,
('http://gu.dbpedia.org/resource/', 'http://mr.dbpedia.org/resource/') ,
('http://kn.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://th.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ca.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ne.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://my.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://te.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://vi.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://no.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://fy.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://fi.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://da.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://nl.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://fr.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ka.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://wa.dbpedia.org/resource/', 'http://zh.dbpedia.org/resource/') ,
('http://sv.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ms.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://simple.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://sk.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ro.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://tg.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ru.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://be.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://he.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://tl.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://el.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://te.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://dbpedia.org/resource/', 'http://sws.geonames.org/3433488/') ,
('http://scn.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://tr.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://arz.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://vec.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://nap.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://lv.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://az.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://af.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://pl.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://et.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://gl.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://jv.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://pt.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://oc.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://mk.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://kk.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://be-x-old.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://cy.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ht.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://bn.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://an.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ar.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://bs.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://hy.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ko.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ceb.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://wa.dbpedia.org/resource/', 'http://yi.dbpedia.org/resource/') ,
('http://tt.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://hu.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://uk.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://sr.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ja.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://eu.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://hi.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ia.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://uz.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://su.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ast.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://sl.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://is.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://id.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ur.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://de.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://sw.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://hr.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ta.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://it.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://pnb.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://wa.dbpedia.org/resource/', 'http://www.wikidata.org/entity/') ,
('http://sq.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://nn.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://wa.dbpedia.org/resource/', 'http://zh-yue.dbpedia.org/resource/') ,
('http://cs.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://lt.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://bg.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://sh.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://kn.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ky.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://eo.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://es.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://io.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://la.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ml.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://nap.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ceb.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://map-bms.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://mr.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://su.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://ne.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ga.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://lb.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://wa.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://gd.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://sco.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://wa.dbpedia.org/resource/', 'http://war.dbpedia.org/resource/') ,
('http://lmo.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://mg.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ky.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://dbpedia.org/resource/', 'http://www.rdfabout.com/rdf/usgov/congress/people/') ,
('http://gu.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ba.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://map-bms.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://als.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://bat-smg.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://diq.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://nds.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://ku.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://am.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://dbpedia.org/resource/Europe/', 'http://www.wikidata.org/entity/') ,
('http://dbpedia.org/resource/Europe/', 'http://wikidata.org/entity/') ,
('http://new.dbpedia.org/resource/', 'http://wa.dbpedia.org/resource/') ,
('http://gu.dbpedia.org/resource/', 'http://yo.dbpedia.org/resource/') ,
('http://dbpedia.org/resource/', 'http://sws.geonames.org/2945358/')
]


def partition_pymetis(graph, num_partitions = 2) :
	print ('partition starts')
	# obtain an index from 0 onwards.
	adjacency = {}

	ele_to_index = {}
	index_to_ele = {}

	index = 0

	for n in graph.nodes():
		if n not in ele_to_index.keys():
			ele_to_index[n] = index
			index_to_ele[index] = n
			index += 1
	# print ('index = ', index)
	# print ('which should be the same as num of nodes = ', graph.number_of_nodes())

	for n in graph.nodes():
		adjacency.setdefault(ele_to_index[n], [])

	for (m,n) in graph.edges():
		adjacency[ele_to_index[m]].append(ele_to_index[n])

	cuts, part_vert  = pymetis.part_graph(num_partitions, adjacency=adjacency)

	# print ('cuts = ', cuts)

	all_edges_to_remove = []
	for (n,m) in graph.edges():
		index_n = ele_to_index[n]
		index_m = ele_to_index[m]
		if part_vert[index_n] != part_vert[index_m]:
			# remove
			all_edges_to_remove.append( (n,m) )
	print ('# edges removed: ', len (all_edges_to_remove))
	return all_edges_to_remove

# sameas_index_to_size_1000.tsv
index_big_graphs = load_big_graphs_info("sameas_index_to_size_1000.tsv")

print ('big graphs are : ', index_big_graphs)
for g_index in list(index_big_graphs.keys())[-2:]: #
	print ('\n\n******')
	print ('index = ', g_index)
	g = load_graph('./big_connected_components/' +str(g_index)+'.tsv')

	bf = g.number_of_nodes()

	g = obtain_edges(g)
	# aft = g.number_of_nodes()
	# print ('there are ', g.number_of_nodes(), ' edges')
	# if aft != bf:
	# 	print ('not the same!')
	# 	print ('before adding edges, there are ', bf)
	# 	print ('before adding edges, there are ', aft)
	print ('The original graph has ', g.number_of_nodes(), ' nodes')
	print ('The original graph has ', g.number_of_edges(), ' edges')


	# step 2: obtain metalink ID:
	# for (l, r) in g.edges():
	# 	meta_id = find_statement_id(l, r)
	# 	if meta_id != None:
	# 		g[l][r]['metalink_id'] = meta_id
	# 	else:
	# 		g[l][r]['metalink_id'] = None

	# #step 3: export the edges and the metalink ID
	# dir = './big_connected_components/'
	# edges_file_name = dir + str(g_index) + '_edges.tsv'
	# print('the export path for edges = ',edges_file_name )
	# export_graph_edges(edges_file_name, g)

	# step 4
	# test if the grpah is connected
	# print('connected (or not):', nx.is_connected(g))

	# find all the disambiguation_entities
	# collect_dis_entities = set()
	# for e in g.nodes():
	# 	(_, cardi) = hdt_dis.search_triples(e, "", "")
	# 	if cardi > 0:
	# 		collect_dis_entities.add(e)
	#
	collect_typed_disambiguation = set()
	for e in g.nodes():
		triples, cardinality = hdt_lod_a_lot.search_triples(e, use_template, dbr_disambig)
		if cardinality > 0:
			collect_typed_disambiguation.add(e)

		triples, cardinality = hdt_lod_a_lot.search_triples(e, use_template, dbr_disambiguation)
		if cardinality > 0:
			collect_typed_disambiguation.add(e)

		# pairs_dis
		for (p, o) in pairs_dis:
			triples, cardinality = hdt_lod_a_lot.search_triples(e, p, o)
			if cardinality > 0:
				collect_typed_disambiguation.add(e)

	collect_dis_entities = collect_typed_disambiguation

	print ('there are in total ', len (collect_dis_entities), ' disambiguation entities')
	# after removing 1720 disambiguation entities: [87601, 2082, 1255, 1173, 1138, 1034, 1003, ...

	# remove them and test the size of connected components
	g.remove_nodes_from(list(collect_dis_entities))

	print ('After removing nodes, the graph has ', g.number_of_nodes(), ' nodes')
	print ('After removing nodes, the graph has ', g.number_of_edges(), ' edges')

	# Then we remove edges from those eight pairs of prefix
	# collect_edges_to_remove_due_to_prefix = set()
	# for e in g.edges():
	# 	(s,t) = e
	# 	ps = get_namespace_prefix(s)
	# 	pt = get_namespace_prefix(t)
	# 	if ps > pt :
	# 		if (pt, ps) in pair_prefix_10_pct:
	# 			collect_edges_to_remove_due_to_prefix.add(e)
	# 	else:
	# 		if (ps, pt) in pair_prefix_10_pct:
	# 			collect_edges_to_remove_due_to_prefix.add(e)
	# print ('after checking against ', len (pair_prefix_10_pct), ' pairs of prefix')
	# print ('count total edges removed due to prefix check: ', len (collect_edges_to_remove_due_to_prefix))

	# prefix_to_check = ["http://wa.dbpedia.org/resource/",
	# 	"http://www4.wiwiss.fu-berlin.de/flickrwrappr/photos/",
	# 	"http://dbpedia.org/resource/Europe/",
	# 	"http://yo.dbpedia.org/resource/",
	# 	"http://yi.dbpedia.org/resource/",
	# 	"http://www.w3.org/2006/03/wn/wn20/instances/"]
	#
	# count_remove_additional = 0
	#
	# nodes_to_remove = set()
	# for n in g.nodes():
	# 	flag = False
	# 	for prefix in prefix_to_check:
	# 		if prefix in n:
	# 			flag = True
	# 	if flag:
	# 		# g.remove_node(n)
	# 		nodes_to_remove.add(n)
	# 		count_remove_additional += 1
	# # nodes_to_remove
	# g.remove_nodes_from(list(nodes_to_remove))
	# print ('removed ', count_remove_additional, ' additional nodes')

	#
	# d = "http://dbpedia.org/ontology/wikiPageDisambiguates"
	# f = "http://dbpedia.org/property/disambiguates"
	#
	# collect_d_edges = set()
	# num_d_refl = 0
	# for n in g.nodes():
	# 	triples, cardinality = hdt_lod_a_lot.search_triples(n, d, "")
	# 	for (_,_,m) in triples:
	# 		if m in g.nodes():
	# 			if n!=m:
	# 				collect_d_edges.add((n, m))
	# 			else:
	# 				num_d_refl += 1
	#
	# collect_f_edges = set()
	# num_f_refl = 0
	# for n in g.nodes():
	# 	triples, cardinality = hdt_lod_a_lot.search_triples(n, f, "")
	# 	for (_,_,m) in triples:
	# 		if m in g.nodes():
	# 			if n!=m:
	# 				collect_f_edges.add((n, m))
	# 			else:
	# 				num_f_refl += 1
	#
	# print ('before removing edges, there are ', g.number_of_edges(), ' edges')
	#
	# print('num d edges = ', len (collect_d_edges))
	# print('num f edges = ', len (collect_f_edges))
	#
	# count_d_removed = 0
	# for (l, r) in collect_d_edges:
	# 	print ('removing', l, r)
	# 	if (l,r) in g.edges():
	# 		g.remove_edge(l, r)
	# 		count_d_removed += 1
	#
	# count_f_removed = 0
	# for (l, r) in collect_f_edges:
	# 	print ('removing', l, r)
	# 	if (l,r) in g.edges():
	# 		g.remove_edge(l, r)
	# 		count_f_removed += 1
	#
	# print ('removed d = ', count_d_removed)
	# print ('removed f = ', count_f_removed)

	# g.remove_edges_from(list(collect_d_edges))
	# g.remove_edges_from(list(collect_f_edges))

	# g.remove_edges_from(list(collect_edges_to_remove_due_to_prefix))

	# print ('after removing edges (due to prefix pair checking) ', g.number_of_nodes(), ' nodes remain')
	# print ('after removing edges (due to prefix pair checking) ', g.number_of_edges(), ' edges remain')

	print ('Now the connnected components are as follows: ')
	component_sizes = [len(c) for c in sorted(nx.connected_components(g), key=len, reverse=True)]
	the_largest_component = g.subgraph([c for c in sorted(nx.connected_components(g), key=len, reverse=True)][0])
	print (component_sizes[:10])
	print('a total of ', len (component_sizes), ' components')
	print ('a sum of ', sum (component_sizes), ' nodes')

	print ('the largest component has ', the_largest_component.number_of_nodes(), ' nodes')
	print ('the largest component has ', the_largest_component.number_of_edges(), ' edges')

	# remove the nodes of indegree or out degree one:
	# collect_nodes_with_degree_one = set()
	# size = -1
	# the_largest_component = nx.Graph(the_largest_component)
	# while (len(collect_nodes_with_degree_one) != size):
	# 	size = len(collect_nodes_with_degree_one)
	# 	for n in the_largest_component.nodes():
	# 		if the_largest_component.degree(n) == 1:
	# 			collect_nodes_with_degree_one.add(n)
	# 	the_largest_component.remove_nodes_from(list(collect_nodes_with_degree_one))
	#
	# print ('collected ', len (collect_nodes_with_degree_one), ' nodes with degree one')
	# print ('the largest component has ', the_largest_component.number_of_nodes(), ' nodes')
	# print ('the largest component has ', the_largest_component.number_of_edges(), ' edges')
	count_cut = 1
	while the_largest_component.number_of_nodes() > 1000:
		print ('cut ', count_cut)
		edges_removed = partition_pymetis(the_largest_component, num_partitions = 2)
		print('num of edges removed = ', len (edges_removed))
		the_largest_component = nx.Graph(the_largest_component)
		the_largest_component.remove_edges_from(edges_removed)

		print ('after partitioning by pymetis, the connnected components are as follows: ')
		component_sizes = [len(c) for c in sorted(nx.connected_components(the_largest_component), key=len, reverse=True)]
		new_largest_component = g.subgraph([c for c in sorted(nx.connected_components(the_largest_component), key=len, reverse=True)][0])
		print (component_sizes[:10])
		print('a total of ', len (component_sizes), ' components')
		print ('a sum of ', sum (component_sizes), ' nodes')
		print ('the largest component has ', new_largest_component.number_of_nodes(), ' nodes')
		print ('the largest component has ', new_largest_component.number_of_edges(), ' edges')
		the_largest_component = new_largest_component
		count_cut += 1

	# print ("*"*30)
	# print ('number of nodes now: ', g.number_of_nodes())
	# print ('number of edges now: ', g.number_of_edges())
