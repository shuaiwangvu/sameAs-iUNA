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
from urllib.parse import urlparse
import gzip
from extend_metalink import *
import requests
from requests.exceptions import Timeout




PATH_META = "/home/jraad/ssd/data/identity/metalink/metalink-2/metalink-2.hdt"
hdt_metalink = HDTDocument(PATH_META)


hdt_source = HDTDocument("typeA.hdt")
hdt_label = HDTDocument("label_May.hdt")
hdt_comment = HDTDocument("comment_May.hdt")


PATH_LOD = "/scratch/wbeek/data/LOD-a-lot/data.hdt"
hdt_lod_a_lot = HDTDocument(PATH_LOD)

#
gs = [4170, 5723,6617,6927,9411,9756,11116,12745,14872,18688,25604,33122,37544,
39036, 42616,96073,97757,99932,236350,240577,337339,395175,712342,1133953,
1140988,4635725,9994282,14514123]

# gs = [6617]

single = [9411, 9756, 18688, 25604, 96073, 97757, 99932, 337339,
712342, 1133953, 1140988, 9994282]

related = [4170, 5723, 11116]

multiple = [6617, 6927, 12745, 14872, 33122, 37544, 39036, 42616, 236350, 240577,
395175, 4635725, 14514123]


# there are in total 28 entities. 14 each
validate_single = [96073, 712342, 9994282, 18688, 1140988, 25604]
validate_multiple = [6617, 4170, 42616, 39036, 33122, 6927, 11116, 12745]

evaluation_single = [9411, 9756, 97757, 99932, 337339, 1133953]
evaluation_multipel = [5723, 14872, 37544, 236350, 240577, 395175, 4635725, 14514123]




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

def read_file (file_name):
	pairs = []
	eq_file = open(file_name, 'r')
	reader = csv.DictReader(eq_file, delimiter='\t',)
	for row in reader:
		s = row["Entity"]
		o = row["Annotation"]
		pairs.append((s,o))
	return pairs

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


def export_graph_edges (file_name, graph):
	file =  open(file_name, 'w', newline='')
	writer = csv.writer(file, delimiter='\t')
	writer.writerow([ "SUBJECT", "OBJECT", "METALINK_ID"])
	for (l, r) in graph.edges:
		if graph.edges[l, r]['metalink_id'] == None:
			writer.writerow([l, r, 'None'])
		else:
			writer.writerow([l, r, graph.edges[l, r]['metalink_id']])

# type A: explicit sources
def export_explicit_source (file_name, graph):
	count_A = 0
	with open(file_name, 'w') as output:
		for n in graph.nodes:
			triples, cardinality = hdt_source.search_triples(n, "", "")
			for (_, predicate, file) in triples:
				line = '<' + n + '> '
				line += '<' + predicate + '> '
				if str(file)[0] == '"':

					if "^^" in str(file):
						line += '' + file + ' .\n'
						# edited = splited[-2][1:-1] # keep that original one
						# example "http://xmlns.com/foaf/0.1/"^^<http://www.w3.org/2001/XMLSchema#anyURI>
					else: # else, add the following
						line += '' + file + "^^<http://www.w3.org/2001/XMLSchema#string>" + ' .\n'
						# edited = splited[-2][1:-1] + "^^<http://www.w3.org/2001/XMLSchema#string>"

					# print (file, ' has a first char " !!!! ')
				else:
					line += '<' + file + '>. \n'

				output.write(str(line))
				count_A += 1
	print ('count A ', count_A)

# type B: implicit label sources
def export_implicit_label_source (file_name, graph):
	# type B
	count_B = 0
	with open( file_name, 'w') as output:
		for n in graph.nodes:
			triples, cardinality = hdt_label.search_triples(n, "", "")
			for (_, predicate, file) in triples:
				line = '<' + n + '> '
				line += '<' + predicate + '> '
				line += '<' + file + '>. \n'
				output.write(str(line))
				count_B += 1
	print ('count B ', count_B)


# type C: implicit comment sources
def export_implicit_comment_source (file_name, graph):
	count_C = 0
	with open( file_name, 'w') as output:
		for n in graph.nodes():
			triples, cardinality = hdt_comment.search_triples(n, "", "")
			for (_, predicate, file) in triples:
				line = '<' + n + '> '
				line += '<' + predicate + '> '
				line += '<' + file + '>. \n'
				output.write(str(line))
				count_C += 1
	print ('count C ', count_C)


standard_timeout =  (0.01, 0.05)
retry_timeout = (0.5, 2.5)
final_try_timeout = (5, 25)


# standard_timeout =  (0.01, 0.05)
# retry_timeout = (0.05, 0.5)
# final_try_timeout = (0.5, 1)


NOTFOUND = 1
NOREDIRECT = 2
ERROR = 3
TIMEOUT = 4
REDIRECT = 5
# redirected till time out ?


def find_redirects (iri, timeout = standard_timeout):
	try:
		collect_urls = []
		response = requests.get(iri, timeout= timeout, allow_redirects=True)

		if response.status_code == 404:
			# print ('not found', iri)
			return NOTFOUND, None

		if response.history:
			if response.url == iri:
				return NOREDIRECT, None
			else:
				# print("Request was redirected", iri)
				for resp in response.history:
					# print(resp.status_code, resp.url)
					collect_urls.append(resp.url)
				# print("Final destination:")
				# print(response.status_code, response.url)

				collect_urls.append(response.url)
				# print (iri,' was redirected: ', collect_urls)
				return REDIRECT, collect_urls
		else:
			# print("Request was not redirected", iri)
			return NOREDIRECT, None
	except Timeout:
		# print('The request timed out', iri)
		return TIMEOUT, None
	except:
		# print ('error: ', iri)
		return ERROR, None


def obtain_redirect_graph(graph):

	start = time.time()


	redi_graph = nx.DiGraph()

	for e in graph.nodes():
		redi_graph.add_node(e, remark = 'unknown')
	entities_to_test = set( redi_graph.nodes())

	for timeout_parameters in [standard_timeout, retry_timeout, final_try_timeout]:
		timeout_entities = set()
		end_of_redirect_entities = set()
		for e in redi_graph.nodes():
			if redi_graph.nodes[e]['remark'] == 'unknown':
				entities_to_test.add(e)

		for e in entities_to_test:
			# print ('testing ', e)
			# find_redirects
			result, via_entities = find_redirects(e, timeout = timeout_parameters )
			# print ('testing ',e)
			if result == NOTFOUND:
				redi_graph.nodes[e]['remark'] = 'not_found'
			elif result == NOREDIRECT:
				redi_graph.nodes[e]['remark'] = 'not_redirect'
				# print ('mark ',e, ' notredirected')
			elif result == ERROR:
				# print ('why not showing error?')
				redi_graph.nodes[e]['remark'] = 'error'
			elif result == TIMEOUT:
				timeout_entities.add(e)
				redi_graph.nodes[e]['remark'] = 'timeout'
			elif result == REDIRECT:
				redi_graph.nodes[e]['remark'] = 'redirected'
				if len (via_entities) > 1:
					for i, s in enumerate(via_entities[:-1]):
						t = via_entities[i+1]
						if s not in redi_graph.nodes():
							redi_graph.add_node(s, remark = 'redirected')
						else:
							redi_graph.nodes[s]['remark'] = 'redirected'

						if t not in redi_graph.nodes():
							redi_graph.add_node(t, remark = 'unknown')

						redi_graph.add_edge(s, t)

					if via_entities[-1] not in end_of_redirect_entities and via_entities[-1] not in timeout_entities:
						end_of_redirect_entities.add(via_entities[-1])
				else:
					print ('error: ', via_entities)
			else:
				print ('error')

		print ('TIMEOUT: there are ', len (timeout_entities), ' timeout entities')
		print ('End Of Redirect: there are ', len (end_of_redirect_entities), ' end of redirect entities')
		entities_to_test = timeout_entities.union(end_of_redirect_entities)


	for m in end_of_redirect_entities:
		redi_graph.add_node(m, remark = 'timeout')

	update_against = set()
	for n in redi_graph.nodes():
		if redi_graph.nodes[n]['remark'] != 'redirected':
			update_against.add(n)

	# count_redirect_until_timeout = 0
	for n in redi_graph.nodes():
		if redi_graph.nodes[n]['remark'] == 'redirected':
			for m in update_against:
				if nx.has_path(redi_graph, n, m):
					if redi_graph.nodes[m]['remark'] == 'timeout':
						redi_graph.nodes[n]['remark'] = 'redirect_until_timeout'
					elif redi_graph.nodes[m]['remark'] == 'not_found':
						redi_graph.nodes[n]['remark'] = 'redirect_until_not_found'
					elif redi_graph.nodes[m]['remark'] == 'error':
						redi_graph.nodes[n]['remark'] = 'redirect_until_error'
					elif redi_graph.nodes[m]['remark'] == 'not_redirect':
						redi_graph.nodes[n]['remark'] = 'redirect_until_landed'
					else:
						print('Error? reaching m = ', redi_graph.nodes[m]['remark'])

	count_not_found = 0
	count_not_redirected = 0
	count_error = 0
	count_timeout = 0
	count_redirected = 0
	count_redirect_until_timeout = 0
	count_redirect_until_not_found = 0
	count_redirect_until_error = 0
	count_redirect_until_landed = 0
	count_other = 0

	for n in redi_graph.nodes():
		if n in graph.nodes():
			if redi_graph.nodes[n]['remark'] == 'not_found':
				count_not_found += 1
			elif redi_graph.nodes[n]['remark'] == 'not_redirect':
				count_not_redirected += 1
			elif redi_graph.nodes[n]['remark'] == 'error':
				count_error += 1
			elif redi_graph.nodes[n]['remark'] == 'timeout':
				count_timeout += 1
			elif redi_graph.nodes[n]['remark'] == 'redirect_until_timeout':
				count_redirect_until_timeout += 1
			elif redi_graph.nodes[n]['remark'] == 'redirect_until_not_found':
				count_redirect_until_not_found += 1
			elif redi_graph.nodes[n]['remark'] == 'redirect_until_error':
				count_redirect_until_error += 1
			elif redi_graph.nodes[n]['remark'] == 'redirect_until_landed':
				count_redirect_until_landed += 1
			else:
				print ('strange : ', redi_graph.nodes[n]['remark'])
				count_other += 1

	count_redirected = count_redirect_until_timeout + count_redirect_until_not_found + count_redirect_until_error + count_redirect_until_landed

	print ('Regarding the original graph:')
	print ('\tcount not found: ', count_not_found)
	print ('\tcount not redirected: ', count_not_redirected)
	print ('\tcount error: ', count_error)
	print ('\tcount timeout: ', count_timeout)
	print ('*****')
	print ('\tcount redirect until timeout: ', count_redirect_until_timeout)
	print ('\tcount redirect until not found: ', count_redirect_until_not_found)
	print ('\tcount redirect until error: ', count_redirect_until_error)
	print ('\tcount redirect until landed: ', count_redirect_until_landed)
	print ('\tTOTAL REDIRECTED: ', count_redirected)
	print ('\tcount other (mistake): ', count_other)

	# Validate
	count = 0
	for n in redi_graph.nodes():
		if redi_graph.nodes[n]['remark'] == 'unknown':
			print  ('unknown exists (but should not)!')
			print (n)
			count += 1

		if  redi_graph.nodes[n]['remark'] == 'end_of_redirect' :
			print ('end of redirect exists (but should not)!')
			print (n)

	# print('count unknown = ', count)
	print ('total num edges in the new redirect graph = ', len(redi_graph.edges()))

	end = time.time()
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)

	time_formated = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
	print("Time taken = ", time_formated)

	return redi_graph


def export_redirect_graph_edges(file_name, graph):
	# count_line = 0
	with open(file_name, 'w') as output:
		for (l,r) in graph.edges():
			line = '<' + l + '> '
			line += '<' + my_redirect + '> '
			if r[0] == '"':
				if "^^" in r:
					line += '' + r + ' .\n'
					# edited = splited[-2][1:-1] # keep that original one
					# example "http://xmlns.com/foaf/0.1/"^^<http://www.w3.org/2001/XMLSchema#anyURI>
				else: # else, add the following
					line += '' + r + "^^<http://www.w3.org/2001/XMLSchema#string>" + ' .\n'
					# edited = splited[-2][1:-1] + "^^<http://www.w3.org/2001/XMLSchema#string>"
			else:
				line +=  '<' + r +'>. \n'
			output.write(str(line))
			# count_line += 1
	# print ('count line = ', count_line)

def export_redirect_graph_nodes(file_name, graph):
	file =  open(file_name, 'w', newline='')
	writer = csv.writer(file,  delimiter='\t')
	writer.writerow([ "Entity", "Remark"])
	for n in graph.nodes:
		if graph.nodes[n]['remark'] == 'not_found':
			writer.writerow([n, 'NotFound'])
		elif graph.nodes[n]['remark'] == 'not_redirect':
			writer.writerow([n, 'NotRedirect'])
		elif graph.nodes[n]['remark'] == 'error':
			writer.writerow([n, 'Error'])
		elif graph.nodes[n]['remark'] == 'timeout':
			writer.writerow([n, 'Timeout'])
		elif graph.nodes[n]['remark'] == 'redirected':
			writer.writerow([n, 'Redirected'])
		elif graph.nodes[n]['remark'] == 'redirect_until_timeout':
			writer.writerow([n, 'RedirectedUntilTimeout'])
		elif graph.nodes[n]['remark'] == 'redirect_until_error':
			writer.writerow([n, 'RedirectedUntilError'])
		elif graph.nodes[n]['remark'] == 'redirect_until_landed':
			writer.writerow([n, 'RedirectedUntilLanded'])
		elif graph.nodes[n]['remark'] == 'redirect_until_not_found':
			writer.writerow([n, 'RedirectedUntilNotFound'])
		else:
			print ('Error: ', graph.nodes[n]['remark'])

sum_num_entities = 0
sum_num_edges = 0
total_num_unknown = 0

sum_error_edges = 0
sum_correct_edges = 0

prefix_ct = Counter()
prefix_ct_unknown = Counter()
for id in gs:
	print ('\n***************\n')
	dir = './gold/'
	filename = dir + str(id) +'.tsv'
	pairs = read_file(filename)

	print ('Loading ', id)

	sum_num_entities += len (pairs)
	g = nx.DiGraph()

	for (e, a) in pairs:
		g.add_node(e, annotation = a)
		if a == 'unknown':
			total_num_unknown += 1
	# step 1: obtain the whole graph
	obtain_edges(g)
	print ('There are ', g.number_of_nodes(), ' nodes')
	print ('There are ', g.number_of_edges(), ' edges')
	sum_num_edges += g.number_of_edges()
	for (l,r) in g.edges():
		if g.nodes[l]['annotation'] != 'unknown' and g.nodes[r]['annotation'] != 'unknown':
			if g.nodes[l]['annotation']  == g.nodes[r]['annotation']:
				sum_correct_edges += 1
			elif g.nodes[l]['annotation']  != g.nodes[r]['annotation']:
				sum_error_edges += 1


	# step 2: obtain metalink ID:
	for (l, r) in g.edges():
		meta_id = find_statement_id(l, r)
		if meta_id != None:
			g[l][r]['metalink_id'] = meta_id
		else:
			g[l][r]['metalink_id'] = None

	#step 3: export the edges and the metalink ID
	edges_file_name = dir + str(id) +'_edges.tsv'
	export_graph_edges(edges_file_name, g)

	# step 4: export the sources: Type A B C
	explicit_file_path = dir + str(id) + '_explicit_source.nt'
	export_explicit_source(explicit_file_path, g)

	label_file_path = dir + str(id) + '_implicit_label_source.nt'
	export_implicit_label_source(label_file_path, g)

	comment_file_path = dir +  str(id) + '_implicit_comment_source.nt'
	export_implicit_comment_source(comment_file_path, g)

	# step 4: obtain redirect nodes
	redi_graph = obtain_redirect_graph(g)
	redirect_file_name = dir + str(id) +'_redirect_edges.nt'
	export_redirect_graph_edges(redirect_file_name, redi_graph)
	redirect_file_name = dir + str(id) +'_redirect_nodes.tsv'
	export_redirect_graph_nodes(redirect_file_name, redi_graph)

print ('there are in total ', sum_num_entities, ' nodes')
print ('\tamong them, ', total_num_unknown, ' are unkonwn')
print ('\nthere are in total ', sum_num_edges, ' edges')
print ('\tamong them,', sum_correct_edges, ' are correct ->{:10.2f}'.format(sum_correct_edges/sum_num_edges * 100))
print ('\tamong them,', sum_error_edges, ' are errorenous ->{:10.2f}'.format(sum_error_edges/sum_num_edges *100))