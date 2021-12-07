# this script is for testing only and is not being maintained anymore 
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

sameas = 'http://www.w3.org/2002/07/owl#sameAs'
my_redirect = "https://krr.triply.cc/krr/metalink/def/redirectedTo" # a relation


import requests
from requests.exceptions import Timeout


NOTFOUND = 1
NOREDIRECT = 2
ERROR = 3
TIMEOUT = 4
REDIRECT = 5
# redirected till time out ?

#
# standard_timeout =  (0.1, 0.5)
# retry_timeout = (1, 5)
# final_try_timeout = (5, 10)

standard_timeout =  (0.01, 0.05)
retry_timeout = (0.5, 2.5)
final_try_timeout = (1, 5)

def find_redirects (iri, timeout = standard_timeout):
	try:
		collect_urls = []
		response = requests.get(iri, timeout= timeout, allow_redirects=True)

		if response.status_code == 404:
			return NOTFOUND, None

		if response.history:
			if response.url == iri:
				return NOREDIRECT, None
			else:
				# print("Request was redirected")
				for resp in response.history:
					# print(resp.status_code, resp.url)
					collect_urls.append(resp.url)
				# print("Final destination:")
				# print(response.status_code, response.url)

				collect_urls.append(response.url)
				return REDIRECT, collect_urls
		else:
			# print("Request was not redirected")
			return NOREDIRECT, None
	except Timeout:
		# print('The request timed out', iri)
		return TIMEOUT, None
	except:
		# print ('error: ', iri)
		return ERROR, None

def load_entities(graph_id):

	input_graph = nx.Graph()
	path_to_input_graph = './Evaluate_May/' + str(graph_id) + '_edges_original.csv'
	input_graph_data = pd.read_csv(path_to_input_graph)

	sources = input_graph_data['SUBJECT']
	targets = input_graph_data['OBJECT']
	edge_data = zip(sources, targets)

	entities = set()
	for (s,t) in edge_data:
		entities.add(s)
		entities.add(t)
	return entities

# graph_ids = [11116]
graph_ids = [11116, 240577, 395175, 14514123]

for graph_id in graph_ids:
	print ('\n\n\ngraph id = ', graph_id)
	start = time.time()

	redi_graph = nx.DiGraph()

	entities_to_test = load_entities(graph_id)
	print ('there are ', len (entities_to_test), ' entities in the graph ')
	timeout_entities = set()

	# count_notfound = 0
	# count_no_redirect = 0
	# count_error = 0
	# count_timeout = 0
	# count_redirect = 0
	original_entities = entities_to_test.copy()
	redirect_result = {}
	for e in entities_to_test:
		redirect_result[e] = None

	while len(entities_to_test) != 0:
		collect_new_entities_to_test = set()
		for e in entities_to_test:
			# find_redirects
			result, via_entities = find_redirects(e)
			# print ('testing ',e)
			if result == NOTFOUND:
				redirect_result[e] = NOTFOUND
			elif result == NOREDIRECT:
				redirect_result[e] = NOREDIRECT
			elif result == ERROR:
				redirect_result[e] = ERROR
			elif result == TIMEOUT:
				redirect_result[e] = TIMEOUT
				timeout_entities.add(e)
			elif result == REDIRECT:
				redirect_result[e] = REDIRECT
				if len (via_entities) > 1:
					for i, s in enumerate(via_entities[:-1]):
						t = via_entities[i+1]
						redi_graph.add_edge(s, t)

					if via_entities[-1] not in entities_to_test:
						collect_new_entities_to_test.add(via_entities[-1])
				else:
					print ('error: ', via_entities)
			else:
				print ('error')

		print ('TIMEOUT: there are ', len (timeout_entities), ' timeout entities')

		count_timeout = 0
		remain_timeout_entities = set()
		for e in timeout_entities:
			# find_redirects
			result, via_entities = find_redirects(e, timeout = retry_timeout)
			# print ('testing ',e)
			if result == NOTFOUND:
				redirect_result[e] = NOTFOUND
			elif result == NOREDIRECT:
				redirect_result[e] = NOREDIRECT
			elif result == ERROR:
				redirect_result[e] = ERROR
			elif result == TIMEOUT:
				count_timeout += 1
				remain_timeout_entities.add(e)
			elif result == REDIRECT:
				redirect_result[e] = REDIRECT
				if len (via_entities) > 1:
					# count_redirect += 1
					for i, s in enumerate(via_entities[:-1]):
						t = via_entities[i+1]
						redi_graph.add_edge(s, t)

					if via_entities[-1] not in entities_to_test:
						collect_new_entities_to_test.add(via_entities[-1])
				else:
					print ('too short? error: ',via_entities)
			else:
				print ('error')
		print ('TIMEOUT: still timeout ', count_timeout)

		count_timeout = 0
		for e in remain_timeout_entities:
			# find_redirects
			result, via_entities = find_redirects(e, timeout = final_try_timeout)
			# print ('testing ',e)
			if result == NOTFOUND:
				redirect_result[e] = NOTFOUND
			elif result == NOREDIRECT:
				redirect_result[e] = NOREDIRECT
			elif result == ERROR:
				redirect_result[e] = ERROR
			elif result == TIMEOUT:
				count_timeout += 1
				# timeout_entities.add(e)
				print (e)
			elif result == REDIRECT:
				redirect_result[e] = REDIRECT
				if len (via_entities) > 1:
					# count_redirect += 1
					for i, s in enumerate(via_entities[:-1]):
						t = via_entities[i+1]
						redi_graph.add_edge(s, t)

					if via_entities[-1] not in entities_to_test:
						collect_new_entities_to_test.add(via_entities[-1])
				else:
					print ('too short? error: ',via_entities)
			else:
				print ('error')
		print ('TIMEOUT: still timeout after final try', count_timeout)

		timeout_entities = set()
		entities_to_test = collect_new_entities_to_test

	redirect_result_filtered = { your_key: redirect_result[your_key] for your_key in original_entities }

	print ('there are in total ', sum(value == NOTFOUND for value in redirect_result_filtered.values()), ' not found')
	print ('there are in total ', sum(value == NOREDIRECT for value in redirect_result_filtered.values()), ' no redirect')
	print ('there are in total ', sum(value == TIMEOUT for value in redirect_result_filtered.values()), ' timeout')
	print ('there are in total ', sum(value == ERROR for value in redirect_result_filtered.values()), ' error')
	print ('there are in total ', sum(value == REDIRECT for value in redirect_result_filtered.values()), 'redirected')

	print ('total num edges in the new redirect graph = ', len(redi_graph.edges()))

	end = time.time()
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)

	time_formated = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
	print("Time taken = ", time_formated)

	print ('Exporting')
	file = open( str(graph_id) + "_redirect.nt", 'w')
	redirect_file_writer = csv.writer(file, delimiter=' ')
	for (s,t) in redi_graph.edges():
		# log_file_writer.writerow([s,t])
		if t[0] != '"':
			redirect_file_writer.writerow(['<'+s+'>', '<'+my_redirect+'>', '<'+t+'>', '.'])
		else:
			redirect_file_writer.writerow(['<'+s+'>', '<'+my_redirect+'>', t+'^^<http://www.w3.org/2001/XMLSchema#string>', '.'])
