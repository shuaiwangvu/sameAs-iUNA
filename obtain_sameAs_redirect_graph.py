
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


import requests
from requests.exceptions import Timeout


NOTFOUND = 1
NOREDIRECT = 2
ERROR = 3
TIMEOUT = 4
REDIRECT = 5

standard_timeout =  (0.1, 0.5)
retry_timeout = (1, 5)

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

graph_ids = [11116, 240577]
# graph_ids = [11116, 240577, 395175, 14514123]

for graph_id in graph_ids:
	print ('\n\n\ngraph id = ', graph_id)
	start = time.time()

	redi_graph = nx.DiGraph()

	entities_to_test = load_entities(graph_id)
	print ('there are ', len (entities_to_test), ' entities in the graph ')
	timeout_entities = set()

	count_notfound = 0
	count_no_redirect = 0
	count_error = 0
	count_timeout = 0
	count_redirect = 0

	while len(entities_to_test) != 0:
		collect_new_entities_to_test = set()
		for e in entities_to_test:
			# find_redirects
			result, via_entities = find_redirects(e)
			# print ('testing ',e)
			if result == NOTFOUND:
				count_notfound += 1
			elif result == NOREDIRECT:
				count_no_redirect += 1
			elif result == ERROR:
				count_error += 1
			elif result == TIMEOUT:
				count_timeout += 1
				timeout_entities.add(e)
			else:
				if len (via_entities) > 1:

					count_redirect += 1
					for i, s in enumerate(via_entities[:-1]):
						t = via_entities[i+1]
						redi_graph.add_edge(s, t)

					if via_entities[-1] not in entities_to_test:
						collect_new_entities_to_test.add(via_entities[-1])
				else:
					print ('error: ', via_entities)


		print ('TIMEOUT: there are ', len (timeout_entities), ' timeout entities')
		count_timeout = 0
		for e in timeout_entities:
			# find_redirects
			result, via_entities = find_redirects(e, timeout = retry_timeout)
			# print ('testing ',e)
			if result == NOTFOUND:
				count_notfound += 1
			elif result == NOREDIRECT:
				count_no_redirect += 1
			elif result == ERROR:
				count_error += 1
			elif result == TIMEOUT:
				count_timeout += 1
				# timeout_entities.add(e)
			else:
				if len (via_entities) > 1:

					count_redirect += 1
					for i, s in enumerate(via_entities[:-1]):
						t = via_entities[i+1]
						redi_graph.add_edge(s, t)

					if via_entities[-1] not in entities_to_test:
						collect_new_entities_to_test.add(via_entities[-1])
				else:
					print ('too short? error: ',via_entities)
		print ('TIMEOUT: still timeout ', count_timeout)

		timeout_entities = set()
		entities_to_test = collect_new_entities_to_test

	# print ('there are in total ', count_notfound, ' not found')
	# print ('there are in total ', count_no_redirect, ' no redirect')
	# print ('there are in total ', count_timeout, ' timeout')
	# print ('there are in total ', count_error, ' error')
	# print ('there are in total ', count_redirect, 'redirected')

	print ('total num edges in the new redirect graph = ', len(redi_graph.edges()))

	end = time.time()
	hours, rem = divmod(end-start, 3600)
	minutes, seconds = divmod(rem, 60)

	time_formated = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
	print("Time taken = ", time_formated)
