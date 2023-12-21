# Please refer to https://github.com/shuaiwangvu/sameAs-iUNA
# for the latest version.
# This files generates the statistics of the evaluation results of each
# prametirc setting. The execution history are in the ./log/ folder
# A summary of the evaulation results are in the /log/<name_of_method>.log files.
# * Please refer to line 72 and select the methods.

import networkx as nx
import collections
import matplotlib.pyplot as plt
from random import randint
import requests
from collections import Counter
from rfc3987 import  parse
import urllib.parse
from hdt import HDTDocument, IdentifierPosition
from z3 import *
import csv
from rdflib import Literal, XSD
from networkx.algorithms.connectivity import is_locally_k_edge_connected
from extend_metalink import *
import csv
import random
import time
from math import exp
import numpy as np

from SameAsEqGraph import *
from GraphSolver import *


debug = False

which_source = 'implicit_label_source'


# there are in total 28 entities. 14 each
# the training set (for the training of the method)
training_single = [96073, 712342, 9994282, 18688, 1140988, 25604]
training_multiple = [33122, 11116, 12745, 6617,4170, 42616, 6927, 39036]
training_set = training_single + training_multiple
# the evaluation set
evaluation_single = [9411, 9756, 97757, 99932, 337339, 1133953]
evaluation_multiple = [5723, 14872, 37544, 236350, 240577, 395175, 4635725, 14514123]
evaluation_set = evaluation_single + evaluation_multiple


gs = training_set + evaluation_set


hard_graphs = [6927, 37544, 4635725]

restricted_prefix_list = ["http://dblp.rkbexplorer.com/id/",
"http://dbpedia.org/resource/",
"http://rdf.freebase.com/ns/m/",
"http://sws.geonames.org/",
"http://dbtune.org/musicbrainz/resource/",
"http://bio2rdf.org/uniprot:"]



graph_ids = []
# training_set or evaluation_set
export_dir = './log_final/'


WITH_WEIGHT = False
WITH_DISAMBIG = False

NUM_ITER = 5

for which_method in ['smt', 'qUNA', 'leiden', 'louvain']: #, 'leiden', 'louvain', 'qUNA', 'smt']:

	if which_method == 'leiden':
		print ('Using the Leiden algorithm')
	# if True:
		overall_logbook_filename = export_dir + which_method + '_overall' + '.log'
		overall_logbook_writer = open(overall_logbook_filename, 'w')
		overall_logbook_writer.write('\nmethod = ' + which_method)
		for which_set in [ 'training', 'evaluation']:
			if which_set == 'training':
				graph_ids =  training_set
				print ('working on the training set')
			else:
				graph_ids =  evaluation_set
				print ('working on the evaluation set')

			overall_logbook_writer.write ('\n********\ndataset = ' + which_set)

			overall_avg_precision = 0
			overall_avg_recall = 0
			overall_avg_omega = 0
			overall_avg_num_edges_removed = 0
			overall_avg_valid_result = 0
			overall_avg_invalid_result = 0

			overall_avg_termination_tp = 0
			overall_avg_termination_fp = 0
			overall_avg_termination_accuracy = 0

			for i in range (NUM_ITER): # repeat 5 times.
				logbook_filename = export_dir + which_method + '_' + which_set +'_Run' + str(i) + '.log'

				avg_precision = 0
				avg_recall = 0
				avg_termination_tp = 0
				avg_termination_fp = 0
				termination_tp = 0
				termination_tn = 0
				termination_fp = 0
				termination_fn = 0

				avg_omega = 0
				num_edges_removed = 0

				count_valid_result = 0
				count_invalid_result = 0
				start = time.time()
				count_graph_no_error_edges = 0
				for graph_id in graph_ids: # graph_ids:

					# removed_edge_name  = open( "convert_typeC_progress.tsv", 'w')
					# no_metalink_writer = csv.writer(file_no_metalink, delimiter='\t')
					# no_metalink_writer.writerow(['Count', 'Time'])
					filename_removed_edges = export_dir + which_method + '_' + which_set +'_Run' + str(i) + '_Graph' + str(graph_id) + '_removed_edges.tsv'
					edge_writer = csv.writer(open(filename_removed_edges, 'w'), delimiter='\t')
					edge_writer.writerow(['Source', 'Target'])

					# print ('\n\n\ngraph id = ', str(graph_id))
					gold_dir = './gold/'
					gs = GraphSolver(gold_dir, graph_id = graph_id)

					# gs.show_input_graph()
					# gs.show_gold_standard_graph()
					# gs.show_redirect_graph()
					# gs.show_encoding_equivalence_graph()
					# if which_method == "louvain":
					gs.partition_leiden()

					e_result = gs.evaluate_partitioning_result()

					if e_result ['num_edges_removed'] != 0:
						for (s, t) in gs.removed_edges:
							edge_writer.writerow([s, t]) #removed_edges

					if e_result ['num_error_edges_gold_standard'] == 0:
						count_graph_no_error_edges += 1
					print ('removed ', e_result['num_edges_removed'])

					num_edges_removed += e_result['num_edges_removed']
					avg_omega += e_result['Omega']
					if e_result['flag'] == 'valid precision and recall':
						p = e_result['precision']
						r = e_result['recall']
						m = e_result ['Omega']
						if debug:
							print ('precision =', p)
							print ('recall   =', r)
							print ('omega   =', m)
						count_valid_result += 1
						avg_precision += e_result['precision']
						avg_recall += e_result['recall']


					else:
						count_invalid_result += 1

					if e_result['num_edges_removed'] == 0:
						if e_result['num_error_edges_gold_standard'] == 0:
							termination_tp += 1
						else:
							termination_fp += 1
					else:
						if e_result['num_error_edges_gold_standard'] == 0:
							termination_fn += 1
						else:
							termination_tn += 1

					avg_termination_tp += termination_tp
					avg_termination_fp += termination_fp
					# avg_termination_accuracy += temination_tp /(termination_tp + temination_fp)
					# if (termination_tp + termination_fp) >0 :
					# 	avg_termination_tp = termination_tp / (termination_tp + termination_fp)
					# if (termination_tp+ termination_fn) > 0:
					# 	avg_termination_fp = termination_tp / (termination_tp+ termination_fn)

				# evaluation_result ['num_edges_removed'] = len(self.removed_edges)
				# evaluation_result ['num_error_edges_gold_standard'] = len(self.removed_edges)

				avg_omega /= len(graph_ids)
				# gs.show_result_graph()
				overall_avg_omega += avg_omega
				print ('The average Omega: ', avg_omega)
				print ('Count result [where precision and recall works] ', count_valid_result)
				print ('Count invalid [where precision and recall do not apply] ', count_invalid_result)

				if count_valid_result > 0:
					avg_precision /= count_valid_result
					avg_recall /= count_valid_result
					print ('*'*20)
					print ('There are ', len (graph_ids), ' graphs in ', which_set)
					print ('   ', count_graph_no_error_edges, ' has no error edge')
					print ('The average precision: ', avg_precision)
					print ('The average recall: ', avg_recall)
					print ('The average Omega: ', avg_omega)
					print ('Total num edges removed ', num_edges_removed)

					overall_avg_precision += avg_precision
					overall_avg_recall += avg_recall

				overall_avg_num_edges_removed += num_edges_removed


				# if count_invalid_result > 0:
				avg_termination_tp /= len(graph_ids)
				avg_termination_fp /= len(graph_ids)
				# avg_termination_accuracy /= len(graph_ids)

				overall_avg_termination_tp += avg_termination_tp
				overall_avg_termination_fp += avg_termination_fp
				# overall_avg_termination_accuracy += avg_termination_accuracy

				overall_avg_valid_result += count_valid_result
				overall_avg_invalid_result += count_invalid_result

				end = time.time()
				hours, rem = divmod(end-start, 3600)
				minutes, seconds = divmod(rem, 60)
				time_formated = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
				print ('time taken: ' + time_formated)

			overall_avg_precision /= NUM_ITER
			overall_avg_recall /= NUM_ITER
			overall_avg_omega /= NUM_ITER
			overall_avg_num_edges_removed /= NUM_ITER
			overall_avg_valid_result /= NUM_ITER
			overall_avg_invalid_result /= NUM_ITER
			overall_avg_termination_tp /= NUM_ITER
			overall_avg_termination_fp /= NUM_ITER
			# overall_avg_termination_accuracy /= NUM_ITER

			print ('='*20)
			print ('total number of iterations over the dataset ', NUM_ITER)
			print ('OVERALL There are ', len (graph_ids), ' graphs')
			print ('   ' + str(count_graph_no_error_edges) + ' has no error edge')
			print ('OVERALL Count result [where precision and recall works] ', count_valid_result)
			print ('OVERALL Count inresult [where precision and recall do not apply] ', count_invalid_result)
			print ('OVERALL The average precision: ', overall_avg_precision)
			print ('OVERALL The average recall: ', overall_avg_recall)
			# print ('OVERALL The average tp: [for termination]', overall_avg_termination_tp)
			# print ('OVERALL The average fp: [for termination]', overall_avg_termination_fp)
			# print ('OVERALL The average accuracy: [for termination]', overall_avg_termination_accuracy)
			print ('OVERALL The average Omega: ', overall_avg_omega)
			print ('OVERALL Total num edges removed ', overall_avg_num_edges_removed)

			overall_logbook_writer.write ('\n\ntotal number of iterations over this dataset ' +str(NUM_ITER))
			overall_logbook_writer.write ('\nOVERALL There are '+str(len (graph_ids)) + ' graphs')
			overall_logbook_writer.write ('   ' + str(count_graph_no_error_edges) + ' has no error edge')
			overall_logbook_writer.write ('\nOVERALL Count result [where precision and recall works] '+ str(count_valid_result))
			overall_logbook_writer.write ('\nOVERALL Count result [where precision and recall do not apply] ' +str(count_invalid_result))
			overall_logbook_writer.write ('\nOVERALL The average precision: ' +str(overall_avg_precision))
			overall_logbook_writer.write ('\nOVERALL The average recall: '+str(overall_avg_recall))
			overall_logbook_writer.write ('\nOVERALL The average tp [for termination]: ' +str(overall_avg_termination_tp))
			overall_logbook_writer.write ('\nOVERALL The average fp [for termination]: '+str(overall_avg_termination_fp))
			# overall_logbook_writer.write ('\nOVERALL The average accuracy [for termination]: '+str(overall_avg_termination_accuracy))

			overall_logbook_writer.write ('\nOVERALL The average Omega: '+str(overall_avg_omega))
			overall_logbook_writer.write ('\nOVERALL Total num edges removed '+str(overall_avg_num_edges_removed))

	elif which_method == 'louvain':
		for louvain_res in [1.0, 0.01]: # louvain_res = 1.0 #0.01
			print ('Using resolution ' + str(louvain_res))
			overall_logbook_filename = export_dir + which_method + str(louvain_res)  +'_overall' + '.log'
			overall_logbook_writer = open(overall_logbook_filename, 'w')
			overall_logbook_writer.write('\nmethod = ' + which_method)
			for which_set in ['training', 'evaluation']:
				if which_set == 'training':
					graph_ids =  training_set
					print ('working on the training set')
				else:
					graph_ids =  evaluation_set
					print ('working on the evaluation set')

				overall_logbook_writer.write ('\n********\ndataset = ' + which_set)

				overall_avg_precision = 0
				overall_avg_recall = 0
				overall_avg_omega = 0
				overall_avg_num_edges_removed = 0
				overall_avg_valid_result = 0
				overall_avg_invalid_result = 0

				overall_avg_termination_tp = 0
				overall_avg_termination_fp = 0
				overall_avg_termination_accuracy = 0

				for i in range (NUM_ITER): # repeat 5 times.
					logbook_filename = export_dir + which_method + str(louvain_res) + '_' + which_set +'_Run' + str(i) + '.log'

					avg_precision = 0
					avg_recall = 0
					avg_termination_tp = 0
					avg_termination_fp = 0
					termination_tp = 0
					termination_tn = 0
					termination_fp = 0
					termination_fn = 0

					avg_omega = 0
					num_edges_removed = 0

					count_valid_result = 0
					count_invalid_result = 0
					start = time.time()
					count_graph_no_error_edges = 0
					for graph_id in graph_ids: # graph_ids:

						# removed_edge_name  = open( "convert_typeC_progress.tsv", 'w')
						# no_metalink_writer = csv.writer(file_no_metalink, delimiter='\t')
						# no_metalink_writer.writerow(['Count', 'Time'])
						filename_removed_edges = export_dir + which_method +  str(louvain_res) + '_' + which_set +'_Run' + str(i) + '_Graph' + str(graph_id) + '_removed_edges.tsv'
						edge_writer = csv.writer(open(filename_removed_edges, 'w'), delimiter='\t')
						edge_writer.writerow(['Source', 'Target'])

						# print ('\n\n\ngraph id = ', str(graph_id))
						gold_dir = './gold/'
						gs = GraphSolver(gold_dir, graph_id = graph_id)

						# gs.show_input_graph()
						# gs.show_gold_standard_graph()
						# gs.show_redirect_graph()
						# gs.show_encoding_equivalence_graph()
						# if which_method == "louvain":
						gs.partition_louvain(res = louvain_res)

						e_result = gs.evaluate_partitioning_result()

						if e_result ['num_edges_removed'] != 0:
							for (s, t) in gs.removed_edges:
								edge_writer.writerow([s, t]) #removed_edges

						if e_result ['num_error_edges_gold_standard'] == 0:
							count_graph_no_error_edges += 1

						num_edges_removed += e_result['num_edges_removed']
						avg_omega += e_result['Omega']
						if e_result['flag'] == 'valid precision and recall':
							p = e_result['precision']
							r = e_result['recall']
							m = e_result ['Omega']
							if debug:
								print ('precision =', p)
								print ('recall   =', r)
								print ('omega   =', m)
							count_valid_result += 1
							avg_precision += e_result['precision']
							avg_recall += e_result['recall']


						else:
							count_invalid_result += 1

						if e_result['num_edges_removed'] == 0:
							if e_result['num_error_edges_gold_standard'] == 0:
								termination_tp += 1
							else:
								termination_fp += 1
						else:
							if e_result['num_error_edges_gold_standard'] == 0:
								termination_fn += 1
							else:
								termination_tn += 1

						avg_termination_tp += termination_tp
						avg_termination_fp += termination_fp
						# avg_termination_accuracy += temination_tp /(termination_tp + temination_fp)
						# if (termination_tp + termination_fp) >0 :
						# 	avg_termination_tp = termination_tp / (termination_tp + termination_fp)
						# if (termination_tp+ termination_fn) > 0:
						# 	avg_termination_fp = termination_tp / (termination_tp+ termination_fn)

					# evaluation_result ['num_edges_removed'] = len(self.removed_edges)
					# evaluation_result ['num_error_edges_gold_standard'] = len(self.removed_edges)

					avg_omega /= len(graph_ids)
					# gs.show_result_graph()
					overall_avg_omega += avg_omega
					print ('The average Omega: ', avg_omega)
					print ('Count result [where precision and recall works] ', count_valid_result)
					print ('Count invalid [where precision and recall do not apply] ', count_invalid_result)

					if count_valid_result > 0:
						avg_precision /= count_valid_result
						avg_recall /= count_valid_result
						print ('*'*20)
						print ('There are ', len (graph_ids), ' graphs in ', which_set)
						print ('   ', count_graph_no_error_edges, ' has no error edge')
						print ('The average precision: ', avg_precision)
						print ('The average recall: ', avg_recall)
						print ('The average Omega: ', avg_omega)
						print ('Total num edges removed ', num_edges_removed)

						overall_avg_precision += avg_precision
						overall_avg_recall += avg_recall

					overall_avg_num_edges_removed += num_edges_removed


					# if count_invalid_result > 0:
					avg_termination_tp /= len(graph_ids)
					avg_termination_fp /= len(graph_ids)
					# avg_termination_accuracy /= len(graph_ids)

					overall_avg_termination_tp += avg_termination_tp
					overall_avg_termination_fp += avg_termination_fp
					# overall_avg_termination_accuracy += avg_termination_accuracy

					overall_avg_valid_result += count_valid_result
					overall_avg_invalid_result += count_invalid_result

					end = time.time()
					hours, rem = divmod(end-start, 3600)
					minutes, seconds = divmod(rem, 60)
					time_formated = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
					print ('time taken: ' + time_formated)

				overall_avg_precision /= NUM_ITER
				overall_avg_recall /= NUM_ITER
				overall_avg_omega /= NUM_ITER
				overall_avg_num_edges_removed /= NUM_ITER
				overall_avg_valid_result /= NUM_ITER
				overall_avg_invalid_result /= NUM_ITER
				overall_avg_termination_tp /= NUM_ITER
				overall_avg_termination_fp /= NUM_ITER
				# overall_avg_termination_accuracy /= NUM_ITER

				print ('='*20)
				print ('total number of iterations over the dataset ', NUM_ITER)
				print ('OVERALL There are ', len (graph_ids), ' graphs')
				print ('   ' + str(count_graph_no_error_edges) + ' has no error edge')
				print ('OVERALL Count result [where precision and recall works] ', count_valid_result)
				print ('OVERALL Count inresult [where precision and recall do not apply] ', count_invalid_result)
				print ('OVERALL The average precision: ', overall_avg_precision)
				print ('OVERALL The average recall: ', overall_avg_recall)
				# print ('OVERALL The average tp: [for termination]', overall_avg_termination_tp)
				# print ('OVERALL The average fp: [for termination]', overall_avg_termination_fp)
				# print ('OVERALL The average accuracy: [for termination]', overall_avg_termination_accuracy)
				print ('OVERALL The average Omega: ', overall_avg_omega)
				print ('OVERALL Total num edges removed ', overall_avg_num_edges_removed)

				overall_logbook_writer.write ('\n\ntotal number of iterations over this dataset ' +str(NUM_ITER))
				overall_logbook_writer.write ('\nOVERALL There are '+str(len (graph_ids)) + ' graphs')
				overall_logbook_writer.write ('   ' + str(count_graph_no_error_edges) + ' has no error edge')
				overall_logbook_writer.write ('\nOVERALL Count result [where precision and recall works] '+ str(count_valid_result))
				overall_logbook_writer.write ('\nOVERALL Count result [where precision and recall do not apply] ' +str(count_invalid_result))
				overall_logbook_writer.write ('\nOVERALL The average precision: ' +str(overall_avg_precision))
				overall_logbook_writer.write ('\nOVERALL The average recall: '+str(overall_avg_recall))
				overall_logbook_writer.write ('\nOVERALL The average tp [for termination]: ' +str(overall_avg_termination_tp))
				overall_logbook_writer.write ('\nOVERALL The average fp [for termination]: '+str(overall_avg_termination_fp))
				# overall_logbook_writer.write ('\nOVERALL The average accuracy [for termination]: '+str(overall_avg_termination_accuracy))

				overall_logbook_writer.write ('\nOVERALL The average Omega: '+str(overall_avg_omega))
				overall_logbook_writer.write ('\nOVERALL Total num edges removed '+str(overall_avg_num_edges_removed))

	elif which_method == 'metalink':
		for error_rate in [0.9, 0.99]: # louvain_res = 1.0 #0.01
			print ('Using Metalink error rate ' + str(error_rate))
			overall_logbook_filename = export_dir + which_method + str(error_rate)  +'_overall' + '.log'
			overall_logbook_writer = open(overall_logbook_filename, 'w')
			overall_logbook_writer.write('\nmethod = ' + which_method)
			for which_set in ['training', 'evaluation']:
				if which_set == 'training':
					graph_ids =  training_set
					print ('working on the training set')
				else:
					graph_ids =  evaluation_set
					print ('working on the evaluation set')

				overall_logbook_writer.write ('\n********\ndataset = ' + which_set)

				overall_avg_precision = 0
				overall_avg_recall = 0
				overall_avg_omega = 0
				overall_avg_num_edges_removed = 0
				overall_avg_valid_result = 0
				overall_avg_invalid_result = 0

				overall_avg_termination_tp = 0
				overall_avg_termination_fp = 0
				overall_avg_termination_accuracy = 0

				for i in range (NUM_ITER): # repeat 5 times.
					logbook_filename = export_dir + which_method + str(error_rate) + '_' + which_set +'_Run' + str(i) + '.log'

					avg_precision = 0
					avg_recall = 0
					avg_termination_tp = 0
					avg_termination_fp = 0
					termination_tp = 0
					termination_tn = 0
					termination_fp = 0
					termination_fn = 0

					avg_omega = 0
					num_edges_removed = 0

					count_valid_result = 0
					count_invalid_result = 0
					start = time.time()
					count_graph_no_error_edges = 0
					for graph_id in graph_ids: # graph_ids:

						# removed_edge_name  = open( "convert_typeC_progress.tsv", 'w')
						# no_metalink_writer = csv.writer(file_no_metalink, delimiter='\t')
						# no_metalink_writer.writerow(['Count', 'Time'])
						filename_removed_edges = export_dir + which_method +  str(error_rate) + '_' + which_set +'_Run' + str(i) + '_Graph' + str(graph_id) + '_removed_edges.tsv'
						edge_writer = csv.writer(open(filename_removed_edges, 'w'), delimiter='\t')
						edge_writer.writerow(['Source', 'Target'])

						# print ('\n\n\ngraph id = ', str(graph_id))
						gold_dir = './gold/'
						gs = GraphSolver(gold_dir, graph_id = graph_id)


						gs.partition_metalink(threshold = error_rate)

						e_result = gs.evaluate_partitioning_result()

						if e_result ['num_edges_removed'] != 0:
							for (s, t) in gs.removed_edges:
								edge_writer.writerow([s, t]) #removed_edges

						if e_result ['num_error_edges_gold_standard'] == 0:
							count_graph_no_error_edges += 1

						num_edges_removed += e_result['num_edges_removed']
						avg_omega += e_result['Omega']
						if e_result['flag'] == 'valid precision and recall':
							p = e_result['precision']
							r = e_result['recall']
							m = e_result ['Omega']
							if debug:
								print ('precision =', p)
								print ('recall   =', r)
								print ('omega   =', m)
							count_valid_result += 1
							avg_precision += e_result['precision']
							avg_recall += e_result['recall']


						else:
							count_invalid_result += 1

						if e_result['num_edges_removed'] == 0:
							if e_result['num_error_edges_gold_standard'] == 0:
								termination_tp += 1
							else:
								termination_fp += 1
						else:
							if e_result['num_error_edges_gold_standard'] == 0:
								termination_fn += 1
							else:
								termination_tn += 1

						avg_termination_tp += termination_tp
						avg_termination_fp += termination_fp

					avg_omega /= len(graph_ids)
					# gs.show_result_graph()
					overall_avg_omega += avg_omega
					print ('The average Omega: ', avg_omega)
					print ('Count result [where precision and recall works] ', count_valid_result)
					print ('Count invalid [where precision and recall do not apply] ', count_invalid_result)

					if count_valid_result > 0:
						avg_precision /= count_valid_result
						avg_recall /= count_valid_result
						print ('*'*20)
						print ('There are ', len (graph_ids), ' graphs in ', which_set)
						print ('   ', count_graph_no_error_edges, ' has no error edge')
						print ('The average precision: ', avg_precision)
						print ('The average recall: ', avg_recall)
						print ('The average Omega: ', avg_omega)
						print ('Total num edges removed ', num_edges_removed)

						overall_avg_precision += avg_precision
						overall_avg_recall += avg_recall

					overall_avg_num_edges_removed += num_edges_removed


					# if count_invalid_result > 0:
					avg_termination_tp /= len(graph_ids)
					avg_termination_fp /= len(graph_ids)
					# avg_termination_accuracy /= len(graph_ids)

					overall_avg_termination_tp += avg_termination_tp
					overall_avg_termination_fp += avg_termination_fp
					# overall_avg_termination_accuracy += avg_termination_accuracy

					overall_avg_valid_result += count_valid_result
					overall_avg_invalid_result += count_invalid_result

					end = time.time()
					hours, rem = divmod(end-start, 3600)
					minutes, seconds = divmod(rem, 60)
					time_formated = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
					print ('time taken: ' + time_formated)

				overall_avg_precision /= NUM_ITER
				overall_avg_recall /= NUM_ITER
				overall_avg_omega /= NUM_ITER
				overall_avg_num_edges_removed /= NUM_ITER
				overall_avg_valid_result /= NUM_ITER
				overall_avg_invalid_result /= NUM_ITER
				overall_avg_termination_tp /= NUM_ITER
				overall_avg_termination_fp /= NUM_ITER
				# overall_avg_termination_accuracy /= NUM_ITER

				print ('='*20)
				print ('total number of iterations over the dataset ', NUM_ITER)
				print ('OVERALL There are ', len (graph_ids), ' graphs')
				print ('   ' + str(count_graph_no_error_edges) + ' has no error edge')
				print ('OVERALL Count result [where precision and recall works] ', count_valid_result)
				print ('OVERALL Count inresult [where precision and recall do not apply] ', count_invalid_result)
				print ('OVERALL The average precision: ', overall_avg_precision)
				print ('OVERALL The average recall: ', overall_avg_recall)
				# print ('OVERALL The average tp: [for termination]', overall_avg_termination_tp)
				# print ('OVERALL The average fp: [for termination]', overall_avg_termination_fp)
				# print ('OVERALL The average accuracy: [for termination]', overall_avg_termination_accuracy)
				print ('OVERALL The average Omega: ', overall_avg_omega)
				print ('OVERALL Total num edges removed ', overall_avg_num_edges_removed)

				overall_logbook_writer.write ('\n\ntotal number of iterations over this dataset ' +str(NUM_ITER))
				overall_logbook_writer.write ('\nOVERALL There are '+str(len (graph_ids)) + ' graphs')
				overall_logbook_writer.write ('   ' + str(count_graph_no_error_edges) + ' has no error edge')
				overall_logbook_writer.write ('\nOVERALL Count result [where precision and recall works] '+ str(count_valid_result))
				overall_logbook_writer.write ('\nOVERALL Count result [where precision and recall do not apply] ' +str(count_invalid_result))
				overall_logbook_writer.write ('\nOVERALL The average precision: ' +str(overall_avg_precision))
				overall_logbook_writer.write ('\nOVERALL The average recall: '+str(overall_avg_recall))
				overall_logbook_writer.write ('\nOVERALL The average tp [for termination]: ' +str(overall_avg_termination_tp))
				overall_logbook_writer.write ('\nOVERALL The average fp [for termination]: '+str(overall_avg_termination_fp))
				# overall_logbook_writer.write ('\nOVERALL The average accuracy [for termination]: '+str(overall_avg_termination_accuracy))

				overall_logbook_writer.write ('\nOVERALL The average Omega: '+str(overall_avg_omega))
				overall_logbook_writer.write ('\nOVERALL Total num edges removed '+str(overall_avg_num_edges_removed))


	elif which_method == 'smt':
		for selected_UNA in ['iUNA','nUNA']: # or nUNA
			print ('<<<< refining using ', selected_UNA)
			for which_source in ['implicit_label_source', 'implicit_comment_source']: # 'implicit_comment_source': implicit_label_source
				for selected_weighting_scheme in ['w3', 'w4', 'w5', 'w1', 'w2']: # 'w2',

					additional = ''
					if WITH_WEIGHT:
						additional += '_[weights]'
					if WITH_DISAMBIG:
						additional += '_[disambiguation]'

					overall_logbook_filename = export_dir + which_method +'_' + selected_UNA+ '_' + which_source+ '_'+ selected_weighting_scheme + '_overall' + additional + '.log'
					overall_logbook_writer = open(overall_logbook_filename, 'w')
					overall_logbook_writer.write('\n method = ' + which_method)
					overall_logbook_writer.write('\n UNA = ' + selected_UNA)

					if WITH_WEIGHT:
						overall_logbook_writer.write('\n Additioinal info = weight')
					if WITH_DISAMBIG:
						overall_logbook_writer.write('\n Additioinal info = disambiguation')

					overall_logbook_writer.write('\n source = ' + which_source)
					overall_logbook_writer.write('\n weighting scheme = ' + selected_weighting_scheme)

					for which_set in ['training', 'evaluation']: #
						print ('<<<<< working on the ', which_set, 'dataset')
						time_taken = 0
						if which_set == 'training':
							graph_ids =  training_set
						else:
							graph_ids =  evaluation_set

						overall_logbook_writer.write ('\n********\ndataset = ' + which_set)

						overall_avg_precision = 0
						overall_avg_recall = 0
						overall_avg_omega = 0
						overall_avg_num_edges_removed = 0
						overall_avg_valid_result = 0
						overall_avg_invalid_result = 0

						overall_avg_termination_tp = 0
						overall_avg_termination_fp = 0
						overall_avg_termination_accuracy = 0

						overall_avg_timeout = 0
						for i in range (NUM_ITER): # repeat 5 times.
							print ('\n\nRound ', i)
							logbook_filename = export_dir + which_method +'_' + selected_UNA+ '_'+ which_source +'_' + selected_weighting_scheme + '_' + which_set +'_Run' + str(i) + '.log'

							avg_precision = 0
							avg_recall = 0
							avg_termination_tp = 0
							avg_termination_fp = 0
							avg_termination_accuracy = 0

							termination_tp = 0
							termination_tn = 0
							termination_fp = 0
							termination_fn = 0

							avg_omega = 0
							num_edges_removed = 0

							count_valid_result = 0
							count_invalid_result = 0
							start = time.time()
							count_graph_no_error_edges = 0
							count_timeout = 0

							for graph_id in graph_ids: # graph_ids:
								print ('\nworking on graph ', graph_id)
								# removed_edge_name  = open( "convert_typeC_progress.tsv", 'w')
								# no_metalink_writer = csv.writer(file_no_metalink, delimiter='\t')
								# no_metalink_writer.writerow(['Count', 'Time'])
								filename_removed_edges = export_dir + which_method +'_' + selected_UNA+ '_'+ which_source +'_' + selected_weighting_scheme + which_set +'_Run' + str(i) + '_Graph' + str(graph_id) + additional + '_removed_edges.tsv'
								edge_writer = csv.writer(open(filename_removed_edges, 'w'), delimiter='\t')
								edge_writer.writerow(['Source', 'Target'])

								# print ('\n\n\ngraph id = ', str(graph_id))
								gold_dir = './gold/'
								gs = GraphSolver(gold_dir, graph_id = graph_id, weighting_scheme = selected_weighting_scheme, source = which_source)

								# gs.show_input_graph()
								# gs.show_gold_standard_graph()
								# gs.show_redirect_graph()
								# gs.show_encoding_equivalence_graph()

								solving_result = gs.solve_SMT(una = selected_UNA)
								if solving_result  == SMT_UNKNOWN:
									count_timeout += 1

								e_result = gs.evaluate_partitioning_result()

								if e_result ['num_edges_removed'] != 0:
									for (s, t) in gs.removed_edges:
										edge_writer.writerow([s, t]) #removed_edges

								if e_result ['num_error_edges_gold_standard'] == 0:
									count_graph_no_error_edges += 1

								avg_omega += e_result['Omega']
								num_edges_removed += e_result['num_edges_removed']
								print ('omega   =', e_result ['Omega'])
								print ('removed ', e_result['num_edges_removed'])

								if e_result['flag'] == 'valid precision and recall':
									p = e_result['precision']
									r = e_result['recall']
									m = e_result ['Omega']
									print ('precision =', p)
									print ('recall   =', r)
									print ('omega = ', m)

									count_valid_result += 1
									avg_precision += e_result['precision']
									avg_recall += e_result['recall']
								else:
									count_invalid_result += 1


								if e_result['num_edges_removed'] == 0:
									if e_result['num_error_edges_gold_standard'] == 0:
										termination_tp += 1
									else:
										termination_fp += 1
								else:
									if e_result['num_error_edges_gold_standard'] == 0:
										termination_fn += 1
									else:
										termination_tn += 1

								avg_termination_tp += termination_tp
								avg_termination_fp += termination_fp

							avg_omega /= len(graph_ids)
							overall_avg_timeout += count_timeout
							# gs.show_result_graph()
							overall_avg_omega += avg_omega
							overall_avg_num_edges_removed += num_edges_removed
							print ('The average Omega: ', avg_omega)
							print ('Count results with precision-recall', count_valid_result)
							print ('Count results without precision-recall', count_invalid_result)
							print ('Count timeout (SMT)', count_timeout)
							print ('Count edges removed', num_edges_removed)
							if count_valid_result > 0:
								avg_precision /= count_valid_result
								avg_recall /= count_valid_result
								print ('*'*20)
								print ('There are ', len (graph_ids), ' graphs in ', which_set)
								print ('   ', count_graph_no_error_edges, ' has no error edge')
								print ('The average precision: ', avg_precision)
								print ('The average recall: ', avg_recall)
								print ('The average Omega: ', avg_omega)
								print ('Total num edges removed ', num_edges_removed)

							overall_avg_precision += avg_precision
							overall_avg_recall += avg_recall



							# if count_invalid_result > 0:
							avg_termination_tp /= len(graph_ids)
							avg_termination_fp /= len(graph_ids)

							overall_avg_termination_tp += avg_termination_tp
							overall_avg_termination_fp += avg_termination_fp
							# overall_avg_termination_accuracy += avg_termination_accuracy

							overall_avg_valid_result += count_valid_result
							overall_avg_invalid_result += count_invalid_result

							end = time.time()
							time_taken += end-start
							hours, rem = divmod(end-start, 3600)
							minutes, seconds = divmod(rem, 60)
							time_formated = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
							print ('time taken: ' + time_formated)

						time_taken /= NUM_ITER
						overall_avg_precision /= NUM_ITER
						overall_avg_recall /= NUM_ITER
						overall_avg_omega /= NUM_ITER
						overall_avg_num_edges_removed /= NUM_ITER
						overall_avg_valid_result /= NUM_ITER
						overall_avg_invalid_result /= NUM_ITER
						overall_avg_termination_tp /= NUM_ITER
						overall_avg_termination_fp /= NUM_ITER
						# overall_avg_termination_accuracy /= NUM_ITER
						overall_avg_timeout /= NUM_ITER

						print ('='*20)
						print ('total number of iterations over the dataset ', NUM_ITER)
						print ('OVERALL There are ', len (graph_ids), ' graphs')
						print ('   ' + str(count_graph_no_error_edges) + ' has no error edge')
						print ('OVERALL Count results precision-recall', count_valid_result)
						print ('OVERALL Count results without precision-recall', count_invalid_result)
						print ('OVERALL COUNT SMT timeout ', overall_avg_timeout)
						print ('OVERALL The average precision: ', overall_avg_precision)
						print ('OVERALL The average recall: ', overall_avg_recall)
						print ('OVERALL The average Omega: ', overall_avg_omega)
						print ('OVERALL The average num edges removed ', overall_avg_num_edges_removed)


						hours, rem = divmod(time_taken, 3600)
						minutes, seconds = divmod(rem, 60)
						time_formated = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
						print ('avg time taken: ' + time_formated)

						overall_logbook_writer.write ('\n\ntotal number of iterations over this dataset ' +str(NUM_ITER))
						overall_logbook_writer.write ('\nOVERALL There are '+str(len (graph_ids)) + ' graphs')
						overall_logbook_writer.write ('   ' + str(count_graph_no_error_edges) + ' has no error edge')
						overall_logbook_writer.write ('\nOVERALL Count results [where precision and recall works] '+ str(count_valid_result))
						overall_logbook_writer.write ('\nOVERALL Count results [where precision and recall do not apply] ' +str(count_invalid_result))
						overall_logbook_writer.write ('\nOVERALL Avg timeout ' +str(overall_avg_timeout))
						overall_logbook_writer.write ('\nOVERALL The average precision: ' +str(overall_avg_precision))
						overall_logbook_writer.write ('\nOVERALL The average recall: '+str(overall_avg_recall))
						overall_logbook_writer.write ('\nOVERALL The average tp [for termination]: ' +str(overall_avg_termination_tp))
						overall_logbook_writer.write ('\nOVERALL The average fp [for termination]: '+str(overall_avg_termination_fp))
						# overall_logbook_writer.write ('\nOVERALL The average accuracy [for termination]: '+str(overall_avg_termination_accuracy))
						overall_logbook_writer.write ('\nOVERALL The average Omega: '+str(overall_avg_omega))
						overall_logbook_writer.write ('\nOVERALL Total num edges removed '+str(overall_avg_num_edges_removed))
						overall_logbook_writer.write ('\n avg time taken: ' + time_formated)



	elif which_method == 'qUNA':
		for selected_UNA in ['qUNA']: # or nUNA
			# for which_source in ['implicit_comment_source']: # 'implicit_comment_source': implicit_label_source
			for selected_weighting_scheme in ['w1', 'w2', 'w3', 'w4', 'w5']: # 'w2',

				additional = ''
				if WITH_WEIGHT:
					additional += '_[weights]'
				if WITH_DISAMBIG:
					additional += '_[disambiguation]'

				overall_logbook_filename = export_dir + which_method +'_' + selected_UNA+ '_' + which_source+ '_'+ selected_weighting_scheme + '_overall' + additional + '.log'
				overall_logbook_writer = open(overall_logbook_filename, 'w')
				overall_logbook_writer.write('\n method = ' + which_method)
				overall_logbook_writer.write('\n UNA = ' + selected_UNA)

				if WITH_WEIGHT:
					overall_logbook_writer.write('\n Additioinal info = weight')
				if WITH_DISAMBIG:
					overall_logbook_writer.write('\n Additioinal info = disambiguation')

				# overall_logbook_writer.write('\n source = ' + which_source)
				overall_logbook_writer.write('\n weighting scheme = ' + selected_weighting_scheme)

				for which_set in [ 'training',  'evaluation']: #'training',
					time_taken = 0
					if which_set == 'training':
						graph_ids =  training_set
					else:
						graph_ids =  evaluation_set

					overall_logbook_writer.write ('\n********\ndataset = ' + which_set)

					overall_avg_precision = 0
					overall_avg_recall = 0
					overall_avg_omega = 0
					overall_avg_num_edges_removed = 0
					overall_avg_valid_result = 0
					overall_avg_invalid_result = 0

					overall_avg_termination_tp = 0
					overall_avg_termination_fp = 0
					overall_avg_termination_accuracy = 0

					overall_avg_timeout = 0
					for i in range (NUM_ITER): # repeat 5 times.
						print ('\n\nRound ', i)
						logbook_filename = export_dir + which_method +'_' + selected_UNA+ '_' +'_' + selected_weighting_scheme + '_' + which_set +'_Run' + str(i) + '.log'

						avg_precision = 0
						avg_recall = 0
						avg_termination_tp = 0
						avg_termination_fp = 0
						avg_termination_accuracy = 0

						termination_tp = 0
						termination_tn = 0
						termination_fp = 0
						termination_fn = 0

						avg_omega = 0
						num_edges_removed = 0

						count_valid_result = 0
						count_invalid_result = 0
						start = time.time()
						count_graph_no_error_edges = 0
						count_timeout = 0

						for graph_id in graph_ids: # graph_ids:
							print ('\nworking on graph ', graph_id)
							# removed_edge_name  = open( "convert_typeC_progress.tsv", 'w')
							# no_metalink_writer = csv.writer(file_no_metalink, delimiter='\t')
							# no_metalink_writer.writerow(['Count', 'Time'])
							filename_removed_edges = export_dir + which_method +'_' + selected_UNA+ '_' +'_' + selected_weighting_scheme + which_set +'_Run' + str(i) + '_Graph' + str(graph_id) + additional + '_removed_edges.tsv'
							edge_writer = csv.writer(open(filename_removed_edges, 'w'), delimiter='\t')
							edge_writer.writerow(['Source', 'Target'])

							# print ('\n\n\ngraph id = ', str(graph_id))
							gold_dir = './gold/'
							gs = GraphSolver(gold_dir, graph_id = graph_id, weighting_scheme = selected_weighting_scheme)

							# gs.show_input_graph()
							# gs.show_gold_standard_graph()
							# gs.show_redirect_graph()
							# gs.show_encoding_equivalence_graph()

							solving_result = gs.solve_SMT(una = selected_UNA)
							if solving_result  == SMT_UNKNOWN:
								count_timeout += 1

							e_result = gs.evaluate_partitioning_result()

							if e_result ['num_edges_removed'] != 0:
								for (s, t) in gs.removed_edges:
									edge_writer.writerow([s, t]) #removed_edges

							if e_result ['num_error_edges_gold_standard'] == 0:
								count_graph_no_error_edges += 1

							avg_omega += e_result['Omega']
							num_edges_removed += e_result['num_edges_removed']
							print ('omega   =', e_result ['Omega'])
							print ('removed ', e_result['num_edges_removed'])

							if e_result['flag'] == 'valid precision and recall':
								p = e_result['precision']
								r = e_result['recall']
								m = e_result ['Omega']
								# print ('precision =', p)
								# print ('recall   =', r)

								count_valid_result += 1
								avg_precision += e_result['precision']
								avg_recall += e_result['recall']
							else:
								count_invalid_result += 1


							if e_result['num_edges_removed'] == 0:
								if e_result['num_error_edges_gold_standard'] == 0:
									termination_tp += 1
								else:
									termination_fp += 1
							else:
								if e_result['num_error_edges_gold_standard'] == 0:
									termination_fn += 1
								else:
									termination_tn += 1

							avg_termination_tp += termination_tp
							avg_termination_fp += termination_fp
							# avg_termination_accuracy += termination_tp /(termination_tp + termination_fp)
							# if (termination_tp + termination_fp) >0 :
							# 	avg_termination_tp = termination_tp / (termination_tp + termination_fp)
							# if (termination_tp+ termination_fn) > 0:
							# 	avg_termination_fp = termination_tp / (termination_tp+ termination_fn)

						# evaluation_result ['num_edges_removed'] = len(self.removed_edges)
						# evaluation_result ['num_error_edges_gold_standard'] = len(self.removed_edges)

						avg_omega /= len(graph_ids)
						overall_avg_timeout += count_timeout
						# gs.show_result_graph()
						overall_avg_omega += avg_omega
						print ('The average Omega: ', avg_omega)
						print ('Count results with precision-recall', count_valid_result)
						print ('Count results without precision-recall', count_invalid_result)
						print ('Count timeout (SMT)', count_timeout)
						if count_valid_result > 0:
							avg_precision /= count_valid_result
							avg_recall /= count_valid_result
							print ('*'*20)
							print ('There are ', len (graph_ids), ' graphs in ', which_set)
							print ('   ', count_graph_no_error_edges, ' has no error edge')
							print ('The average precision: ', avg_precision)
							print ('The average recall: ', avg_recall)
							print ('The average Omega: ', avg_omega)
							print ('Total num edges removed ', num_edges_removed)

						overall_avg_precision += avg_precision
						overall_avg_recall += avg_recall
						overall_avg_num_edges_removed += num_edges_removed


						# if count_invalid_result > 0:
						avg_termination_tp /= len(graph_ids)
						avg_termination_fp /= len(graph_ids)

						overall_avg_termination_tp += avg_termination_tp
						overall_avg_termination_fp += avg_termination_fp
						# overall_avg_termination_accuracy += avg_termination_accuracy

						overall_avg_valid_result += count_valid_result
						overall_avg_invalid_result += count_invalid_result

						end = time.time()
						time_taken += end-start
						hours, rem = divmod(end-start, 3600)
						minutes, seconds = divmod(rem, 60)
						time_formated = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
						print ('time taken: ' + time_formated)

					time_taken /= NUM_ITER
					overall_avg_precision /= NUM_ITER
					overall_avg_recall /= NUM_ITER
					overall_avg_omega /= NUM_ITER
					overall_avg_num_edges_removed /= NUM_ITER
					overall_avg_valid_result /= NUM_ITER
					overall_avg_invalid_result /= NUM_ITER
					overall_avg_termination_tp /= NUM_ITER
					overall_avg_termination_fp /= NUM_ITER
					# overall_avg_termination_accuracy /= NUM_ITER
					overall_avg_timeout /= NUM_ITER

					print ('='*20)
					print ('total number of iterations over the dataset ', NUM_ITER)
					print ('OVERALL There are ', len (graph_ids), ' graphs')
					print ('   ' + str(count_graph_no_error_edges) + ' has no error edge')
					print ('OVERALL Count results precision-recall', count_valid_result)
					print ('OVERALL Count results without precision-recall', count_invalid_result)
					print ('OVERALL COUNT SMT timeout ', overall_avg_timeout)
					print ('OVERALL The average precision: ', overall_avg_precision)
					print ('OVERALL The average recall: ', overall_avg_recall)
					# print ('OVERALL The average tp: [for termination]', overall_avg_termination_tp)
					# print ('OVERALL The average fp: [for termination]', overall_avg_termination_fp)
					# print ('OVERALL The average accuracy: [for termination]', overall_avg_termination_accuracy)
					print ('OVERALL The average Omega: ', overall_avg_omega)
					print ('OVERALL Total num edges removed ', overall_avg_num_edges_removed)

					hours, rem = divmod(time_taken, 3600)
					minutes, seconds = divmod(rem, 60)
					time_formated = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
					print ('avg time taken: ' + time_formated)

					overall_logbook_writer.write ('\n\ntotal number of iterations over this dataset ' +str(NUM_ITER))
					overall_logbook_writer.write ('\nOVERALL There are '+str(len (graph_ids)) + ' graphs')
					overall_logbook_writer.write ('   ' + str(count_graph_no_error_edges) + ' has no error edge')
					overall_logbook_writer.write ('\nOVERALL Count results with precision-recall '+ str(count_valid_result))
					overall_logbook_writer.write ('\nOVERALL Count results without precision-recall ' +str(count_invalid_result))
					overall_logbook_writer.write ('\nOVERALL Avg timeout ' +str(overall_avg_timeout))
					overall_logbook_writer.write ('\nOVERALL The average precision: ' +str(overall_avg_precision))
					overall_logbook_writer.write ('\nOVERALL The average recall: '+str(overall_avg_recall))
					overall_logbook_writer.write ('\nOVERALL The average tp [for termination]: ' +str(overall_avg_termination_tp))
					overall_logbook_writer.write ('\nOVERALL The average fp [for termination]: '+str(overall_avg_termination_fp))
					# overall_logbook_writer.write ('\nOVERALL The average accuracy [for termination]: '+str(overall_avg_termination_accuracy))
					overall_logbook_writer.write ('\nOVERALL The average Omega: '+str(overall_avg_omega))
					overall_logbook_writer.write ('\nOVERALL Total num edges removed '+str(overall_avg_num_edges_removed))
					overall_logbook_writer.write ('\n avg time taken: ' + time_formated)





# --
# gs.get_encoding_equality_graph()
# gs.get_redirect_graph()
# gs.get_namespace_graph()
# gs.get_typeA_graph()
# gs.get_typeB_graph()
# gs.get_typeC_graph()
# gs.add_redundency_weight()

# -- visualization --
# gs.show_input_graph()
# gs.show_redirect_graph()
# gs.show_encoding_equivalence_graph()
# gs.show_namespace_graph()


# -- solve --

# -- show result --
# gs.show_gold_standard_graph()
