# sameas_laundromat_metalink.hdt

# <https://krr.triply.cc/krr/metalink/fileMD5/00c48eee800fedbe0e1e5679c35dc5d5> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <https://krr.triply.cc/krr/metalink/def/File> .

from hdt import HDTDocument, IdentifierPosition

file_name = "sameas_laundromat_metalink.hdt"

type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
metalink_file = "https://krr.triply.cc/krr/metalink/def/File"

hdt_weight = HDTDocument(file_name)
(triples, cardinality) = hdt_weight.search_triples("", type, metalink_file)

print ('There are ', cardinality, 'knowledge bases with sameAs triples')

count = 0
for t, _, _ in triples:
	count += 1
	print (t)
	if count > 10:
		break
