# load the local test.nt file and convert it to a format ready to be handled.
# when a string is there, conver it to ""^^<http://www.w3.org/2001/XMLSchema#string>


# Using readlines()
# file1 = open('test.nt', 'r')
# file2 = open('test_output.nt', 'w')

file1 = open('typeA_f.nt', 'r')
file2 = open('typeA_edited.nt', 'w')


for l in file1.readlines():
	splited = l.split(' ')
	if splited[-2][1] == '"':
		if "^^" in splited[-2]:
			edited = splited[-2][1:-1] # keep that original one
			# example "http://xmlns.com/foaf/0.1/"^^<http://www.w3.org/2001/XMLSchema#anyURI>
		else: # else, add the following
			edited = splited[-2][1:-1] + "^^<http://www.w3.org/2001/XMLSchema#string>"
	new_line = splited[:2]

	new_line.append(edited)

	new_line.append(splited[-1])
	# print ('\n',l)
	# print ('converted to')
	new_line= ' '.join(new_line)
	# print (new_line)
	file2.writelines(new_line)

file2.close()
