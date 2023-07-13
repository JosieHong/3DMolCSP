ATOM_LIST = ['C', 'H', 'O', 'N', 'F', 'S', 'Cl', 'P', 'B', 'Br', 'I']

def convert2cls(chir, csp_category): 
	if csp_category == '1': 
		# For polysaccharide CSPs:
		if chir < 1.15:
			y = 0
		elif chir < 1.2:
			y = 1
		elif chir < 2.1:
			y = 2
		else:
			y = 3
	elif csp_category == '2': 
		# For Pirkle CSPs:
		if chir < 1.05: 
			y = 0
		elif chir < 1.15:
			y = 1
		elif chir < 2: 
			y = 2
		else:
			y = 3
	else:
		raise Exception("The category for CSP should be 1 or 2, rather than {}.".format(csp_category))
	return y