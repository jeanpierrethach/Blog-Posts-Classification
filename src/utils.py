def read_word_list(file_name):
	"""
	This function reads a list of words and return it as a set

	Parameter:
			file_name
	Returns a set of the words
	"""
	with open(file_name) as word_list_file:
		return set(word.strip() for word in word_list_file)