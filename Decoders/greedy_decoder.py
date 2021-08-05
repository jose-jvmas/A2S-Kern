import numpy as np


"""Best path CTC decoding"""
def greedy_decoder(probability_matrix):
	decoded_sequence = list()
	
	#Looping over temporal slices to analyze:
	for array in probability_matrix:
		#Estimated symbol:
		decoded_value = [np.where(array.max() == array)[0][0] if np.where(array.max() == array)[0][0] != len(array) -1 else -1]

		#Appending symbol:
		decoded_sequence.extend(decoded_value)
	return decoded_sequence


"""Main"""
if __name__=='__main__':
	probability_matrix = np.array([
		[0.4, 0.4, 0.4],
		[0.0, 0.0, 0.0],
		[0.6, 0.6, 0.6]
	]).T
	print("Initial sequence: {}".format(greedy_decoder(probability_matrix)))
