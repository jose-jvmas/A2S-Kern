

class ResultsClass:
	def __init__(self):
		self.best_epoch = -1
		self.error_dict = {
			'Validation' : {
				'SymER' : float('inf'),
				'SeqER' : float('inf')
			},
			'Test': {
				'SymER' : float('inf'),
				'SeqER' : float('inf')
			},
		}
		return

	def __str__(self):
		out = "VALIDATION\t\t=> SymER: {:.2f} % - SeqER: {:.2f} % ".format(self.error_dict['Validation']['SymER'], self.error_dict['Validation']['SeqER'])
		out += "\nTEST\t\t\t=> SymER: {:.2f} % - SeqER: {:.2f} %".format(self.error_dict['Test']['SymER'], self.error_dict['Test']['SeqER'])
		return out



"""Main"""
if __name__=='__main__':

	#Folder to process:
	results = ResultsClass()
	print(results)
