# https://github.com/githubharald/CTCDecoder/tree/master/src

# https://distill.pub/2017/ctc/

import numpy as np

"""Beam entry class"""
class BeamEntry:
	def __init__(self):
		self.probTotal = 0
		self.probNonBlank = 0
		self.probBlank = 0
		self.probText = 1
		self.labeling = tuple()
		self.lmApplied = False
		
"""Beam State class"""
class BeamState:
	#Initializing the state with an empty dictionary:
	def __init__(self):
		self.entries = dict()

	#LM normalization:
	def norm(self):
		"length-normalise LM score"
		for k, _ in self.entries.items():
			labelingLen = len(self.entries[k].labeling)
			self.entries[k].probText = self.entries[k].probText ** (1.0 / (labelingLen if labelingLen else 1.0))
		return
		
	#Sorting according to the probability value:
	def sort(self):
		"return beam-labelings, sorted by probability"
		beams = [v for (_, v) in self.entries.items()]
		sortedBeams = sorted(beams, reverse=True, key=lambda x: x.probTotal*x.probText)
		return [x.labeling for x in sortedBeams]

"""Function for adding a certain beam to the Beam State"""
def addBeam(beamState, labeling):
	"add beam if it does not yet exist"
	if labeling not in beamState.entries:
		beamState.entries[labeling] = BeamEntry()


def applyLM(parentBeam, childBeam, lm):
	"calculate LM score of child beam by taking score from parent beam and bigram probability of last two chars"
	if lm and not childBeam.lmApplied:
		# c1 = parentBeam.labeling[-1] if parentBeam.labeling else classes.index(' ') # first char
		try:
			c1 = parentBeam.labeling[-1] # first char
			c2 = childBeam.labeling[-1] # second char
			lmFactor = 0.01 # influence of language model
			bigramProb = lm[c1][c2] ** lmFactor # probability of seeing first and second char next to each other
			childBeam.probText = parentBeam.probText * bigramProb # probability of char sequence
			childBeam.lmApplied = True # only apply LM once per beam entry
		except:
			pass



"""CTC-based Beam Search"""
def ctc_beam_search(probability_matrix, beam_width = 2, lm = None):

	# First iteration: origin of the different beams:
	lastBeamState = BeamState()
	labeling = tuple()
	lastBeamState.entries[labeling] = BeamEntry()
	lastBeamState.entries[labeling].probBlank = 1
	lastBeamState.entries[labeling].probTotal = 1

	blankIdx = len(probability_matrix[0]) - 1

	for row in probability_matrix:
		currentBeamState = BeamState()

		# get beam-labelings of best beams
		bestLabelings = lastBeamState.sort()[0:beam_width]

		for labeling in bestLabelings:
			# probability of paths ending with a non-blank
			probNonBlank = 0
			# in case of non-empty beam
			if labeling:
				# probability of paths with repeated last char at the end
				probNonBlank = lastBeamState.entries[labeling].probNonBlank * row[labeling[-1]]

			# probability of paths ending with a blank
			probBlank = (lastBeamState.entries[labeling].probTotal) * row[blankIdx]

			# add beam at current time-step if needed
			addBeam(currentBeamState, labeling)


			# fill in data
			currentBeamState.entries[labeling].labeling = labeling
			currentBeamState.entries[labeling].probNonBlank += probNonBlank
			currentBeamState.entries[labeling].probBlank += probBlank
			currentBeamState.entries[labeling].probTotal += probBlank + probNonBlank
			currentBeamState.entries[labeling].probText = lastBeamState.entries[labeling].probText # beam-labeling not changed, therefore also LM score unchanged from
			currentBeamState.entries[labeling].lmApplied = True # LM already applied at previous time-step for this beam-labeling



			# extend current beam-labeling
			for c in range(len(row)-1):
				# add new char to current beam-labeling
				newLabeling = labeling + (c,)

				# if new labeling contains duplicate char at the end, only consider paths ending with a blank
				if labeling and labeling[-1] == c:
					probNonBlank = row[c] * lastBeamState.entries[labeling].probBlank
				else:
					probNonBlank = row[c] * lastBeamState.entries[labeling].probTotal

				# add beam at current time-step if needed
				addBeam(currentBeamState, newLabeling)
				
				# fill in data
				currentBeamState.entries[newLabeling].labeling = newLabeling
				currentBeamState.entries[newLabeling].probNonBlank += probNonBlank
				currentBeamState.entries[newLabeling].probTotal += probNonBlank
				
			 	# apply LM
				applyLM(currentBeamState.entries[labeling], currentBeamState.entries[newLabeling], lm)

		# set new beam state
		lastBeamState = currentBeamState

	# normalise LM scores according to beam-labeling-length
	lastBeamState.norm()

	# sort by probability
	bestLabeling = lastBeamState.sort()[0] # get most probable labeling

	return list(bestLabeling)





"""Main"""
if __name__=='__main__':
	probability_matrix = np.array([
		[0.4, 0.4, 0.4],
		[0.0, 0.0, 0.0],
		[0.6, 0.6, 0.6]
	]).T
	print("Initial sequence: {}".format(ctc_beam_search(probability_matrix)))