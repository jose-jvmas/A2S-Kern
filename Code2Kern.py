

# Pred:   *clefG2,*k[b-e-a-],*M3/8,Pb,5,4,Pe,PA-,5,8,PA-,8,Pc,6,8,Pb,5,8,Pa,5,8,Pa,5,8,16,5,16,Pg,5,16
# GT:     *clefG2,*k[b-e-a-],*M3/4,Pb,PA-,5,4,Pe,PA-,5,8,Pb,PA-,5,8,Pc,6,8,Pb,PA-,5,8,=,Pa,PA-,5,8,Pa,PA-,5,8,Pr,16,Pa,PA-,5,16,Pg,5,16,Pa,PA-,5,16,Pb,PA-,5,16,Pa,PA-,5,16,Pg,5,16,Pa,PA-,5,16,=






def decode_prediction(input_seq):
	out_seq = list()

	it = 0

	while it < len(input_seq):
		print(input_seq[it])


		if "clef" in input_seq[it]: #CLEFs:
			out_seq.append(input_seq[it])
			it += 1

		elif "k[" in input_seq[it]: #KEYs:
			out_seq.append(input_seq[it])
			it += 1

		elif "*M" in input_seq[it]: #METERs:
			out_seq.append(input_seq[it])
			it += 1

		elif "=" in input_seq[it]: #SILENCEs:
			out_seq.append(input_seq[it])
			it += 1

		elif "s" in input_seq[it]: #SLURs:
			out_seq.append(input_seq[it])
			it += 1

		else: #NOTEs:
			if len(input_seq[it]) == 2 and input_seq[it][0] == 'P' and input_seq[it][1] != 'A':
				# Note case:
				out_note = list()
				
				# Pitch:
				out_note.append(input_seq[it][1])
				it += 1

				# Alteration?
				if 'PA' in input_seq[it]:
					alteration = input_seq[it][2]
					out_note.append(alteration)
					it += 1
				
				# Octave?
				if 'O' in input_seq[it]:
					octave = input_seq[it][1]
					out_note.append(octave)
					it += 1

				# Duration?
				if 'D' in input_seq[it]:
					duration = input_seq[it][1]
					out_note.append(duration)
					it += 1

				# Duration alteration?
				if 'DA' in input_seq[it]:
					duration_alteration = input_seq[it][2]
					out_note.append(duration_alteration)
					it += 1
			
			if out_note[0] != 'r':
				note = pitchOctave2Kern(out_note[0], octave)
				note = duration + note

				print(f'{out_note} <-> {note}')



			out_seq.append(out_note)
			
	return



def pitchOctave2Kern(in_pitch, in_octave):
	# #Obtaining octave:
	# octave = ''
	# if pitch_raw.isupper():
	# 	octave = str(4 - len(pitch_raw))
	# else:
	# 	octave = str(len(pitch_raw) + 3)

	# Extracting octave:
	octave = int("".join(in_octave.split("O")[1:]))

	if octave < 4:
		return "".join([in_pitch.upper() for u in range(4-octave)])
	else:
		return "".join([in_pitch.lower() for u in range(octave - 3)])

	







# return 'P'+pitch, 'PA' + str(alteration), 'O'+str(octave), 'D'+str(duration), 'DA'+dot

if __name__ == '__main__':
	print("hello")
	GT = ['*clefG2','*k[b-e-a-]','*M3/4','Pb','PA-','O5','D4','Pe','PA-','O5','D8','Pb','PA-','O5','D8','Pc','O6','D8','Pb','PA-','O5','D8','=','Pa','PA-','O5','D8','Pa','PA-','O5','D8','Pr','D16','Pa','PA-','O5','D16','Pg','O5','D16','Pa','PA-','O5','D16','Pb','PA-','O5','D16','Pa','PA-','O5','D16','Pg','O5','D16','Pa','PA-','O5','D16','=']
	
	decode_prediction(GT)
	# pitchOctave2Kern('g','O7')

	