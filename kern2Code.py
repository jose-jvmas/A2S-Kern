import re

init_krn = [
	'*clefG2',
	'*k[f#c#g#]',
	'*M3/4',
	'8aL',
	'16cc#y',
	'16bJ',
	'8f#L',
	'8g#y',
	'8f#y',
	'8e#J',
	'=',
	'8f#yL',
	'16a',
	'16g#yJ',
	'8f#yL',
	'8e',
	'8d',
	'8c#yJ',
	'=',
	'8dL',
	'16c#y',
	'16BJ',
	'4e',
	'4e',
	'=',
	'8AL',
	'8e',
	'8f#y',
	'8g#y',
	'8a',
	'8bJ',
	'='
]


clef_CTCFriendly_dict = {
	'clefG2' : 'U',
	'clefC5' : 'V',
	'clefC4' : 'W',
	'clefC3' : 'X',
	'clefF3' : 'Y',
	'clefF4' : 'Z',
	'clefC1' : '$',
	'clefG1' : '@',
	'clefC2' : '&'
}

key_CTCFriendly_dict = {
	'k[b-e-a-d-g-c-f-]'	:	'u',
	'k[b-e-a-d-g-c-]'	:	't',
	'k[b-e-a-d-g-]'	:	'h',
	'k[b-e-a-d-]'	:	'i',
	'k[b-e-a-]'	:	'j',
	'k[b-e-]'	:	'k',
	'k[b-]'	:	'l',
	'k[]':	None,
	'k[f#]'	:	'm',
	'k[f#c#]'	:	'n',
	'k[f#c#g#]'	:	'o',
	'k[f#c#g#d#]'	:	'p',
	'k[f#c#g#d#a#]'	:	'q',
	'k[f#c#g#d#a#e#]'	:	'r',
	'k[f#c#g#d#a#e#b#]'	:	's'
}



meter_CTCFriendly_dict = {
	'*M3/4' : 't',
	'*M8/8' : 'u',
	'*M4/2' : 'v',
	'*M2/1' : 'w',
	'*M2/2' : 'x',
	'*M4/4' : 'y',
	'*M5/4' : 'z',
	'*M12/16' : 'A',
	'*M2/3' : 'B',
	'*M9/8' : 'C',
	'*M3/8' : 'D',
	'*M6/8' : 'E',
	'*M24/16' : 'F',
	'*M6/16' : 'G',
	'*M4/1' : 'H',
	'*M12/8' : 'I',
	'*M6/4' : 'J',
	'*M6/2' : 'K',
	'*M9/16' : 'L',
	'*M7/4' : 'M',
	'*M3/1' : 'N',
	'*M4/8' : 'O',
	'*M2/4' : 'P',
	'*M3/6' : 'Q',
	'*M8/2' : 'R',
	'*M1/2' : 'S',
	'*M3/2' : 'T',
}



"""Processing notes and rests"""
def process_kern_note(note):
	duration_dict = {
		'64'	:	'¿',
		'32'	:	'^',
		'16'	:	'=',
		'8'		:	'*',
		'4'		:	'!',
		'2'		:	'¡',
		'1'		:	'?',
		'0'		:	'+'
	}

	duration = re.findall('[0-9]+', note)

	duration = duration[0] if len(duration) == 1 else '4'


	dot = '.' if '.' in note else ''

	if 'r' not in note :
		#Obtaining pitch:
		pitch_raw = re.findall('[a-gA-G]+', note)[0]
		pitch = "".join(set(pitch_raw.lower()))

		#Obtaining octave:
		octave = ''
		if pitch_raw.isupper():
			octave = str(4 - len(pitch_raw))
		else:
			octave = str(len(pitch_raw) + 3)

		#Obtaining alteration:
		alteration = re.findall('[n#-]+', note)
		
		if len(alteration) > 0:
			pitch += " " + "".join(alteration)

		#return pitch + "".join(alteration) + octave + duration_dict[duration] + dot
		return pitch + " " + octave + " " + duration_dict[duration] + " " + dot
	else:
		return ',' + " " + duration_dict[duration] + " " + dot

def krn2CTCFriendly():
	ctc_friendly = list()

	print(init_krn)

	#CLEF:
	clef = "".join([s for s in init_krn if "clef" in s]).replace('*','')
	ctc_friendly.append(clef_CTCFriendly_dict[clef])

	#KEY:
	key = "".join([s for s in init_krn if "k[" in s]).replace('*','')
	if key != '': ctc_friendly.append(key_CTCFriendly_dict[key])

	#METER:
	meter = "".join([s for s in init_krn if "*M" in s])
	if meter != '': ctc_friendly.append(meter_CTCFriendly_dict[meter])

	music_elements = [music_element for music_element in init_krn if not music_element.startswith('*') and not music_element.startswith('!')]


	for element in music_elements:
		try:
			if '=' in element: #Barline
				ctc_friendly.append('|')
			elif 's' in element: #Slur
				ctc_friendly.append('_')
			else: #Notes
				if not 'q' in element: ctc_friendly.append(process_kern_note(element))
		except:
			pass

	return







if __name__ == '__main__':
	krn2CTCFriendly()