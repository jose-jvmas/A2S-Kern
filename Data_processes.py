import os
import cv2
import json
import random
import numpy as np
import CTC_model as CTC_model
import audio_extraction

"""Function for listing the data partitions inside the selected fold"""
def list_files(yml_parameters):

	#Train partition:
	with open(os.path.join(yml_parameters['path_to_partitions'],'Train.lst')) as f:
		train_files = [u + yml_parameters['src_extension'] for u in f.read().splitlines()]

	#Validation partition:
	with open(os.path.join(yml_parameters['path_to_partitions'],'Val.lst')) as f:
		val_files = [u + yml_parameters['src_extension'] for u in f.read().splitlines()]

	#Test partition:
	with open(os.path.join(yml_parameters['path_to_partitions'],'Test.lst')) as f:
		test_files = [u + yml_parameters['src_extension'] for u in f.read().splitlines()]

	return train_files, val_files, test_files


"""Create symbol data"""
def create_symbol_data(yml_parameters):

	#Initializing variables:	
	symbol_dict = dict()
	inverse_symbol_dict = dict()
	complete_symbol_list = list()

	files = os.listdir(yml_parameters['path_to_GT'])
	files = [f for f in files if f.endswith(yml_parameters['GT_extension'])]

	for single_file in files:
		with open(os.path.join(yml_parameters['path_to_GT'],single_file)) as f:
			current_sequence = f.readlines()[0].split()
			complete_symbol_list.extend(current_sequence)

	#Obtaining the unique symbols (as a set):
	symbol_list = sorted(list(set(complete_symbol_list)))

	###- Original datum to encoded value:
	symbol_dict = {u:symbol_list.index(u) for u in symbol_list}
	###- Encoded value to original datum:
	inverse_symbol_dict = {str(v): k for k, v in symbol_dict.items()}

	#Create the JSON file with the dictionary:
	with open(os.path.join(yml_parameters['path_to_corpus'], 'symbol_dict.json'), 'w') as f: 
		json.dump(symbol_dict, f)

	#Create the JSON file with the inverse dictionary:
	with open(os.path.join(yml_parameters['path_to_corpus'], 'inverse_symbol_dict.json'), 'w') as f: 
		json.dump(inverse_symbol_dict, f)

	return symbol_dict, inverse_symbol_dict


"""Read symbol data (from file)"""
def retrieve_symbols(yml_parameters):
	#Read the JSON file with the dictionary:
	with open(os.path.join(yml_parameters['path_to_corpus'], 'symbol_dict.json')) as f: 
		symbol_dict = json.load(f)

	#Read the JSON file with the inverse dictionary:
	with open(os.path.join(yml_parameters['path_to_corpus'], 'inverse_symbol_dict.json')) as f: 
		inverse_symbol_dict = json.load(f)

	return symbol_dict, inverse_symbol_dict

"""Load specific range of data"""
def load_selected_range(init_index, end_index, files, symbol_dict, yml_parameters):

	X = list()
	Y = list()
	X_len = list()
	Y_len = list()
	for itFile in range(init_index, end_index):
		#Audios:
		file_in = np.array(audio_extraction.get_x_from_file(os.path.join(yml_parameters['path_to_audios'], files[itFile])))

		#Adapt and normalize:
		file_in = np.flip(np.transpose(file_in),0)
		file_in = (file_in - np.amin(file_in)) / (np.amax(file_in) - np.amin(file_in))
		init_width = file_in.shape[1]

		X.append(np.expand_dims(file_in, file_in.ndim))
		X_len.append(init_width)

		#Labels:
		##-Label itself:
		f = open(os.path.join(yml_parameters['path_to_GT'], files[itFile].split(yml_parameters['src_extension'])[0] + yml_parameters['GT_extension']))
		temp_symbols_list = f.readlines()[0].split()
		f.close()

		temp_symbol_array = np.array([symbol_dict[u] for u in temp_symbols_list])
		Y_len.append(len(temp_symbol_array))
		Y.append(temp_symbol_array)

	#Padding Y to the longest GT of the batch:
	max_len_Y = max(Y_len)
	for Y_it in range(len(Y)):
		temp = (len(symbol_dict) + 1) * np.ones(max_len_Y-len(Y[Y_it]),dtype='int64')
		Y[Y_it] = np.concatenate((Y[Y_it], temp), axis = 0)

	#Padding X to the largest image width of the batch:
	max_len_X = max(X_len)
	for X_it in range(len(X)):
		temp = 0*np.ones((X[0].shape[0], max_len_X - X_len[X_it], 1))
		X[X_it] = np.concatenate((X[X_it], temp), axis = 1)


	#Casting lists to Numpy arrays:
	X = np.array(X,dtype='float64')
	Y = np.array(Y)
	X_len = np.array(X_len, dtype='int64')
	Y_len = np.array(Y_len, dtype='int64')

	return X, Y, X_len, Y_len




"""Train data generator"""
def data_generator_train(files, symbol_dict, yml_parameters):

	#Shuffle file list:
	random.shuffle(files)

	#Init index:
	aux_index = 0

	#Infinite loop for getting all the data:
	while aux_index <  len(files):
		#Loading the selected range of data for current batch:
		X_train, Y_train, X_len, Y_len = load_selected_range(init_index = aux_index,\
			end_index = min(aux_index + yml_parameters['batch_size'], len(files)), files = files,\
			symbol_dict = symbol_dict, yml_parameters = yml_parameters)

		#Additional vectors for training:
		input_length_train = np.zeros([X_train.shape[0], 1], dtype='int64')
		label_length_train = np.zeros([X_train.shape[0], 1], dtype='int64')
		for i in range (X_train.shape[0]):
			input_length_train[i] = X_len[i]//yml_parameters['architecture']['width_reduction']
			label_length_train[i] = min(Y_len[i], input_length_train[i]) #THIS WAY WE ALLOW LONGER OUTPUTS THAN INPUTS
			# label_length_train[i] = Y_len[i] #NOT ALLOWING LONGER OUTPUTS THAN INPUTS

		#Data structure for the input vectors:
		inputs_fit = {'image': X_train,
					'y_true': Y_train,
					'input_length': input_length_train,
					'label_length': label_length_train
					}
		#Output vector:
		outputs_train = {'ctc': np.zeros([X_train.shape[0]])} 
		
		#Updating index:
		aux_index = min(aux_index + yml_parameters['batch_size'], len(files))

		#Yielding data (tuple of input-output vectors):
		yield(inputs_fit, outputs_train)



"""Load image for manual test"""
def load_image_for_manual_test(image_name, symbol_dict, yml_parameters):
	
	X_test = list()
	Y_test = list()

	#Images:
	file_img = cv2.imread(os.path.join(yml_parameters['path_to_images'], image_name))
	init_img = (255. - cv2.cvtColor(file_img, cv2.COLOR_BGR2GRAY))/255
	temp_img, init_width = adapt_images_aspect_ratio(img = init_img, new_height = yml_parameters['user_max_height'],\
		new_width =  yml_parameters['user_max_width'])

	X_test.append(np.expand_dims(temp_img, temp_img.ndim))
	X_len = [init_width]

	#Labels:
	##-Label itself:
	f = open(os.path.join(yml_parameters['path_to_GT'], image_name.split(yml_parameters['src_extension'])[0] + yml_parameters['GT_extension']))
	temp_symbols_list = f.readlines()[0].split()
	f.close()
	Y_test = temp_symbols_list
	Y_len = [len(temp_symbols_list)]

	#Casting lists to Numpy arrays:
	X_test = np.array(X_test,dtype='float64')

	return X_test, X_len, Y_test


"""Obtain confidence matrices for each data partition"""
def obtain_confidence_matrices(model_path, symbol_dict, yml_parameters):

	#Confidence matrices path:
	if not os.path.exists(yml_parameters['path_to_confidence_matrices']):
		os.mkdir(yml_parameters['path_to_confidence_matrices'])

	#Loading the .h5 model:
	prediction_model = CTC_model.load_disk_model(model_path)

	#Listing files:
	train_files, val_files, test_files = list_files(yml_parameters)

	###-Listing and removing previous existing files:
	for single_file in os.listdir(yml_parameters['path_to_confidence_matrices']):
		os.remove(os.path.join(yml_parameters['path_to_confidence_matrices'], single_file))

	###-Train confidence matrices:
	prediction_confidence_matrices(files = train_files, yml_parameters = yml_parameters,\
		prediction_model = prediction_model, symbol_dict = symbol_dict, partition = 'train')

	###-Validation confidence matrices:
	prediction_confidence_matrices(files = val_files, yml_parameters = yml_parameters,\
		prediction_model = prediction_model, symbol_dict = symbol_dict, partition = 'val')

	###-Test confidence matrices:
	prediction_confidence_matrices(files = test_files, yml_parameters = yml_parameters,\
		prediction_model = prediction_model, symbol_dict = symbol_dict, partition = 'test')

	return


"""Write the confidence matrices into a file"""
def write_confidence_matrices(yml_parameters, partition, files, init_index, end_index, input_length_train, y_prediction):

	with open(os.path.join(yml_parameters['path_to_confidence_matrices'], 'confMatrix-' + partition), 'a+') as f:
		i = 0
		for index in range(init_index, end_index):
			f.write("{} [\n".format(files[index].split(".")[0]))
			for time_index in range(input_length_train[i]):
				f.write(" ".join([str(u) for u in np.log(y_prediction[i][time_index])]) + '\n')
			f.write("]\n")
			i += 1
	return


"""Obtain confidence matrices for train data"""
def prediction_confidence_matrices(files, prediction_model, yml_parameters, symbol_dict, partition):

	init_index = 0
	while init_index < len(files):
		
		end_index = min(init_index + yml_parameters['batch_size'], len(files))

		#Loading data:
		X, Y, X_len, Y_len = load_selected_range(init_index = init_index, end_index = end_index,\
										files = files, symbol_dict = symbol_dict, yml_parameters = yml_parameters)

		#Additional vectors for training:
		input_length_train = np.zeros([X.shape[0],], dtype='int64')
		for i in range (X.shape[0]):
			input_length_train[i] = X_len[i]//yml_parameters['architecture']['width_reduction']

		# Predictions (current group):
		y_prediction = prediction_model.predict(
			x = X
		)

		#Writing confidence matrices into files:
		write_confidence_matrices(yml_parameters, partition, files, init_index, end_index, input_length_train, y_prediction)

		#Updating index:
		init_index = end_index

	return