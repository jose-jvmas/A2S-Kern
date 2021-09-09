import os
import argparse
import numpy as np
import CTC_model as CTC_model
import Data_processes as Data_processes
from results_class import *
import copy
import shutil
import audio_extraction

"""Train CRNN-CTC model"""
def train_model(yml_parameters, model, prediction_model, symbol_dict, inverse_symbol_dict):
	#Listing files:
	train_files, val_files, test_files = Data_processes.list_files(yml_parameters)

	#Creating paths (if necessary):
	#Model path:
	if not os.path.exists(yml_parameters['path_to_model']):
		os.mkdir(yml_parameters['path_to_model'])

	#Weights path:
	if not os.path.exists(yml_parameters['path_to_weights']):
		os.mkdir(yml_parameters['path_to_weights'])

	#Fitting model:
	best_results = ResultsClass()
	current_iteration_results = ResultsClass()
	steps_per_epoch = len(train_files)//yml_parameters['batch_size']
	for it_super_epoch in range(yml_parameters['super_epochs']):
		print("\n--- Super epoch #{} ---".format(it_super_epoch + 1))
		for inner_epoch in range(yml_parameters['epochs']):
			history = model.fit(
				x = Data_processes.data_generator_train(files = train_files,\
					symbol_dict = symbol_dict, yml_parameters = yml_parameters),
				steps_per_epoch = steps_per_epoch
			)

		# /VALIDATION\ #
		current_iteration_results.error_dict['Validation']['SeqER'], current_iteration_results.error_dict['Validation']['SymER'],\
			current_iteration_results.error_dict['Validation']['SeqER_kern'], current_iteration_results.error_dict['Validation']['SymER_kern'] = evaluate_set(files = val_files,\
			yml_parameters = yml_parameters, prediction_model = prediction_model, symbol_dict = symbol_dict,\
			inverse_symbol_dict = inverse_symbol_dict)
		# \VALIDATION/ #
	
		# /TEST\ #
		current_iteration_results.error_dict['Test']['SeqER'], current_iteration_results.error_dict['Test']['SymER'],\
			current_iteration_results.error_dict['Test']['SeqER_kern'], current_iteration_results.error_dict['Test']['SymER_kern'] = evaluate_set(files = test_files,\
			yml_parameters = yml_parameters, prediction_model = prediction_model, symbol_dict = symbol_dict,\
			inverse_symbol_dict = inverse_symbol_dict)
		# \TEST/ #

		#Saving best obtained models & weights:
		if current_iteration_results.error_dict['Validation']['SymER'] < best_results.error_dict['Validation']['SymER']:

			#Saving the models with the best SER rates:
			best_results = copy.deepcopy(current_iteration_results)
			best_results.best_epoch = it_super_epoch
			prediction_model.save(os.path.join(yml_parameters['path_to_model'], yml_parameters['name'] + '.h5'))
			model.save_weights(os.path.join(yml_parameters['path_to_weights'], yml_parameters['name'] + '.h5'))


		#Printing results:
		print("\n\t </-->\tRESULTS\t<--\>")
		print(current_iteration_results)

		print("\nBEST VALIDATION RESULTS => epoch {}".format(best_results.best_epoch+1))
		print(best_results)
		print("\t <\-->\tRESULTS\t<--/>")

	return



"""Function for testing the trained model on a set of data"""
def evaluate_set(files, prediction_model, yml_parameters, symbol_dict, inverse_symbol_dict):
	SeqER_error_list = list()
	SymER_error_list = list()
	SeqER_kern_error_list = list()
	SymER_kern_error_list  = list()

	init_index = 0
	while init_index < len(files):
		
		end_index = min(init_index + yml_parameters['batch_size'], len(files))

		#Loading data:
		X, Y, X_len, Y_len = Data_processes.load_selected_range(init_index = init_index, end_index = end_index,\
										files = files, symbol_dict = symbol_dict, yml_parameters = yml_parameters)

		#Additional vectors for training:
		input_length_train = np.zeros([X.shape[0],], dtype='int64')
		for i in range (X.shape[0]):
			input_length_train[i] = X_len[i]//yml_parameters['architecture']['width_reduction']

		# Predictions (current group):
		y_prediction = prediction_model.predict(
			x = X
		)

		#Decoding test predictions (current group):
		result_CTC_Decoding = CTC_model.ctc_manual_decoding(y_prediction, input_length_train, yml_parameters)

		#Figures of merit:
		SeqER_error, SymER_error, SeqER_kern_error, SymER_kern_error = CTC_model.error_functions_batch(result_CTC_Decoding, Y, Y_len, inverse_symbol_dict, files[init_index:end_index])

		SeqER_error_list.extend(SeqER_error)
		SymER_error_list.extend(SymER_error)
		SeqER_kern_error_list.extend(SeqER_kern_error)
		SymER_kern_error_list.extend(SymER_kern_error)


		#Updating index:
		init_index = end_index

	#Figures of merit:
	SeqER_error = 100*sum(SeqER_error_list)/len(files)
	SymER_error = 100*sum(SymER_error_list)/len(files)
	SeqER_kern_error = 100*sum(SeqER_kern_error_list)/len(files)
	SymER_kern_error = 100*sum(SymER_kern_error_list)/len(files)

	return SeqER_error, SymER_error, SeqER_kern_error, SymER_kern_error



# """Function for testing the trained model on a set of data"""
# def evaluate_set_OLD(files, prediction_model, yml_parameters, symbol_dict, inverse_symbol_dict):
# 	result_CTC_Decoding_global = list()
# 	Y_global = list()
# 	Y_length_global = list()

# 	init_index = 0
# 	while init_index < len(files):
		
# 		end_index = min(init_index + yml_parameters['batch_size'], len(files))

# 		#Loading data:
# 		X, Y, X_len, Y_len = Data_processes.load_selected_range(init_index = init_index, end_index = end_index,\
# 										files = files, symbol_dict = symbol_dict, yml_parameters = yml_parameters)

# 		#Additional vectors for training:
# 		input_length_train = np.zeros([X.shape[0],], dtype='int64')
# 		for i in range (X.shape[0]):
# 			input_length_train[i] = X_len[i]//yml_parameters['architecture']['width_reduction']

# 		# Predictions (current group):
# 		y_prediction = prediction_model.predict(
# 			x = X
# 		)

# 		#Storing length of the expected sequences:
# 		Y_length_global.extend(Y_len)

# 		#Decoding test predictions (current group):
# 		result_CTC_Decoding = CTC_model.ctc_manual_decoding(y_prediction, input_length_train, yml_parameters)

# 		#Extending list for evaluating the results:
# 		result_CTC_Decoding_global.extend(result_CTC_Decoding)
# 		Y_global.extend(Y)

# 		#Updating index:
# 		init_index = end_index

# 	#Figures of merit:
# 	SeqER_error, SymER_error, SeqER_kern_error, SymER_kern_error = CTC_model.error_functions(result_CTC_Decoding_global, Y_global, Y_length_global, inverse_symbol_dict)

# 	return SeqER_error, SymER_error, SeqER_kern_error, SymER_kern_error