import os
import numpy as np
import CTC_model as CTC_model
import Data_processes as Data_processes




"""Function for explicitly testing the trained model"""
def test_model(model_path, image_path, yml_parameters):
	#Loading the .h5 model:
	prediction_model = CTC_model.load_disk_model(model_path)
	#Loading selected image:
	file_name = os.path.basename(image_path)
	#Obtaining additional parameters required:
	symbol_dict, inverse_symbol_dict = Data_processes.retrieve_symbols(yml_parameters)
	
	#Reading image...
	X_test, X_test_len, GT_label = Data_processes.load_image_for_manual_test(image_name = file_name,\
		symbol_dict = symbol_dict, yml_parameters = yml_parameters)


	#Additional vectors for training:
	X_test_len[0] = X_test_len[0]//yml_parameters['architecture']['width_reduction']

	# Test predictions:
	y_test_prediction = prediction_model.predict(
		x = X_test
	)

	#Decoding test predictions (current group):
	result_CTC_Decoding = CTC_model.ctc_manual_decoding(y_test_prediction, X_test_len, yml_parameters)
	CTC_prediction = [inverse_symbol_dict[str(u)] for u in np.array(result_CTC_Decoding[0]) if u != -1]

	#Obtaining metrics:
	SeqER, SymER = CTC_model.error_functions_manual_checking(CTC_prediction = CTC_prediction, true_labels = GT_label)

	#Printing results:
	print("Pred:\t" + ",".join(CTC_prediction))
	print("GT:\t" + ",".join(GT_label))
	print("- Symbol error: " + str(SymER) + "\n- Sequence error: " + str(SeqER))

	return