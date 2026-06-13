
Running the experiments:

# Environmental Variables - please install
	python 3.9
	numpy 2.0.2
	pandas 2.3.3


To be able to run the CP adaptations (CPa, CC-CPa, CPC) you need the outputs (prediction probability, prediction label) of the predictions made by the CL-FP models, as well as the true labels of the data instances, as provided in the corresponding ´dataset. 

How to get them:

# Step 1: Reproduce the results of the CL-FP models under study (DeepJIT/ CodeBERT4JIT/LApredict)

# get the FP data
Go to the corresponding repository of the Fault Dataset that you want to use (QT or OpenStack) -- Git repo links in the ReadMe file
Download the data in your local environment

# get the CL-FP model
Go to the corresponding repository of the CL-FP model that you want to work with ( DeepJIT/ CodeBERT4JIT/LApredict ) -- Git repo links in the ReadMe file
Clone the repository in your local environment
make sure to install the environment requirements detailed in the repository
Copy the FP data you downloaded into this repository.
Follow the instructions for running the code (training and evaluation). Compare the accuracy scores with the scores reported in the paper, to ensure that everything runs as it is supposed to do.

# Step 2: Extract the necessary data for running CP and the baseline with the CL-FP models.
Before using the training data (of QT or OpenStack training sets), randomly select 1000 data instances for the calibration set. These instances should then be removed from the training set. 

# Training & evaluating the CL-FP model
-Training: Use the rest of the training set to train the CL-FP model, using 10k cross-validation.
-Validation: When validating the CL-FP model: for each instance in the validation set, use the CL-FP model to make a prediction on it, and extract the corresponding prediction probability (Sigmoid score), the predicted label, and the true label (original label on the dataset) of the instance. Store these data in a Excel file, with the following columns: "Predicted prob", "Predicted label", "True label". 
	Example directory path: "{MODEL}/{DATASET}/validation_set" , e.g., "LApredict/QT/validation_set"
-Testing: repeat the same for each instance in the test set. Extract and store the: "Predicted prob", "Predicted label", "True label", make sure to store them in a separate folder form the validation results. 
	Example directory path: "{MODEL}/{DATASET}/test_set"

# Constructing the calibration set for the Conformal Predictor
-Training the CP model: After the training of the CL-FP model is complete, use the calibration set on the CL-FP model to make predictions - for each instance in the calibration set, use the trained CL-FP model to make predictions, and extract the: "Predicted prob", "Predicted label", "True label", same as  you did for the validation set. Store these values in an excel file. Make sure to separate them from the validation files. 
	Example directory path: "{MODEL}/{DATASET}/calibration_set"

Now you have everything that you need for running the CP adaptations, for example CPa.

# Step 3: run the code of marginal_CPa inside the Ex1 directory. 
Set the global variables correctly:
	DATASET = 'qt' or 'os'
	MODEL = 'LApredict' or 'DeepJIT', or 'CodeBERT'
if you want to extend the study with different datasets and CL-FP models, feel free

Set the CP parameters:
	calib_set_size = 1000  # number of calibration instances for calibrating the CP model
    alpha = 0.05  # (1-alpha) is the desired coverage 
Feel free to adjust these if you wish, just make sure to understand what they are for. 

By setting "testing = True" inside the _main_ CP will be calibrated on the calibration data and evaluated on the test set data. If "testing = False" , CP will be calibrated on the calibration data and evaluated on the validation set data.

Provide the correct directory paths for the validation_set, test_set, and calibration_set data ("Predicted prob", "Predicted label", "True label") that you extracted above. !!! Note that, when running EX2, these directory paths should point to the folders where you stored the data extracted by the calibrated (via Platt-scaling or other recalibration method) CL-FP model, namely the validation_set, test_set, and calibration_set data ("Calibrated_predicted prob", "Predicted label", "True label").

Run the code. 






