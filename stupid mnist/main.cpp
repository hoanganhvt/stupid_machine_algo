#include<iostream>
#include<stdio.h>
#include<cmath>


using namespace std;

struct Layer{
	int number_of_neurons;
	double *cur_val;
};

struct Neural_net{
	int number_of_layers;
	Layer *layer_list;
	int *layers_number_of_neuron;
	double ***weights;
	double **bias;
};


double sigmoid(double x){
	double res=1.0/(1.0+exp(-x));
	return res;
}

void init_layer(Layer &l, int number_of_neurons){
	l.number_of_neurons=number_of_neurons;
	l.cur_val=new double[number_of_neurons];
	
	for(int i=0;i<number_of_neurons;i++){
		l.cur_val[i]=((double)rand()/RAND_MAX-0.5)*0.1;
	}
}

void init_net(Neural_net &net, int number_of_layers, int *layer_neuron_nums){
	net.number_of_layers=number_of_layers;
	net.layer_list=new Layer[number_of_layers];
	net.bias=new double*[number_of_layers+1];
	
	for(int i=0;i<number_of_layers;i++){
		init_layer(net.layer_list[i], layer_neuron_nums[i]);
	}
	net.weights=new double**[number_of_layers-1];
	for(int i=0;i<number_of_layers-1;i++){
		net.weights[i]=new double*[layer_neuron_nums[i]];
	}
	for(int i=0;i<number_of_layers-1;i++){
		for(int j=0;j<layer_neuron_nums[i];j++){
			net.weights[i][j]=new double[layer_neuron_nums[i+1]];
		}
	}
	
	for(int i=0;i<number_of_layers-1;i++){
		for(int j=0;j<layer_neuron_nums[i];j++){
			for(int k=0;k<layer_neuron_nums[i+1];k++){
				net.weights[i][j][k]=((double)rand()/RAND_MAX-0.5)*0.1;
			}
		}
	}
	
	for(int i=0;i<number_of_layers;i++){
		net.bias[i]=new double[net.layer_list[i].number_of_neurons];
	}
	
	for(int i=1;i<number_of_layers;i++){
		for(int j=0;j<net.layer_list[i].number_of_neurons;j++){
			net.layer_list[i].cur_val[j]=((double)rand()/RAND_MAX-0.5)*0.1;
		}
	}
}

void forward(Neural_net &net){
	for(int layer=1;layer<net.number_of_layers;layer++){
		for(int i=0;i<net.layer_list[layer].number_of_neurons;i++){
			net.layer_list[layer].cur_val[i]=0;
			for(int k=0;k<net.layer_list[layer-1].number_of_neurons;k++){
				net.layer_list[layer].cur_val[i]=net.layer_list[layer].cur_val[i]+net.layer_list[layer-1].cur_val[k]*net.weights[layer-1][k][i];
			}
			net.layer_list[layer].cur_val[i]=net.layer_list[layer].cur_val[i]+net.bias[layer][i];
			net.layer_list[layer].cur_val[i]=sigmoid(net.layer_list[layer].cur_val[i]);
		}
	}
}


// void fire_to_layer(Neural_net net, Layer &layer_fired_to, double *bias, int layer_fire_idx){
// 	for(int j=0;j<layer_fired_to.number_of_neurons;j++){
// 		layer_fired_to.cur_val[j]=0;
// 		for(int k=0;k<net.layer_list[layer_fire_idx].number_of_neurons;k++){
// 			layer_fired_to.cur_val[j]=layer_fired_to.cur_val[j]+net.weights[layer_fire_idx][k][j]*net.layer_list[layer_fire_idx].cur_val[k];
// 		}
// 		layer_fired_to.cur_val[j]=layer_fired_to.cur_val[j]+bias[j];
// 		layer_fired_to.cur_val[j]=sigmoid(layer_fired_to.cur_val[j]);
// 	}
// }

void back_propagation_1_sample(Neural_net &net, double *test_input, double *test_output, double ***der_weight_of_layers, double **error_of_layer_bias){
	double **error_of_layer;
	error_of_layer=new double*[net.number_of_layers];
	for(int i=0;i<net.number_of_layers;i++){
		error_of_layer[i]=new double[net.layer_list[i].number_of_neurons];
		for(int j=0;j<net.layer_list[i].number_of_neurons;j++){
			error_of_layer[i][j]=0;
		}
	}
	for(int i=0;i<net.layer_list[0].number_of_neurons;i++){
		net.layer_list[0].cur_val[i]=test_input[i];
	}
	
	forward(net);
	
	//calculate the outermost layer error 
	int L=net.number_of_layers-1;
	for(int i=0;i<net.layer_list[L].number_of_neurons;i++){
		error_of_layer[L][i]=(net.layer_list[L].cur_val[i]-test_output[i])*(1-net.layer_list[L].cur_val[i])*net.layer_list[L].cur_val[i];
		error_of_layer_bias[L][i]=error_of_layer[L][i];
	}
	
	//calculate the rest layer error 
	for(L=net.number_of_layers-2;L>=0;L--){
		for(int i=0;i<net.layer_list[L].number_of_neurons;i++){
			for(int j=0;j<net.layer_list[L+1].number_of_neurons;j++){
				error_of_layer[L][i]=error_of_layer[L][i]+net.weights[L][i][j]*error_of_layer[L+1][j];
			}
			error_of_layer[L][i]=error_of_layer[L][i]*net.layer_list[L].cur_val[i]*(1-net.layer_list[L].cur_val[i]);
			error_of_layer_bias[L][i+1]=error_of_layer[L][i];
		}
	}
	
	for(L=net.number_of_layers-2;L>=0;--L){
		for(int i=0;i<net.layer_list[L].number_of_neurons;i++){
			for(int j=0;j<net.layer_list[L+1].number_of_neurons;j++){
				der_weight_of_layers[L][i][j]=error_of_layer[L+1][j]*net.layer_list[L].cur_val[i];
			}
		}
	}
	
	//kill the thing we dont need anymore
	for(int i=0;i<net.number_of_layers;i++){
		delete[] error_of_layer[i];
	}
	delete[] error_of_layer;
}

int main() {
	//this part is written by gemini cuz i fucking lazy to write the test after these horrific codes
	
	// --- 1. NEURAL NETWORK CONFIGURATION ---
	// Input: 784 (28x28 image)
	// Hidden: 128 neurons
	// Output: 10 (digits 0-9)
	int layers_config[] = { 784, 128, 10 };
	int num_layers = 3;

	Neural_net net;
	init_net(net, num_layers, layers_config);

	// --- 2. ALLOCATE MEMORY FOR GRADIENTS ---
	// We need arrays with the exact structure of net.weights and net.bias to store back_propagation results
	double ***der_weights = new double**[num_layers - 1];
	for (int i = 0; i < num_layers - 1; i++) {
		der_weights[i] = new double*[layers_config[i]];
		for (int j = 0; j < layers_config[i]; j++) {
			der_weights[i][j] = new double[layers_config[i + 1]];
		}
	}

	double **der_biases = new double*[num_layers];
	for (int i = 0; i < num_layers; i++) {
		// Allocate a bit extra (+5) to be safe with the indexing logic in the provided back_prop function
		der_biases[i] = new double[layers_config[i] + 5];
	}

	// --- 3. OPEN DATASET FILES ---
	// We need all 4 files: Train images/labels and Test images/labels
	FILE *f_train_img = fopen("train-images.idx3-ubyte", "rb");
	FILE *f_train_lbl = fopen("train-labels.idx1-ubyte", "rb");
	FILE *f_test_img  = fopen("t10k-images.idx3-ubyte", "rb");
	FILE *f_test_lbl  = fopen("t10k-labels.idx1-ubyte", "rb");

	if (!f_train_img || !f_train_lbl || !f_test_img || !f_test_lbl) {
		cout << "ERROR: Could not find all 4 dataset files!" << endl;
		cout << "Please ensure file names match the code and are in the same directory." << endl;
		return 0;
	}

	// Skip headers (16 bytes for images, 8 bytes for labels)
	fseek(f_train_img, 16, SEEK_SET);
	fseek(f_train_lbl, 8, SEEK_SET);
	fseek(f_test_img, 16, SEEK_SET);
	fseek(f_test_lbl, 8, SEEK_SET);

	// --- 4. HELPER VARIABLES ---
	int num_train_samples = 60000;
	int num_test_samples  = 10000;
	double learning_rate  = 0.1;

	unsigned char *buffer_img = new unsigned char[784];
	unsigned char buffer_label;
	
	double *input = new double[784];
	double *target = new double[10];
	int correct_predictions = 0;

	// ==========================================
	// PHASE 1: TRAINING
	// ==========================================
	cout << "=== STARTING TRAINING (60,000 samples) ===" << endl;
	
	for (int n = 0; n < num_train_samples; n++) {
		// Read image and label from Train set
		fread(buffer_img, 1, 784, f_train_img);
		fread(&buffer_label, 1, 1, f_train_lbl);

		// Normalize input (0-255 -> 0.0-1.0)
		for (int i = 0; i < 784; i++) input[i] = (double)buffer_img[i] / 255.0;

		// Create One-hot Target vector (e.g., label 3 -> [0,0,0,1,0,0...])
		for (int i = 0; i < 10; i++) target[i] = 0.0;
		target[buffer_label] = 1.0;

		// 1. Calculate Gradients (Backpropagation)
		back_propagation_1_sample(net, input, target, der_weights, der_biases);

		// 2. Update Weights
		for (int i = 0; i < num_layers - 1; i++) {
			for (int j = 0; j < layers_config[i]; j++) {
				for (int k = 0; k < layers_config[i + 1]; k++) {
					net.weights[i][j][k] -= learning_rate * der_weights[i][j][k];
				}
			}
		}

		// 3. Update Biases
		// (Note: The provided back_prop function calculates bias errors based on the next layer logic)
		int L = num_layers - 1; // Output layer
		for(int i=0; i<layers_config[L]; i++) {
			net.bias[L][i] -= learning_rate * der_biases[L][i];
		}
		// Hidden layers
		for(int layer = num_layers - 2; layer >= 0; layer--) {
			for(int i=0; i<layers_config[layer+1]; i++) {
				// We map the gradient index to the correct bias layer
				if(layer+1 < num_layers)
					net.bias[layer+1][i] -= learning_rate * der_biases[layer][i+1]; 
			}
		}

		// Print progress every 5000 samples
		if ((n + 1) % 5000 == 0) {
			cout << "Finished training " << (n + 1) << " / " << num_train_samples << " samples." << endl;
		}
	}

	cout << "-> Training complete." << endl << endl;

	// ==========================================
	// PHASE 2: TESTING
	// ==========================================
	cout << "=== STARTING TESTING (10,000 samples) ===" << endl;

	for (int n = 0; n < num_test_samples; n++) {
		// Read image and label from Test set
		fread(buffer_img, 1, 784, f_test_img);
		fread(&buffer_label, 1, 1, f_test_lbl);

		// Load Input to the first layer
		for (int i = 0; i < 784; i++) {
			net.layer_list[0].cur_val[i] = (double)buffer_img[i] / 255.0;
		}

		// Forward propagation to predict
		forward(net);

		// Find neuron with max value in the output layer
		int predicted = 0;
		double max_val = -999.0;
		int out_layer_idx = num_layers - 1;

		for (int i = 0; i < 10; i++) {
			if (net.layer_list[out_layer_idx].cur_val[i] > max_val) {
				max_val = net.layer_list[out_layer_idx].cur_val[i];
				predicted = i;
			}
		}

		// Check if prediction matches actual label
		if (predicted == buffer_label) {
			correct_predictions++;
		}
	}

	// ==========================================
	// FINAL RESULTS
	// ==========================================
	cout << "---------------------------------" << endl;
	cout << "FINAL RESULTS:" << endl;
	cout << "Correct predictions: " << correct_predictions << " / " << num_test_samples << endl;
	cout << "Accuracy: " << (double)correct_predictions / num_test_samples * 100.0 << "%" << endl;
	cout << "---------------------------------" << endl;

	// Cleanup memory
	fclose(f_train_img); fclose(f_train_lbl);
	fclose(f_test_img);  fclose(f_test_lbl);
	delete[] buffer_img;
	delete[] input;
	delete[] target;

	return 0;
}
