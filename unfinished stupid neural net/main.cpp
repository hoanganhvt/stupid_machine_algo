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

void back_propagation(Neural_net &net, double *test_input, double *test_output, double ***der_weight_of_layers, double **error_of_layer_bias){
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

int main(){
	Neural_net net;
	int net_shape[]={1,64,64,1};
	init_net(net,4,net_shape);
	double *input=new double[1];
	input[0]=0.1;
	double *output=new double[1];
	output[0]=0.2;
	
	forward(net);
	cout<<net.layer_list[3].cur_val[0]<<endl;
	
	double ***der_weight_of_layers=new double**[3];
	double **error_of_layer_bias=new double*[4];
	for(int i=0;i<4;i++){
		error_of_layer_bias[i]=new double[net_shape[i]];
	}
	for(int i=0;i<3;i++){
		der_weight_of_layers[i]=new double*[net_shape[i]];
		for(int j=0;j<net_shape[i];j++){
			der_weight_of_layers[i][j]=new double[net_shape[i+1]];
		}
	}
	for(int L=0;L<3;L++){
		for(int i=0;i<net_shape[L];i++){
			for(int j=0;j<net_shape[L+1];j++){
				der_weight_of_layers[L][i][j]=0;
			}
		}
	}
	back_propagation(net, input, output, der_weight_of_layers, error_of_layer_bias);
	
	for(int L=0;L<3;L++){
		for(int i=0;i<net_shape[L];i++){
			for(int j=0;j<net_shape[L+1];j++){
				net.weights[L][i][j]-=der_weight_of_layers[L][i][j];
			}
		}
	}
	
	for(int L=0;L<3;L++){
		for(int i=0;i<net_shape[L];i++){
			net.bias[L][i]-=error_of_layer_bias[L][i];
		}
	}
	
	forward(net);
	cout<<net.layer_list[3].cur_val[0]<<endl;
	// for(int L=0;L<3;L++){
	// 	for(int i=0;i<net_shape[L];i++){
	// 		for(int j=0;j<net_shape[L+1];j++){
	// 			cout<<der_weight_of_layers[L][i][j]<<" ";
	// 		}
	// 		cout<<endl;
	// 	}
	// 	cout<<"-----"<<endl;
	// }
}
