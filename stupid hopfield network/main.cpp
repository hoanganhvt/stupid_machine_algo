#include<bits/stdc++.h>

using namespace std;

struct Hopfield_net{
	int number_of_neurals;
	int** weight_list;
	int *current_state;
};

void init_net(Hopfield_net &H, int N){
	H.number_of_neurals=N;
	H.current_state=new int[N];
	H.weight_list=new int*[N];
	for(int i=0;i<N;i++){
		H.weight_list[i]=new int[N];
	}
	
	
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			H.weight_list[i][j]=0;
		}
	}
}

void learn_from_patterns(Hopfield_net &net, vector<vector<int>>patterns){
	for(int P=0;P<patterns.size();P++){
		for(int i=0;i<net.number_of_neurals;i++){
			for(int j=0;j<net.number_of_neurals;j++){
				if(i!=j){
					net.weight_list[i][j]=net.weight_list[i][j]+patterns[P][i]*patterns[P][j];
				}
			}
		}
	}
}

void neuron_fire(Hopfield_net &net, int neuron_index, int thresh_hold){
	int res=0;
	for(int i=0;i<net.number_of_neurals;i++){
		if(i==neuron_index) continue;
		res=res+net.weight_list[neuron_index][i]*net.current_state[i];
	}
	net.current_state[neuron_index]=(res>=thresh_hold)?1:-1;
}

int energy_level(Hopfield_net net){
	int res=0;
	for(int i=0;i<net.number_of_neurals;i++){
		for(int j=0;j<net.number_of_neurals;j++){
			if(i==j) continue;
			res=res+net.weight_list[i][j]*net.current_state[i]*net.current_state[j];
		}
	}
	return -res/2;
}

void run_model(Hopfield_net net,int max_iter){
	int old_energy=energy_level(net);
	for(int i=0;i<max_iter;i++){
		for(int i=0;i<net.number_of_neurals;i++){
			neuron_fire(net,i,0);
		}
		int new_energy=energy_level(net);
		if(new_energy>=old_energy) break;
	}
}


int main(){
	int number_on_1_line=0;
	//inpuut images
	vector<int>x1;
	vector<int>x2;
	
	ifstream inputfile("cat1.txt");
	string cur_s;
	for(int k=0;k<40;k++){
		inputfile>>cur_s;
		for(int i=0;i<cur_s.length();i++){
			if(cur_s[i]=='0'){
				x1.push_back(-1);
			}
			else{
				x1.push_back(1);
			}
		}
	}
	inputfile.close();
	
	inputfile.open("cat2.txt");
	for(int k=0;k<40;k++){
		inputfile>>cur_s;
		number_on_1_line=cur_s.length();
		for(int i=0;i<cur_s.length();i++){
			if(cur_s[i]=='0'){
				x2.push_back(-1);
			}
			else{
				x2.push_back(1);
			}
		}
	}
	
	//create and train the fucking model
	Hopfield_net net;
	init_net(net,x1.size());
	vector<vector<int>>pattern_list;
	pattern_list.push_back(x1);
	pattern_list.push_back(x2);
	inputfile.close();
	learn_from_patterns(net,pattern_list);
	
	for(int i=0;i<x1.size();i++){
		net.current_state[i]=0;
	}
	
	run_model(net,10);
	for(int i=0;i<x1.size();i++){
		if(net.current_state[i]==-1){
			cout<<"0";
		}
		else{
			cout<<"1";
		}
		if((i+1)%number_on_1_line==0) cout<<"\n";
	}
}
