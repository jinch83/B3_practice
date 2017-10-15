#include <iostream>
#include <random>
#include <vector>

//file system
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

class Model_multi
{
  public:
    Model_multi(int, double);
    ~Model_multi();
    void add_position(double*, double);
    void create_diff();
    void loss_func();
    void update();

    void saveMode(string);
    void readMode(string);

  private:
    vector<double*> _x;
    vector<double> _y;  // ground truth
    vector<double> _weight;
    double *_diff;
    double _lr; // learning rate
    int _n;  // n-dimention
    vector<double> _loss;
    // addON
    bool isUse;
};

Model_multi::Model_multi(int n, double lr){
  _lr = lr;
  _n = n+1;
  isUse = false;
}

Model_multi::~Model_multi(){
  if(isUse){
    delete[] _diff;
    for(int i=0; i < _x.size(); i++){
      delete[] _x[i];
    }
  }
}

void Model_multi::add_position(double* data, double truth){
  _x.push_back(data);
  _y.push_back(truth);
}

void Model_multi::create_diff(){
  isUse = true;
  _diff = new double[_n*_x.size()];
  for(int i=0; i < _n; i++){
    _weight.push_back(10.0); // random or setup your rule!
    for(int j=i*_x.size(); j < _x.size()+i*_x.size(); j++){
      _diff[j] = 0.0;
    }
  }
}

void Model_multi::loss_func(){
  double error = 0;

  for(int i=0; i < _y.size(); i++){
    double diff = 0;
    double sum = 0;
    double* x = _x[i];
    for(int j=0; j < _n; j++){
      sum = sum + x[j] * _weight[j];
      _diff[j + _n*i] = x[j];
    }

    diff = _y[i] - sum;
    error += diff * diff;

    for(int k=0; k < _n; k++){
      _diff[k + _n*i] = _diff[k + _n*i] * diff;
    }
  }
  error = error / 2;
  _loss.push_back(error);

  cout << "loss = " << error << endl;
}

void Model_multi::update() {
  for(int i=0; i < _n; i++){
    double sum = 0;
    for(int j=0; j < _y.size(); j++){
      sum = sum + _diff[i + j*_n];
    }

    _weight[i] = _weight[i] + _lr*sum;
    cout << "weight[" << i << "]=" << _weight[i] << endl;
  }
}

void Model_multi::saveMode(string filename) {
  ofstream ofs(filename);
  for(int i=0; i < _loss.size(); i++){
    ofs << i << "," << _loss[i] << endl;
  }

  ofs.close();
}

void Model_multi::readMode(string filename) {
  ifstream reading_file;

  reading_file.open(filename, ios::in);
  string read_line_buffer;

  cout << "reading " << filename << "..." << endl;
  while(getline(reading_file, read_line_buffer)){
    istringstream stream(read_line_buffer);
    int num = 0;
    string lines;
    double truth = 0;
    double *database = new double[_n];
    while(getline(stream, lines, ',')){
      if(num == _n){
        truth = stof(lines);
        break;
      }
      database[num] = stof(lines);
      num++;
    }
    add_position(database, truth);
    //delete[] database;
  }
}

int main() {
  Model_multi model(3, 0.01); // (dimention, learning rate)

  model.readMode("data.csv");
  model.create_diff();
  cout << "debug" << endl;
  int learning_count = 0;
  while(learning_count < 10){
    cout << "iter: " << learning_count << endl;
    model.loss_func();
    model.update();
    learning_count++;
    cout << endl;
  }
  model.saveMode("out.csv");
}
