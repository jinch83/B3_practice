#include <iostream>
#include <random>
#include <vector>
// file system
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

class Model_2D
{
  public:
    Model_2D(double, double, double);
    ~Model_2D();
    void add_position(double, double);
    void create_diff();
    void loss_func();
    void update();
    //add on!
    void saveMode(string);

  private:
    vector<double> _x;
    vector<double> _y;
    double _a;
    double _b;
    double *_diff;
    bool isUse;
    double _lr; // learning rate
    //add on!!
    vector<double> _loss;
};

Model_2D::Model_2D(double a, double b, double lr) {
  _a = a;
  _b = b;
  _lr = lr;
  isUse = false;
}

Model_2D::~Model_2D(){
  if(isUse)
    delete[] _diff;
}

void Model_2D::add_position(double x, double y) {
  _x.push_back(x);
  _y.push_back(y);
}

void Model_2D::create_diff(){
  _diff = new double[_x.size()];
  for(int cont=0; cont < _x.size(); cont++)
    _diff[cont] = 0;
  isUse = true;
}

void Model_2D::loss_func(){
  double error = 0;
  double diff = 0;
  for(int i=0; i < _x.size(); i++) {
    diff = (_y[i] - _a*_x[i] - _b);
    _diff[i] = diff;
    error += diff * diff; // sigma(y_predict - y)^2 / 2
  }
  _loss.push_back(error/2);
  cout << "loss = " << error/2 << endl;
}

void Model_2D::update() {
  double diff_a = 0;
  double diff_b = 0;
  for(int i=0; i < _x.size(); i++){
    diff_a += _diff[i] * _x[i];
    diff_b += _diff[i];
  }

  _a = _a + _lr * diff_a;
  _b = _b + _lr * diff_b;
  cout << "new_a = " << _a << endl;
  cout << "new_b = " << _b << endl;
}

void Model_2D::saveMode(string filename){
  ofstream ofs(filename);
  for(int i=0; i < _loss.size(); i++){
    ofs << i << "," << _loss[i] << endl;
  }

  ofs.close();
}

int main() {
  Model_2D model(3.0, 2.0, 0.01);
  model.add_position(1.0, 3.2);
  model.add_position(5.0, 10.8);
  model.add_position(2.2, 5.0);
  model.add_position(10.1, 21.0);
  model.add_position(7.0, 14.7);

  //create
  model.create_diff();
  int learning_count = 0;
  while(learning_count < 50) {
    cout << "iter: " << learning_count << endl;
    model.loss_func();
    model.update();
    learning_count++;
    cout << endl;
  }
  model.saveMode("out.csv");
}
