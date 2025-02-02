#ifndef Network_h
#define Network_h

#include <ArduinoSTL.h>
#include "Arduino.h"
using namespace std;
class Network
{
  public:
    int num_layers;
    vector<vector<vector<long double>>> biases;
    vector<vector<vector<long double>>> weights;
    vector<int> sizes;
    Network(vector<int> s);
    Network(vector<int> s, vector<vector<vector<long double>>> b, vector<vector<vector<long double>>> w);
    vector<long double> feedForward(vector<long double> in);
    long double specialFunc(long double z);
};

#endif