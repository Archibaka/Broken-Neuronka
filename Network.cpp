#include "Arduino.h"
#include "Network.h"

    Network::Network(vector<int> s) {
        sizes = s;
        num_layers = (int) sizes.size();
        srand(num_layers);
        for (int ls = 1; ls < num_layers; ls++) {
            vector<vector<long double>> sb;
            vector<vector<long double>> sw;
            for (int nw = 0; nw < sizes[ls]; nw++) {
                vector<long double> ssb;
                vector<long double> ssw;
                for (int pr = 0; pr < sizes[ls - 1]; pr++) {
                    ssb.push_back((2.0 * rand() / RAND_MAX) - 2);
                    ssw.push_back((2.0 * rand() / RAND_MAX) - 2);
                }
                sb.push_back(ssb);
                sw.push_back(ssw);
            }
            biases.push_back(sb);
            weights.push_back(sw);
        }
    }
    Network::Network(vector<int> s, vector<vector<vector<long double>>> b, vector<vector<vector<long double>>> w) {
        sizes = s;
        biases = b;
        weights = w;
    }

    vector<long double> Network::feedForward(vector<long double> in) {
       vector<long double> prev = in;
       vector<long double> out;
        prev.resize(sizes[0], 1);
        for (int ls = 1; ls < num_layers; ls++) {
            for (int nw = 0; nw < sizes[ls]; nw++) {
                long double d = 0;
                for (int pr = 0; pr < sizes[ls - 1]; pr++) {
                        d += specialFunc(prev[pr]*weights[ls-1][nw][pr] + biases[ls-1][nw][pr]);
                }
                out.push_back(d);
            }
            prev = out;
            out.clear();
        }
        return prev;
    }
    long double Network::specialFunc(long double z) {
        return 1 / (1 + exp(-z));
    }