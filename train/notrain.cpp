#pragma GCC optimize("O3")
#pragma GCC target("avx2")
#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <string>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <math.h>
#include <stdexcept>
using namespace std;
using namespace std::chrono;
using namespace std::filesystem;
class Network
{
public:
    int num_layers;
    vector<vector<vector<long double>>> biases;
    vector<vector<vector<long double>>> weights;
    vector<int> sizes;
    bool sigma;
    Network(vector<int> s, bool sg, long double smaller) {
        sizes = s;
        vector<vector<vector<long double>>> b;
        vector<vector<vector<long double>>> w;
        if (s.size() < 3) {
            sizes.resize(3, 2);
        }
        num_layers = sizes.size();
        srand(num_layers + rand());
        for (int ls = 1; ls < num_layers; ls++) {
            vector<vector<long double>> sb;
            vector<vector<long double>> sw;
            for (unsigned int nw = 0; nw < sizes[ls]; nw++) {
                vector<long double> ssb;
                vector<long double> ssw;
                for (int64_t pr = 0; pr < sizes[ls - 1]; pr++) {
                    ssb.insert(ssb.end(), smaller * ((2.0 * rand() / RAND_MAX) - 2) * (ls != 1));
                    ssw.insert(ssw.end(), smaller * ((2.0 * rand() / RAND_MAX) - 2) / sqrt(ls));
                }
                sb.insert(sb.end(), ssb);
                sw.insert(sw.end(), ssw);
            }
            b.insert(b.end(), sb);
            w.insert(w.end(), sw);
        }
        biases = b;
        weights = w;
        sigma = sg;
    }
    Network(string src, vector<int> szs, bool sg) {
        num_layers = szs.size();
        ifstream file;
        file.open(src);
        string line;
        string buf;
        if (file.is_open()) {
            vector<int>size;
            vector<vector<vector<long double>>> txt;
            vector<vector<long double>> subtxt;
            vector<long double> subsubtxt;
            getline(file, line);
            for (char c : line) {
                if (c == '\t') {
                    size.push_back(atoll(buf.c_str()));
                    buf.clear();
                    buf.shrink_to_fit();
                }
                else {
                    buf.push_back(c);
                }
            }
            sizes = size;
            num_layers = static_cast<int>(size.size());
            while (file) {
                getline(file, line);
                if (*line.c_str() != static_cast<char>(2) && *line.c_str() != static_cast<char>(3)) {
                    if (*line.c_str() != static_cast<char>(13)) {
                        if (*line.c_str() != static_cast<char>(69)) {
                            for (char c : line) {
                                if (c == static_cast<char>(32)) {
                                    subsubtxt.push_back(stold(buf));
                                    buf.clear();
                                }
                                else {
                                    buf.push_back(c);
                                }
                            }
                        }
                        else
                        {
                            subtxt.push_back(subsubtxt);
                            subsubtxt.clear();
                            subsubtxt.shrink_to_fit();
                        }
                    }
                    else {
                        txt.push_back(subtxt);
                        subtxt.clear();
                        subtxt.shrink_to_fit();
                    }
                }
                else {
                    if (*line.c_str() != static_cast<char>(3)) {
                        biases = txt;
                    }
                    else {
                        weights = txt;
                    }
                    txt.clear();
                    txt.shrink_to_fit();
                }
            }
            file.close();
        }
        if (biases.size() == 0 || weights.size() == 0 || sizes.size() == 0) {
            Network costil = Network(szs, true, 1);
            num_layers = costil.num_layers;
            biases = costil.biases;
            weights = costil.biases;
            sizes = costil.sizes;
        }
        sigma = sg;
    }
    vector<long double> feedForward(vector<long double> in) {
        vector<long double> prev = in;
        vector<long double> out;
        prev.resize(static_cast<int>(sizes[0]), 1);
        for (int64_t ls = 1; ls < num_layers; ls++) {
            for (int64_t nw = 0; nw < sizes[ls]; nw++) {
                long double d = 0;
                for (int64_t pr = 0; pr < sizes[ls - 1]; pr++) {
                    long double c = specFunc(prev[pr] * weights[ls - 1][nw][pr] + biases[ls - 1][nw][pr]);
                    if (isnan(c)) {
                        throw invalid_argument("feedforward");
                    }
                    d += c;
                }
                out.insert(out.end(), d);
            }
            prev = out;
            out.clear();
            out.shrink_to_fit();
        }
        return prev;
    }
    void train(long double eta, vector<vector<vector<long double>>> data, int mbsize, int hm, int step, long double lmbda) {
        vector<vector<vector<vector<long double>>>> miniBatches;
        for (int e = 0; e < hm; e++) {
            //shuffle and pack
            shuffle(begin(data), end(data), default_random_engine{});
            vector<vector<vector<long double>>> t;
            for (unsigned int s = 0; s < data.size(); s++) {
                if (data[s].size() != 0) {
                    data[s].resize(2);
                    data[s][0].resize(static_cast<int>(sizes[0]));
                    data[s][1].resize(static_cast<int>(sizes[sizes.size() - 1]));
                    t.insert(t.end(), data[s]);
                    if (s % mbsize == 0 && s != 0) {
                        miniBatches.insert(miniBatches.end(), t);
                        t.clear();
                        t.shrink_to_fit();
                    }
                }
            }
            //update MiniBatch
            if (miniBatches.size() != 0) {
                for (unsigned int m = 0; m < miniBatches.size(); m++) {
                    upd(eta, miniBatches[m], lmbda, static_cast<int>(data.size()));
                    if (step == 2) {
                        cout << e + 1 << "/" << hm << ": " << m + 1 << "/" << miniBatches.size() << "\n";
                    }

                }
                miniBatches.clear();
                //save
                if (sigma) {
                    save("neuronka.txt");
                }
                else {
                    save("RLU.txt");
                }
                if (step == 1) {
                    cout << "Epoch {" << e + 1 << "} :  " << gitgood(data) << "/" << data.size() << "\n";
                    for (long double r : feedForward(data[0][0])) {
                        cout << r << "\n";
                    }
                }
            }
        }
    }
    void upd(long double eta, vector<vector<vector<long double>>> mb, long double lmbda, int sz) {
        //biases.size() = weights.size() = num_layers
        vector<vector<vector<long double>>> nbb = shape(biases);
        vector<vector<vector<long double>>> nbw = shape(weights);
        vector<vector<vector<vector<long double>>>> d;
        for (unsigned int i = 0; i < mb.size(); i++) {
            d = backprop(mb[i][0], mb[i][1]);
            for (unsigned int m = 0; m < nbb.size(); m++) {
                for (unsigned int n = 0; n < nbb[m].size(); n++) {
                    for (unsigned int s = 0; s < nbb[m][n].size(); s++) {
                        nbb[m][n][s] += d[0][m][n][s];
                        nbw[m][n][s] += d[1][m][n][s];
                    }
                }
            }
            d.clear();
            d.shrink_to_fit();
        }
        vector<vector<vector<long double>>> changeW;
        vector<vector<vector<long double>>> changeB;
        for (unsigned int i = 0; i < num_layers - 1; i++) {
            vector<vector<long double>> w;
            vector<vector<long double>> b;
            for (unsigned int j = 0; j < weights[i].size(); j++) {
                vector<long double> ww;
                vector<long double> bb;
                for (unsigned int k = 0; k < weights[i][j].size(); k++) {
                    ww.insert(ww.end(), (1 - eta * (lmbda / sz)) * weights[i][j][k] - (eta / mb.size()) * nbw[i][j][k]);
                    bb.insert(bb.end(), biases[i][j][k] - (eta / mb.size()) * nbb[i][j][k]);
                }
                w.insert(w.end(), ww);
                b.insert(b.end(), bb);
            }
            changeW.insert(changeW.end(), w);
            changeB.insert(changeB.end(), b);
        }
        weights = changeW;
        biases = changeB;
        changeW.clear();
        changeB.clear();
        changeW.shrink_to_fit();
        changeB.shrink_to_fit();
    }
    vector<vector<vector<vector<long double>>>>backprop(vector<long double> a, vector<long double> got) {
        vector<vector<vector<vector<long double>>>> spitted;
        vector<vector<vector<long double>>> nbb = shape(biases);
        vector<vector<vector<long double>>> nbw = shape(weights);
        //feedforward
        vector<long double> act = a;
        act.resize(static_cast<int>(sizes[0]), 1);
        vector<vector<long double>> acts;
        acts.insert(acts.end(), act);
        vector<vector<long double>> zs;
        zs.insert(zs.end(), inverse(act));
        vector<long double> z;
        for (int ls = 1; ls < num_layers; ls++) {
            for (int nw = 0; nw < sizes[ls]; nw++) {
                long double d = 0;
                for (int pr = 0; pr < sizes[ls - 1]; pr++) {
                    long double c = act[pr] * weights[ls - 1][nw][pr] + biases[ls - 1][nw][pr];
                    if (isnan(c)) {
                        throw invalid_argument("backprop");
                    }
                    d += c;
                }
                z.insert(z.end(), d);
            }
            zs.insert(zs.end(), z);
            act = specFunc(z);
            acts.insert(acts.end(), act);
            z.clear();
            z.shrink_to_fit();
        }

        // backward pass
        //for the very last layer
        vector<long double> err = delt(got, acts[acts.size() - 1]);
        vector<long double> ders = der(zs[zs.size() - 2]);
        vector<vector<long double>> delta = mult(ders, err); //multiplying the error to the input of layer
        nbb[nbb.size() - 1] = delta;
        nbw[nbw.size() - 1] = dotProd(acts[acts.size() - 2], delta); //dot product with activation from the same layer

        for (int l = 2; l < num_layers; l++) {
            //for the l-1 layer before the last
            vector<long double> n = dotProd(transpose(weights[weights.size() - l + 1]), transpose(delta)); //express the error in terms of the output of the layer
            delta = transpose(mult(n, der(zs[zs.size() - l - 1]))); //multiplying the error to the input of layer
            nbb[nbb.size() - l] = delta;
            nbw[nbw.size() - l] = dotProd(acts[acts.size() - l - 1], delta);
        }
        spitted.push_back(nbb);
        spitted.push_back(nbw);
        return spitted;
    }
    long double gitgood(vector<vector<vector<long double>>> testData) {
        int rate = 0;
        for (unsigned int i = 0; i < testData.size(); i++) {
            rate += (maxArg(feedForward(testData[i][0])) == maxArg(testData[i][1]));
        }
        return rate;
    }

    vector<long double> delt(vector<long double> a, vector<long double> got) {
        vector<long double> res;
        for (unsigned int gg = 0; gg < got.size(); gg++) {
            res.insert(res.end(), a[gg] - got[gg]);
        }
        return res;
    }

    vector<long double> specFunc(vector<long double> z) {
        vector<long double> s;
        for (unsigned int i = 0; i < z.size(); i++) {
            if (sigma) {
                long double c = specFunc(z[i]);
                if (isnan(c)) {
                    throw invalid_argument("special func");
                }
                s.insert(s.end(), c);
            }
            else {
                s.insert(s.end(), RLU(z[i]));
            }
        }
        return s;
    }
    vector<long double> inverse(vector<long double> z) {
        vector<long double> s;
        for (unsigned int i = 0; i < z.size(); i++) {
            if (sigma) {
                long double c = inverse(z[i]);
                if (isnan(c)) {
                    throw invalid_argument("inverse");
                }
                s.insert(s.end(), c);
            }
            else {
                s.insert(s.end(), RLU(z[i]));
            }
        }
        return s;
    }
    vector<long double> der(vector<long double> z) {
        vector<long double> sig = specFunc(z);
        vector<long double> s;
        for (unsigned int i = 0; i < z.size(); i++) {
            if (sigma) {
                long double c = 2 * sig[i] * (1 - sig[i]);
                if (isnan(c)) {
                    throw invalid_argument("derivative");
                }
                s.insert(s.end(), c);
            }
            else {
                s.insert(s.end(), (static_cast<int>(z[i] > 0)));
            }
        }
        return s;
    }
    //sigmoid
    long double specFunc(long double z) {
        return 2 / (1 + exp(-z)) - 1;
    }
    long double inverse(long double z) {
        return log((1 + z) / (1 - z));
    }
    //RLU
    long double RLU(long double z) {
        return static_cast<int>(z > 0) * z;
    }
    //numpy funcs analogs
    int maxArg(vector<long double> a) {
        int max = 0;
        for (int i = 0; i < a.size(); i++) {
            if (a[max] < a[i]) {
                max = i;
            }
        }
        return max;
    }

    vector<vector<long double>> transpose(vector<vector<long double>> st) {
        vector<vector<long double>> ed;
        ed.resize(st[0].size());
        for (unsigned int i = 0; i < st[0].size(); i++) {
            ed[i].resize(st.size());
        }
        for (unsigned int i = 0; i < st[0].size(); i++) {
            for (unsigned int j = 0; j < st.size(); j++) {
                ed[i][j] = st[j][i];
            }
        }
        return ed;
    }

    vector<vector<long double>> dotProd(vector<long double> f, vector<vector<long double>> s) {
        vector<vector<long double>> result;
        vector<long double> res;
        vector<long double> n = f;
        n.resize(s[0].size());
        for (unsigned int i = 0; i < s.size(); i++) {
            for (unsigned int j = 0; j < s[i].size(); j++) {
                res.insert(res.end(), n[j] * s[i][j]);
            }
            result.insert(result.end(), res);
            res.clear();
            res.shrink_to_fit();
        }

        return result;
    }
    vector<long double> dotProd(vector<vector<long double>>f, vector<vector<long double>> s) {
        vector<long double> result;
        s.resize(s.size() + static_cast<int>(s.size() < f.size()) * ((int)f.size() - (int)s.size()));
        f.resize(f.size() + static_cast<int>(f.size() < s.size()) * ((int)s.size() - (int)f.size()));
        for (unsigned int i = 0; i < f.size(); i++) {
            long double n = dotProd(f[i], s[i]);
            result.insert(result.end(), n);
        }
        return result;
    }
    long double dotProd(vector<long double> f, vector<long double> s) {
        long double result = 0;
        s.resize(s.size() + static_cast<int>(s.size() < f.size()) * ((int)f.size() - (int)s.size()));
        f.resize(f.size() + static_cast<int>(f.size() < s.size()) * ((int)s.size() - (int)f.size()));
        for (unsigned int i = 0; i < f.size(); i++) {
            result += f[i] * s[i];
        }

        return result;
    }

    vector<vector<long double>> shape(vector<vector<long double>> in) {
        vector<vector<long double>> out;
        out.resize(in.size());
        for (unsigned int s = 0; s < out.size(); s++) {
            out[s].resize(in[s].size());
        }
        return out;
    }
    vector<vector<vector<long double>>> shape(vector<vector<vector<long double>>> in) {
        vector<vector<vector<long double>>> out;
        out.resize(in.size());
        for (unsigned int s = 0; s < out.size(); s++) {
            out[s] = shape(in[s]);
        }
        return out;
    }

    vector<vector<long double>> mult(vector<long double> f, vector<long double> s) {
        vector<vector<long double>> result;
        for (unsigned int i = 0; i < s.size(); i++) {
            vector<long double> t;
            for (unsigned int j = 0; j < f.size(); j++) {
                t.insert(t.end(), f[j] * s[i]);
            }
            result.insert(result.end(), t);
            t.clear();
            t.shrink_to_fit();
        }
        return result;
    }

    void save(string as) {
        ofstream file;
        file.open(as);
        for (int i : sizes) {
            file << i << '\t';
        }
        file << "\n";
        for (vector<vector<long double>> x : biases) {
            for (vector<long double> y : x) {
                for (long double z : y) {
                    file << z << " ";
                }
                file << "\n" << static_cast<char>(69) << "\n";
            }
            file << "\n" << static_cast<char>(13) << "\n";
        }
        file << "\n" << static_cast<char>(2) << "\n";
        for (vector<vector<long double>> x : weights) {
            for (vector<long double> y : x) {
                for (long double z : y) {
                    file << z << " ";
                }
                file << "\n" << static_cast<char>(69) << "\n";
            }
            file << "\n" << static_cast<char>(13) << "\n";
        }
        file << static_cast<char>(3) << "\n";
        file.close();
    }
    vector<vector<vector<long double>>> loaddata(string path, vector<int> szs, int best) {
        vector<vector<vector<long double>>> result;
        vector<vector<long double>> sr;
        vector<long double> in;
        ifstream file;
        file.open(path);
        string line;
        string buf;
        int count = 0;
        if (file.is_open()) {
            while (file) {
                vector<long double> out(static_cast<int>(szs[szs.size() - 1]));
                getline(file, line);
                for (char c : line) {
                    if (c == '\t') {
                        if (count <= szs[0] && count > 0) {
                            long double x = stold(string(buf.c_str()));
                            in.insert(in.end(), x);
                            buf.clear();
                            count++;
                        }
                        else {
                            if (count == 0) {
                                count++;
                                buf.clear();
                            }
                        }
                    }
                    else {
                        buf.insert(buf.end(), c);
                    }
                }
                if (isdigit(atoll(buf.c_str()))) {
                    out[atoll(buf.c_str())] = best;
                }
                buf.clear();
                count = 0;
                sr.insert(sr.end(), in);
                sr.insert(sr.end(), out);
                result.insert(result.end(), sr);
                sr.clear();
                in.clear();
                out.clear();
            }
            file.close();
        }
        return result;
    }
};
#pragma GCC optimize("O3")
#pragma GCC target("avx2")
void u(Network& net, vector<string> paths, string saveAs, vector<int> st, int best, long double eta, int howMany, int MiniBatchSize, long double lambda, int ZeroIsSilentOneIsInformativeTwoIsAll) {
    int counter = 1;
    shuffle(begin(paths), end(paths), default_random_engine{});
    vector<vector<vector<long double>>> val;
    for (string p : paths) {
        cout << counter << "/" << paths.size() << ":" << "\n" << "\n";
        val = net.loaddata(p, st, best);
        shuffle(begin(val), end(val), default_random_engine{});
        net.train(eta, val, MiniBatchSize, howMany, ZeroIsSilentOneIsInformativeTwoIsAll, lambda);
        counter++;
        val.clear();
        val.shrink_to_fit();
    }
}
int main() {
    vector<int> st = { 8, 69, 69, 10 };
    string SaveAs = "neuronka.txt";
    string samps = "C:\\Users\\mirop\\Desktop\\Lapa1337\\newdat";
    bool sigmoid = SaveAs == "neuronka.txt";
    Network x = Network(st, sigmoid, 1);
    vector<string> paths;
    int ct = 0;
    for (const auto& dirEntry : recursive_directory_iterator(samps)) {
        string a = dirEntry.path().string();
        ct++;
        if (a[a.size() - 1] == 't') {
            paths.push_back(a);
        }
    }
    shuffle(begin(paths), end(paths), default_random_engine{ 69 });
    long double lambda = 58055*169;
    long double eta = 0.0069;
    u(x, paths, SaveAs, st, 2, eta, 70, 100, lambda, 0);
    chrono::duration<long double> z = system_clock::now().time_since_epoch();
    cout << z.count();
}