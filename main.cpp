#include "myMatrix.hpp"
#include "neuralException.hpp"
#include "neuralLayer.hpp"

#include <iostream>

using namespace std;

int main() {
    try {
        cout << "Vytvaram neuronovu vrstvu (3 vstupy -> 2 neurony)..." << endl;
        NeuralLayer vrstva(3, 2, RELU);

        MyMatrix vstupy(1, 3, MyMatrix::ZERO);
        vstupy.setValue(0, 0, 1.0);
        vstupy.setValue(0, 1, -2.5);
        vstupy.setValue(0, 2, 3.5);

        cout << "\n--- VSTUPNE DATA (Rozmer 1x3) ---" << endl;
        vstupy.print();

        cout << "\n--- SPUSTAM FORWARD PASS ---" << endl;
        MyMatrix vystup = vrstva.forwardPass(vstupy);

        cout << "\n--- VYSTUP Z VRSTVY (Rozmer 1x2) ---" << endl;
        vystup.print();

        cout << "\nSkuska dopadla uspesne!" << endl;

    } catch (NeuralException &e) {
        cerr << e.what() << endl;
    }
    return 0;
}