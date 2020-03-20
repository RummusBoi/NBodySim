//
//  main.cpp
//  NBodySim
//
//  Created by RasmusSamsing on 28/02/2020.
//  Copyright © 2020 Rasmus Samsing. All rights reserved.
//
//#include<OpenCL/OpenCL.h>
#include "GPUSimulator.hpp"
#include "DataPlotter.hpp"
#include <iostream>

using namespace std;

int main(int argc, const char * argv[]) {
    
    GPUSimulator sim = GPUSimulator();
    cout << "setup done, setting state" << endl;
    sim.setInitialState();
    int timesteps = 500;
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timesteps; i++) {
        
        sim.appendState(i);
        sim.runGen();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    
    cout << "Completed simulation in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " ms." << endl;
    
    float** xarrs,** yarrs,** zarrs;
    sim.getStoredData(&xarrs, &yarrs, &zarrs);
    
    DataPlotter dataplotter = DataPlotter(1400, 1400);
    
    cout << "shit" << endl;
    dataplotter.draw3DData(xarrs, yarrs, zarrs, 2, timesteps);
    return 0;
}
