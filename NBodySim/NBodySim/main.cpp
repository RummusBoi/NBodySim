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
#include <chrono>

using namespace std;

int main(int argc, const char * argv[]) {
    
    const int timesteps = 1000;
    const int particle_count = 256;
    const int stepsPerSavedFrame = 50;
    GPUSimulator sim = GPUSimulator(timesteps / stepsPerSavedFrame, particle_count);
    cout << "setup done, setting state" << endl;
    sim.setInitialState();
    
    auto t1 = chrono::high_resolution_clock::now();
    
    float initEnergy = sim.calcTotalEnergy();
    
    for (int i = 0; i < timesteps; i++) {
        if(i % stepsPerSavedFrame == 0) {
            sim.appendState(i / stepsPerSavedFrame);
            cout << i / (double)timesteps * 100 << "% done" << endl;
            //cout << "Energy at step " << i << ": " << sim.calcTotalEnergy() << endl;
        }
        sim.runGen();
    }
    
    float finalEnergy = sim.calcTotalEnergy();
    
    cout << "Change in energy: " << (finalEnergy - initEnergy) / initEnergy * 100 << "%" << endl;
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    cout << "Completed simulation in " << duration << " ms." << endl;
    
    
    
    double** xarrs,** yarrs,** zarrs;
    sim.getStoredData(&xarrs, &yarrs, &zarrs);
    
    
    DataPlotter dataplotter = DataPlotter(1000, 1000);
    
    cout << "shit" << endl;
    dataplotter.draw3DData(xarrs, yarrs, zarrs, particle_count, timesteps / stepsPerSavedFrame);

    return 0;
}
