//
//  GPUSimulator.hpp
//  NBodySim
//
//  Created by RasmusSamsing on 28/02/2020.
//  Copyright © 2020 Rasmus Samsing. All rights reserved.
//

#ifndef GPUSimulator_hpp
#define GPUSimulator_hpp

#include <stdio.h>
#define __CL_ENABLE_EXCEPTIONS
#include <OpenCL/opencl.h>
class GPUSimulator {
private:
    cl_mem xpos;
    cl_mem ypos;
    cl_mem zpos;
    cl_mem xvel;
    cl_mem yvel;
    cl_mem zvel;
    cl_mem mass;
    cl_mem xposres;
    cl_mem yposres;
    cl_mem zposres;
    cl_mem xvelres;
    cl_mem yvelres;
    cl_mem zvelres;
    cl_mem n_particles;
    cl_mem xaccPredicted;
    cl_mem yaccPredicted;
    cl_mem zaccPredicted;
    cl_mem xjerkPredicted;
    cl_mem yjerkPredicted;
    cl_mem zjerkPredicted;
    cl_mem x0acc;
    cl_mem y0acc;
    cl_mem z0acc;
    cl_mem x0jerk;
    cl_mem y0jerk;
    cl_mem z0jerk;
    cl_mem xposp;
    cl_mem yposp;
    cl_mem zposp;
    cl_mem xvelp;
    cl_mem yvelp;
    cl_mem zvelp;
    
    cl_command_queue command_queue;
    cl_context context;
    cl_device_id* device_list;
    cl_kernel kernel;
    cl_kernel accPredictor;
    
    float* xposarr;
    float* yposarr;
    float* zposarr;
    float* xvelarr;
    float* yvelarr;
    float* zvelarr;
    float* massarr;
    
    float** prevxs;
    float** prevys;
    float** prevzs;
    
    int particle_count = 100;
    int time_steps = 5000;
public:
    GPUSimulator();
    void runGen();
    void setState ( float* xpos,
                    float* ypos,
                    float* zpos,
                    float* xvel,
                    float* yvel,
                    float* zvel,
                    float* mass );
    void setInitialState();
    void appendState(int timestep);
    void printState();
    void saveStateToFile ();
    void getStoredData (float*** xarr, float*** yarr, float*** zarr);
};
#endif /* GPUSimulator_hpp */
