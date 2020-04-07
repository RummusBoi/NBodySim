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
#include<CL/cl.h>
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
    
    double* xposarr;
    double* yposarr;
    double* zposarr;
    double* xvelarr;
    double* yvelarr;
    double* zvelarr;
    double* massarr;
    
    double** prevxs;
    double** prevys;
    double** prevzs;
    
    int iteration;
    
    int particle_count;
    int time_steps;
public:
    GPUSimulator(int time_steps, int particle_count);
    void runGen();
    void setState ( double* xpos,
                    double* ypos,
                    double* zpos,
                    double* xvel,
                    double* yvel,
                    double* zvel,
                    double* mass );
    void setInitialState();
    void appendState(int timestep);
    void printState();
    void saveStateToFile ();
    void getStoredData (double*** xarr, double*** yarr, double*** zarr);
    float calcTotalEnergy();
};
#endif /* GPUSimulator_hpp */
