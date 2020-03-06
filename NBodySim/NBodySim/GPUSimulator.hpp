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
#include <OpenCL/opencl.h>
class GPUSimulator {
private:
    cl_mem xpos;
    cl_mem ypos;
    cl_mem xvel;
    cl_mem yvel;
    cl_mem mass;
    cl_mem xposres;
    cl_mem yposres;
    cl_mem xvelres;
    cl_mem yvelres;
    
public:
    GPUSimulator();
};
#endif /* GPUSimulator_hpp */
