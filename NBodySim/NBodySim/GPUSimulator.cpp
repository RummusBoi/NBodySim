//
//  GPUSimulator.cpp
//  NBodySim
//
//  Created by RasmusSamsing on 28/02/2020.
//  Copyright © 2020 Rasmus Samsing. All rights reserved.
//

#include "GPUSimulator.hpp"
//#include <OpenCL/OpenCL.h>

GPUSimulator::GPUSimulator () {
    uint particle_count = 10;
    // insert code here...
    // Get platform and device information
    cl_platform_id * platforms = NULL;
    cl_uint     num_platforms;
    
    //Set up the Platform
    cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
    platforms = (cl_platform_id *)
    malloc(sizeof(cl_platform_id)*num_platforms);
    clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

    //Get the devices list and choose the device you want to run on
    cl_device_id     *device_list = NULL;
    cl_uint           num_devices;

    clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &num_devices);
    device_list = (cl_device_id *)
    malloc(sizeof(cl_device_id)*num_devices);
    clStatus = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);

    // Create one OpenCL context for each device in the platform
    cl_context context;
    context = clCreateContext( NULL, num_devices, device_list, NULL, NULL, &clStatus);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);

    // Create memory buffers on the device for each vector
    xpos = clCreateBuffer(context, CL_MEM_READ_ONLY,particle_count * sizeof(float), NULL, &clStatus);
    ypos = clCreateBuffer(context, CL_MEM_READ_ONLY,particle_count * sizeof(float), NULL, &clStatus);
    
    xvel = clCreateBuffer(context, CL_MEM_READ_ONLY,particle_count * sizeof(float), NULL, &clStatus);
    yvel = clCreateBuffer(context, CL_MEM_READ_ONLY,particle_count * sizeof(float), NULL, &clStatus);
    
    mass = clCreateBuffer(context, CL_MEM_READ_ONLY,particle_count * sizeof(float), NULL, &clStatus);
    
    xvelres = clCreateBuffer(context, CL_MEM_WRITE_ONLY,particle_count * sizeof(float), NULL, &clStatus);
    yvelres = clCreateBuffer(context, CL_MEM_WRITE_ONLY,particle_count * sizeof(float), NULL, &clStatus);
    
    xposres = clCreateBuffer(context, CL_MEM_WRITE_ONLY,particle_count * sizeof(float), NULL, &clStatus);
    yposres = clCreateBuffer(context, CL_MEM_WRITE_ONLY,particle_count * sizeof(float), NULL, &clStatus);
    
    // Copy the Buffer A and B to the device
    /*clStatus = clEnqueueWriteBuffer(command_queue, A_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), A, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(command_queue, B_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), B, 0, NULL, NULL);
*/
    // Create a program from the kernel source
    /*
    cl_program program = clCreateProgramWithSource(context, 1,(const char **)&saxpy_kernel, NULL, &clStatus);

    // Build the program
    clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "saxpy_kernel", &clStatus);
     */
    // Set the arguments of the kernel
    /*
    clStatus = clSetKernelArg(kernel, 0, sizeof(float), (void *)&alpha);
    clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&A_clmem);
    clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&B_clmem);
    clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&C_clmem);
    */
}
