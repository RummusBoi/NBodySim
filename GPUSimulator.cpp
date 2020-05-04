//
//  GPUSimulator.cpp
//  NBodySim
//
//  Created by RasmusSamsing on 28/02/2020.
//  Copyright © 2020 Rasmus Samsing. All rights reserved.
//

#include "GPUSimulator.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include<random>
#include <cstring>
#include <zconf.h>
//#include <OpenCL/OpenCL.h>

using namespace std;

const char *getErrorString(cl_int error);
void printErrorCode (cl_int* code);
void printArray(double* arr, int len);
std::string get_self_path ();

GPUSimulator::GPUSimulator (int time_steps, int particle_count) {
    this->time_steps = time_steps;
    this->particle_count = particle_count;
    xposarr = new double[particle_count];
    yposarr = new double[particle_count];
    zposarr = new double[particle_count];
    
    xvelarr = new double[particle_count];
    yvelarr = new double[particle_count];
    zvelarr = new double[particle_count];
    
    massarr = new double[particle_count];
    
    prevxs = new double*[time_steps];
    prevys = new double*[time_steps];
    prevzs = new double*[time_steps];
    
    iteration = 0;
    
    for (int i = 0; i < time_steps; i++) {
        prevxs[i] = new double[particle_count];
        prevys[i] = new double[particle_count];
        prevzs[i] = new double[particle_count];
    }
    // insert code here...
    // Get platform and device information
    cl_int clStatus;
    // Get platform and device information
    
    /*Step1: Getting platforms and choose an available one.*/
    cl_uint numPlatforms; //the NO. of platforms
    cl_platform_id platform = NULL; //the chosen platform
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS)
    {
        cout << "Error: Getting platforms!" << endl;
        exit(1);
    }

    /*For clarity, choose the first available platform. */
    if (numPlatforms > 0)
    {
        cl_platform_id* platforms =
                     (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        platform = platforms[0];
        free(platforms);
    }

    /*Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.*/
    cl_uint numDevices = 0;
    cl_device_id        *devices;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if (numDevices == 0) //no GPU available.
    {
        cout << "No GPU device available." << endl;
        cout << "Choose CPU as default device." << endl;
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
        devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
    }
    else
    {
        devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    }


    /*Step 3: Create context.*/
    context = clCreateContext(NULL, 1, devices, NULL, NULL, &clStatus);
    printErrorCode(&clStatus);
    
    /*Step 4: Creating command queue associate with the context.*/
    command_queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &clStatus);
    printErrorCode(&clStatus);
    
    // Create memory buffers on the device for each vector
    xpos = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    ypos = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    zpos = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    
    xvel = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    yvel = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    zvel = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    
    mass = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    n_particles = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(int), NULL, &clStatus);
    printErrorCode(&clStatus);
    
    xvelres = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    yvelres = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    zvelres = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    
    xposres = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    yposres = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    zposres = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    
    xaccPredicted = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    yaccPredicted = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    zaccPredicted = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    
    xjerkPredicted = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    yjerkPredicted = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    zjerkPredicted = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    
    x0acc = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    y0acc = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    z0acc = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    
    x0jerk = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    y0jerk = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    z0jerk = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    
    xposp = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    yposp = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    zposp = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    
    xvelp = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    yvelp = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    zvelp = clCreateBuffer(context, CL_MEM_READ_WRITE,particle_count * sizeof(double), NULL, &clStatus);
    printErrorCode(&clStatus);
    
    // -- Create and build the program -- //
    
    cl_program program;
    
    ifstream kernelfile;
    string path = get_self_path();
    std::string::size_type t = path.find_last_of("/");
    path = path.substr(0,t);
    path = path + "/kernel.cl";

    kernelfile.open(path);
    
    if (!kernelfile) {
        cerr << "Unable to open kernel file";
        exit(1); // call system to stop
    }
    string line;
    string srcString = "";
    
    if (kernelfile.is_open())
    {
      while (getline (kernelfile, line) )
      {
          srcString = srcString + line;
      }
      kernelfile.close();
    }
    
    const char* srcCharArr = srcString.c_str();
    
    program = clCreateProgramWithSource(context, 1, &srcCharArr, NULL, &clStatus);
    printErrorCode(&clStatus);
    cout << "building program..." << endl;
    clStatus = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
    printErrorCode(&clStatus);
    cout << "built program..." << endl;
    kernel = clCreateKernel(program, "hermiteIntegrator", &clStatus);
    printErrorCode(&clStatus);
    
    accPredictor = clCreateKernel(program, "forcePredict", &clStatus);
    printErrorCode(&clStatus);
    
    
    // ----- Acc and Jerk predictor kernel ---- //
    
    int a = 0;
    clStatus = clSetKernelArg(accPredictor, a, sizeof(cl_mem), (void *)&xpos);
    printErrorCode(&clStatus);
    clStatus = clSetKernelArg(accPredictor, ++a, sizeof(cl_mem), (void *)&ypos);
    printErrorCode(&clStatus);
    clStatus = clSetKernelArg(accPredictor, ++a, sizeof(cl_mem), (void *)&zpos);
    printErrorCode(&clStatus);
    
    clStatus = clSetKernelArg(accPredictor, ++a, sizeof(cl_mem), (void *)&xvel);
    printErrorCode(&clStatus);
    clStatus = clSetKernelArg(accPredictor, ++a, sizeof(cl_mem), (void *)&yvel);
    printErrorCode(&clStatus);
    clStatus = clSetKernelArg(accPredictor, ++a, sizeof(cl_mem), (void *)&zvel);
    printErrorCode(&clStatus);
    
    clStatus = clSetKernelArg(accPredictor, ++a, sizeof(cl_mem), (void *)&mass);
    clStatus = clSetKernelArg(accPredictor, ++a, sizeof(cl_mem), (void *)&n_particles);
    
    clStatus = clSetKernelArg(accPredictor, ++a, sizeof(cl_mem), (void *)&xposp);
    printErrorCode(&clStatus);
    clStatus = clSetKernelArg(accPredictor, ++a, sizeof(cl_mem), (void *)&yposp);
    printErrorCode(&clStatus);
    clStatus = clSetKernelArg(accPredictor, ++a, sizeof(cl_mem), (void *)&zposp);
    printErrorCode(&clStatus);
    
    clStatus = clSetKernelArg(accPredictor, ++a, sizeof(cl_mem), (void *)&xvelp);
    printErrorCode(&clStatus);
    clStatus = clSetKernelArg(accPredictor, ++a, sizeof(cl_mem), (void *)&yvelp);
    printErrorCode(&clStatus);
    clStatus = clSetKernelArg(accPredictor, ++a, sizeof(cl_mem), (void *)&zvelp);
    printErrorCode(&clStatus);
    
    clStatus = clSetKernelArg(accPredictor, ++a, sizeof(cl_mem), (void *)&x0acc);
    clStatus = clSetKernelArg(accPredictor, ++a, sizeof(cl_mem), (void *)&y0acc);
    clStatus = clSetKernelArg(accPredictor, ++a, sizeof(cl_mem), (void *)&z0acc);
    
    clStatus = clSetKernelArg(accPredictor, ++a, sizeof(cl_mem), (void *)&x0jerk);
    clStatus = clSetKernelArg(accPredictor, ++a, sizeof(cl_mem), (void *)&y0jerk);
    clStatus = clSetKernelArg(accPredictor, ++a, sizeof(cl_mem), (void *)&z0jerk);
    printErrorCode(&clStatus);
    
    
    // ---- Hermite integrator kernel ---- //
    cout << "setting args " << endl;
    a = 0;
    clStatus = clSetKernelArg(kernel, a, sizeof(cl_mem), (void *)&xpos);
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&ypos);
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&zpos);
    
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&xvel);
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&yvel);
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&zvel);
    
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&xposp);
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&yposp);
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&zposp);
    
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&xvelp);
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&yvelp);
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&zvelp);
    
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&x0acc);
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&y0acc);
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&z0acc);
    
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&x0jerk);
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&y0jerk);
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&z0jerk);
    
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&mass);
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&n_particles);
    
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&xposres);
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&yposres);
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&zposres);
    
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&xvelres);
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&yvelres);
    clStatus = clSetKernelArg(kernel, ++a, sizeof(cl_mem), (void *)&zvelres);
}

std::string get_self_path () {
    char buf[1024];

    /* The manpage says it won't null terminate.  Let's zero the buffer. */
    memset(buf, 0, sizeof(buf));

    /* Note we use sizeof(buf)-1 since we may need an extra char for NUL. */
    if (readlink("/proc/self/exe", buf, sizeof(buf)-1) < 0)
    {
        /* There was an error...  Perhaps the path does not exist
         * or the buffer is not big enough.  errno has the details. */
        perror("readlink");
        exit(-2);
    }
    return buf;
}

void GPUSimulator::setState (double* xp,
               double* yp,
               double* zp,
               double* xv,
               double* yv,
               double* zv,
               double* m) {
    for (int i = 0; i < particle_count; i++) {
        xposarr[i] = xp[i];
        yposarr[i] = yp[i];
        zposarr[i] = zp[i];
        
        xvelarr[i] = xv[i];
        yvelarr[i] = yv[i];
        zvelarr[i] = zv[i];
        
        massarr[i] = m[i];
    }
    
    cl_int clStatus;
    clStatus = clEnqueueWriteBuffer(command_queue, xpos, CL_TRUE, 0, particle_count * sizeof(double), xposarr, 0, NULL, NULL);
    printErrorCode(&clStatus);
    clStatus = clEnqueueWriteBuffer(command_queue, ypos, CL_TRUE, 0, particle_count * sizeof(double), yposarr, 0, NULL, NULL);
    printErrorCode(&clStatus);
    clStatus = clEnqueueWriteBuffer(command_queue, zpos, CL_TRUE, 0, particle_count * sizeof(double), zposarr, 0, NULL, NULL);
    printErrorCode(&clStatus);
    clStatus = clEnqueueWriteBuffer(command_queue, xvel, CL_TRUE, 0, particle_count * sizeof(double), xvelarr, 0, NULL, NULL);
    printErrorCode(&clStatus);
    clStatus = clEnqueueWriteBuffer(command_queue, yvel, CL_TRUE, 0, particle_count * sizeof(double), yvelarr, 0, NULL, NULL);
    printErrorCode(&clStatus);
    clStatus = clEnqueueWriteBuffer(command_queue, zvel, CL_TRUE, 0, particle_count * sizeof(double), zvelarr, 0, NULL, NULL);
    printErrorCode(&clStatus);
    clStatus = clEnqueueWriteBuffer(command_queue, mass, CL_TRUE, 0, particle_count * sizeof(double), massarr, 0, NULL, NULL);
    printErrorCode(&clStatus);
    clStatus = clEnqueueWriteBuffer(command_queue, n_particles, CL_TRUE, 0, sizeof(int), &particle_count, 0, NULL, NULL);
    printErrorCode(&clStatus);
    clStatus = clEnqueueWriteBuffer(command_queue, xposres, CL_TRUE, 0, particle_count * sizeof(double), yposarr, 0, NULL, NULL);
    clFinish(command_queue);
}

void GPUSimulator::setInitialState() {
	double* xp = new double[particle_count];
    double* yp = new double[particle_count];
	double* zp = new double[particle_count];
	double* xv = new double[particle_count];
	double* yv = new double[particle_count];
	double* zv = new double[particle_count];
	double* m = new double[particle_count];

	int limit = 100000000;
	int vellimit = 50000;

	std::random_device rseed;
	std::random_device rseed2;
	
	std::mt19937 rpos(rseed());
	std::uniform_int_distribution<int> posdist(-limit, limit);

	std::mt19937 rvel(rseed2());
	std::uniform_int_distribution<int> veldist(-vellimit, vellimit);


    
    for (int i = 0; i < particle_count; i++) {
		xp[i] = posdist(rpos);
        yp[i] = posdist(rpos);
        zp[i] = posdist(rpos);
        xv[i] = veldist(rvel);
		yv[i] = veldist(rvel);
		zv[i] = veldist(rvel);
        m[i] = (double)pow(10, 23);
    }
    
    m[0] *= 30;
    m[1] *= 1;
    
    xp[0] = 0;
    yp[0] = 0;
    zp[0] = 0;
    
    xp[1] = 40000000;
    yp[1] = 0;
    zp[1] = 0;
    
    xp[2] = 2000000;
    yp[2] = 0;
    yp[2] = 0;
    
    yv[0] = -60;
    yv[1] = 2500;
    yv[2] = 15000;
    
    xv[0] = 0;
    xv[1] = 0;
    
    this->setState(xp, yp, zp, xv, yv, zv, m);
}

void GPUSimulator::runGen() {
    
    // ---- Write to buffers for accPredictor ---- //
    
    // -- accPredictor writes to predicted pos, vel and current acc and current jerk -- //
    size_t global_item_size = particle_count;
    size_t local_item_size = 256;
    cl_int clStatus;

    clStatus = clEnqueueNDRangeKernel(command_queue, accPredictor, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
	printErrorCode(&clStatus);

    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
	printErrorCode(&clStatus);

    
    // -- Update kernel args -- //
    int a = 0;
    clSetKernelArg(accPredictor, a++, sizeof(cl_mem), (void *)&xposres);
    clSetKernelArg(accPredictor, a++, sizeof(cl_mem), (void *)&yposres);
    clSetKernelArg(accPredictor, a++, sizeof(cl_mem), (void *)&zposres);
    
    clSetKernelArg(accPredictor, a++, sizeof(cl_mem), (void *)&xvelres);
    clSetKernelArg(accPredictor, a++, sizeof(cl_mem), (void *)&yvelres);
    clSetKernelArg(accPredictor, a++, sizeof(cl_mem), (void *)&zvelres);
    a = 0;
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&xposres);
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&yposres);
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&zposres);
    
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&xvelres);
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&yvelres);
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&zvelres);
    a+=14;
    
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&xpos);
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&ypos);
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&zpos);
    
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&xvel);
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&yvel);
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&zvel);
    
    clFinish(command_queue);
    
    // --- Run the second gen with swapped buffers --- //
    clStatus = clEnqueueNDRangeKernel(command_queue, accPredictor, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    
    // -- Update kernel args -- //
    a = 0;
    clSetKernelArg(accPredictor, a++, sizeof(cl_mem), (void *)&xpos);
    clSetKernelArg(accPredictor, a++, sizeof(cl_mem), (void *)&ypos);
    clSetKernelArg(accPredictor, a++, sizeof(cl_mem), (void *)&zpos);
    
    clSetKernelArg(accPredictor, a++, sizeof(cl_mem), (void *)&xvel);
    clSetKernelArg(accPredictor, a++, sizeof(cl_mem), (void *)&yvel);
    clSetKernelArg(accPredictor, a++, sizeof(cl_mem), (void *)&zvel);
    a = 0;
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&xpos);
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&ypos);
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&zpos);
    
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&xvel);
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&yvel);
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&zvel);
    a+=14;
    
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&xposres);
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&yposres);
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&zposres);
    
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&xvelres);
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&yvelres);
    clSetKernelArg(kernel, a++, sizeof(cl_mem), (void *)&zvelres);
}

void GPUSimulator::appendState(int timestep) {
    cl_int clStatus;
    clStatus = clEnqueueReadBuffer(command_queue, xpos, CL_TRUE, 0, particle_count * sizeof(double), xposarr, NULL, NULL, NULL);
    printErrorCode(&clStatus);
    clStatus = clEnqueueReadBuffer(command_queue, ypos, CL_TRUE, 0, particle_count * sizeof(double), yposarr, NULL, NULL, NULL);
    printErrorCode(&clStatus);
    clStatus = clEnqueueReadBuffer(command_queue, zpos, CL_TRUE, 0, particle_count * sizeof(double), zposarr, NULL, NULL, NULL);
    printErrorCode(&clStatus);
    
    clStatus = clEnqueueReadBuffer(command_queue, xvel, CL_TRUE, 0, particle_count * sizeof(double), xvelarr, NULL, NULL, NULL);
    printErrorCode(&clStatus);
    clStatus = clEnqueueReadBuffer(command_queue, yvel, CL_TRUE, 0, particle_count * sizeof(double), yvelarr, NULL, NULL, NULL);
    printErrorCode(&clStatus);
    clStatus = clEnqueueReadBuffer(command_queue, zvel, CL_TRUE, 0, particle_count * sizeof(double), zvelarr, NULL, NULL, NULL);
    printErrorCode(&clStatus);
    clFinish(command_queue);

	memcpy(prevxs[timestep], xposarr, sizeof(double) * particle_count);
	memcpy(prevys[timestep], yposarr, sizeof(double) * particle_count);
	memcpy(prevzs[timestep], zposarr, sizeof(double) * particle_count);
}

float GPUSimulator::calcTotalEnergy() {
    double ekin = 0;
    for (int i = 0; i < particle_count; i++) {
        double p_ekin = 0.5 * massarr[i] * (xvelarr[i]*xvelarr[i] + yvelarr[i]*yvelarr[i] + zvelarr[i]*zvelarr[i]);
        ekin += p_ekin;
    }
    
    double G = 6.67 * 0.00000000001;
    double epot = 0;
    for (int p1 = 0; p1 < particle_count; p1++) {
        for (int p2 = p1 + 1; p2 < particle_count; p2++) {
            double dist = sqrt((xposarr[p2] - xposarr[p1])*(xposarr[p2] - xposarr[p1]) + (yposarr[p2] - yposarr[p1])*(yposarr[p2] - yposarr[p1]) + (zposarr[p1] - zposarr[p2])*(zposarr[p1] - zposarr[p2]));
            double p1_pot = -G * massarr[p1] * massarr[p2] / dist;
            epot += p1_pot;
        }
    }
    
    return (float)(ekin + epot);
}



void GPUSimulator::saveStateToFile () {
    ofstream myfile;
    myfile.open ("data.txt");
    string outputstring = "";
}

void GPUSimulator::getStoredData(double ***xarr, double ***yarr, double ***zarr) {
    *xarr = prevxs;
    *yarr = prevys;
    *zarr = prevzs;
}

void GPUSimulator::printState() {
    for (int i = 0; i < time_steps; i++) {
        cout << prevxs[i][0] << " : " << prevxs[i][1] << endl;
    }
    for (int t = 0; t < time_steps; t++) {
        for (int p = 0; p < particle_count; p++) {
            cout << (prevxs[t])[p] << " : " << (prevys[t])[p] << " : " << (prevzs[t])[p] << endl;
        }
        cout << " ----- " << endl;
    }
}

void printArray(double* arr, int len) {
    string str = "";
    for (int i = 0; i < len; i++) {
        str += arr[i];
    }
    cout << str << endl;
}

void printErrorCode (cl_int* code) {
    if(*code != CL_SUCCESS) {
        cout << getErrorString(*code) << endl;
    }
}
const char *getErrorString(cl_int error)
{
switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}
