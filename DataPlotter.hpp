//
//  DataPlotter.hpp
//  NBodySim
//
//  Created by RasmusSamsing on 11/03/2020.
//  Copyright © 2020 Rasmus Samsing. All rights reserved.
//

#ifndef DataPlotter_hpp
#define DataPlotter_hpp

#include <stdio.h>
#include <SDL2/SDL.h>
#undef main

class DataPlotter{
private:
    int width, height;
    SDL_Texture* texture;
    SDL_Renderer* renderer;
public:
    DataPlotter(int width, int height);
    void feedNew3DFrame (double* xarr, double* yarr, double* zarr, int size);
    void updateFrame();
    void draw3DData(double **xarrs, double **yarrs, double **zarrs, int particles, int timesteps);
    
};

#endif /* DataPlotter_hpp */
