//
//  DataPlotter.cpp
//  NBodySim
//
//  Created by RasmusSamsing on 11/03/2020.
//  Copyright © 2020 Rasmus Samsing. All rights reserved.
//

#include "DataPlotter.hpp"
#include <iostream>

DataPlotter::DataPlotter(int width, int height) {
    this->width = width;
    this->height = height;
    
    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, width, height);
}

void DataPlotter::feedNew3DFrame(float *xarr, float *yarr, float *zarr, int size) {
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 0);
    SDL_RenderClear(renderer);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
    
    float eyeZ = -1000;
    float planeZ = -500;
    
    for (int i = 0; i < size; i ++) {
        int xproj = xarr[i] * eyeZ / (eyeZ + (zarr[i] + planeZ));
        int yproj = yarr[i] * eyeZ / (eyeZ + (zarr[i] + planeZ));
        xproj += width / 2;
        yproj += height / 2;
        //SDL_RenderDrawPoint(renderer, xproj, yproj);
        SDL_Rect rect;
        rect.x = xproj;
        rect.y = yproj;
        rect.w = 5;
        rect.h = 5;
        
        SDL_RenderDrawRect(renderer, &rect);
    }
    SDL_RenderPresent(renderer);
}

void DataPlotter::draw3DData (float **xarrs, float **yarrs, float **zarrs, int particles, int timesteps) {
    SDL_Window *window;                      // Declare a pointer

    SDL_Init(SDL_INIT_VIDEO);              // Initialize SDL2

    // Create an application window with the following settings:
    SDL_CreateWindowAndRenderer(width/2, height/2, SDL_WINDOW_ALLOW_HIGHDPI, &window, &renderer);

    // Check that the window was successfully created
    if (window == NULL) {
        // In the case that the window could not be made...
        printf("Could not create window: %s\n", SDL_GetError());
    }

    // The window is open: could enter program loop here (see SDL_PollEvent())s
    
    SDL_Event event;
    bool running = true;
    bool paused = true;
    int step = 0;
    
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 0);
    SDL_RenderClear(renderer);
    feedNew3DFrame(xarrs[0], yarrs[0], zarrs[0], particles);
    int stepsPerFrame = 50;
    while (running) {

        if(step % stepsPerFrame == 0) {
            feedNew3DFrame(xarrs[step], yarrs[step], zarrs[step], particles);
            if (!paused) {
                SDL_Delay(10);
            }
        }
        
        
        if(SDL_PollEvent(&event)) {
            if(event.type == SDL_QUIT) {
                running = false;
            }
            if(event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_ESCAPE) {
                    running = false;
                }
                if (event.key.keysym.sym == SDLK_SPACE) {
                    paused = !paused;
                }
                if (event.key.keysym.sym == SDLK_LEFT) {
                    step -= 3*stepsPerFrame;
                }
                if (event.key.keysym.sym == SDLK_RIGHT) {
                    step += 3*stepsPerFrame;
                }
                if (event.key.keysym.sym == SDLK_r) {
                    step = 0;
                }
            }
        }
        if (!paused) {
            step++;
        }
        if(step >= timesteps) step = timesteps - 1;
        if(step <= 0) step = 0;
    }

    // Close and destroy the window
    SDL_DestroyWindow(window);
    
    // Clean up
    SDL_Quit();
}
