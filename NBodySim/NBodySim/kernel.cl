void calcDistanceAndComponent (float x1, float y1, float z1, float x2, float y2, float z2, float* dist, float* x_comp, float* y_comp, float* z_comp);
void calcForceVector (float dist, float mass1, float mass2, float x_comp, float y_comp, float z_comp, float* forceX, float* forceY, float* forceZ);
__kernel void mykernel( __global const float* xpos, 
                        __global const float* ypos, 
                        __global const float* zpos, 
                        __global const float* xvel, 
                        __global const float* yvel, 
                        __global const float* zvel, 
                        __global const float* mass,
                        __global const int* particle_count,
                        __global float* xposres, 
                        __global float* yposres, 
                        __global float* zposres, 
                        __global float* xvelres, 
                        __global float* yvelres, 
                        __global float* zvelres) {
    int p = get_global_id(0);
    float dt = 0.001;

    float pmass = mass[p];
    float px = xpos[p];
    float py = ypos[p];
    float pz = zpos[p];

    float pxvel = xvel[p];
    float pyvel = yvel[p];
    float pzvel = zvel[p];
    
    float forceX = 0, forceY = 0, forceZ = 0;
    float dist = 0, x_comp = 0, y_comp = 0, z_comp = 0;

    float particleR = 3;
    float Cr = 0.95;

    for (int i = 0; i < *particle_count; i++) {
        if (i == p) continue;
        calcDistanceAndComponent(xpos[p], ypos[p], zpos[p], xpos[i], ypos[i], zpos[i], &dist, &x_comp, &y_comp, &z_comp);
        calcForceVector (dist, pmass, mass[i], x_comp, y_comp, z_comp, &forceX, &forceY, &forceZ);   
        /*
        /* collision 
        if (dist > 2 * particleR) {
            calcForceVector (dist, pmass, mass[i], x_comp, y_comp, z_comp, &forceX, &forceY, &forceZ);     
        } else {
            float n_x = (xpos[i] - px) / dist;
            float n_y = (ypos[i] - py) / dist;
            float n_z = (zpos[i] - pz) / dist;

            float v_p1_n = ((pxvel * n_x) + (pyvel * n_y) + (pyvel * n_z)) / sqrt(n_x*n_x + n_y*n_y + n_z*n_z);
            float v_p2_n = ((xvel[i] * n_x) + (yvel[i] * n_y) + (zvel[i] * n_z)) / sqrt(n_x*n_x + n_y*n_y + n_z*n_z);

            

            float normalImpulse = float(pmass * mass[i]) / float(pmass + mass[i]) * (1.0 + Cr) * (v_p2_n - v_p1_n);
            /*printf("normal impulse %f\n", normalImpulse);
            

            printf("x normal %f\n", n_x);

            float G = 6.67 * 0.00000000001;
            float epot = -G * pmass * mass[i] / dist;
            float ekin = 1.0 / 2.0 * pmass * sqrt(pxvel*pxvel + pyvel*pyvel + pzvel*pzvel) * sqrt(pxvel*pxvel + pyvel*pyvel + pzvel*pzvel);

            if (p == 0) {
                printf("vel 1 %f : %f : %f\n", pxvel, pyvel, pzvel);
                printf("epot %f\n", epot / 1000000000000000);
                printf("ekin %f\n", ekin / 1000000000000000);
                printf("energy before %f\n", (epot+ekin) / 1000000000000000);
            }
            pxvel += normalImpulse / pmass * n_x;
            pyvel += normalImpulse / pmass * n_y;
            pzvel += normalImpulse / pmass * n_z;
            
            ekin = 1.0 / 2.0 * pmass * sqrt(pxvel*pxvel + pyvel*pyvel + pzvel*pzvel) * sqrt(pxvel*pxvel + pyvel*pyvel + pzvel*pzvel);
            if(p == 0) {
                printf("vel 2 %f : %f : %f\n", pxvel, pyvel, pzvel);
                printf("energy after %f\n--------\n", (epot+ekin) / 1000000000000000);
            }
            /*printf("new vel %f\n", pxvel);
        }
        */
    }

    float accX = forceX / pmass;
    float accY = forceY / pmass;
    float accZ = forceZ / pmass;

    pxvel = pxvel + dt * accX;
    pyvel = pyvel + dt * accY;
    pzvel = pzvel + dt * accZ;

    xposres[p] = px + pxvel * dt;
    yposres[p] = py + pyvel * dt;
    zposres[p] = pz + pzvel * dt;

    xvelres[p] = pxvel;
    yvelres[p] = pyvel;
    zvelres[p] = pzvel;
}

void calcDistanceAndComponent (float x1, float y1, float z1, float x2, float y2, float z2, float* dist, float* x_comp, float* y_comp, float* z_comp) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;

    *dist = sqrt(dx*dx + dy*dy + dz*dz);
    *x_comp = dx / *dist;
    *y_comp = dy / *dist;
    *z_comp = dz / *dist;
}

void calcForceVector (float dist, float mass1, float mass2, float x_comp, float y_comp, float z_comp, float* forceX, float* forceY, float* forceZ) {
    float G = 6.67 * 0.00000000001;
    float total_force = G * mass1 * mass2 / (dist+1) / (dist+1);
    *forceX += total_force * x_comp;
    *forceY += total_force * y_comp;
    *forceZ += total_force * z_comp;
}

__kernel void forcePredict (__global const float* xpos, 
                            __global const float* ypos, 
                            __global const float* zpos, 
                            __global const float* xvel, 
                            __global const float* yvel, 
                            __global const float* zvel, 
                            __global const float* mass,
                            __global const int* particle_count,
                            __global float* xposp,
                            __global float* yposp,
                            __global float* zposp,
                            __global float* xvelp,
                            __global float* yvelp,
                            __global float* zvelp,
                            __global float* x0acc,
                            __global float* y0acc,
                            __global float* z0acc,
                            __global float* x0jerk,
                            __global float* y0jerk,
                            __global float* z0jerk) {
    int p = get_global_id(0);
    float pmass = mass[p];
    float dt = 0.001;
    
    /* ---- First calculate acceleration and jerk for the particle ---- */

    float dist = 0;
    float x_comp = 0;
    float y_comp = 0;
    float z_comp = 0;

    float forceX = 0, forceY = 0, forceZ = 0;
    float jerkX = 0, jerkY = 0, jerkZ = 0;
    for (int i = 0; i < particle_count; i++) {
        calcDistanceAndComponent(xpos[p], ypos[p], zpos[p], xpos[i], ypos[i], zpos[i], &dist, &x_comp, &y_comp, &z_comp);
        calcForceVector (dist, pmass, mass[i], x_comp, y_comp, z_comp, &forceX, &forceY, &forceZ);
        

        float relvel [3] = {xvel[i] - xvel[p], yvel[i] - yvel[p], zvel[i] - zvel[p]};
        float vdotr = relvel[0] * (x_comp * dist) + relvel[1] * (y_comp * dist) + relvel[2] * (z_comp * dist);

        float sdist = dist + 1;
        float G = 6.67 * 0.00000000001;
        jerkX += G * mass[i] * (relvel[0] / (sdist * sdist * sdist) - 3 * (vdotr) * (x_comp * sdist) / (sdist*sdist*sdist*sdist*sdist));
        jerkY += G * mass[i] * (relvel[1] / (sdist * sdist * sdist) - 3 * (vdotr) * (y_comp * sdist) / (sdist*sdist*sdist*sdist*sdist));
        jerkZ += G * mass[i] * (relvel[2] / (sdist * sdist * sdist) - 3 * (vdotr) * (z_comp * sdist) / (sdist*sdist*sdist*sdist*sdist));
    }

    float accX = forceX / pmass, accY = forceY / pmass, accZ = forceZ / pmass;

    /* ---- Now calculate Taylor expansion for position and velocity from acceleration and jerk ---- */

    float xpos_p = xpos[p] + xvel[p] * dt + 0.5 * accX * dt * dt + 1.0 / 6.0 * jerkX * dt * dt *dt;
    float ypos_p = xpos[p] + xvel[p] * dt + 0.5 * accX * dt * dt + 1.0 / 6.0 * jerkX * dt * dt *dt;
    float zpos_p = xpos[p] + xvel[p] * dt + 0.5 * accX * dt * dt + 1.0 / 6.0 * jerkX * dt * dt *dt;

    float xvel_p = xvel[p] + accX * dt + 0.5 * jerkX * dt * dt;
    float yvel_p = yvel[p] + accY * dt + 0.5 * jerkY * dt * dt;
    float zvel_p = zvel[p] + accZ * dt + 0.5 * jerkZ * dt * dt;



    x0acc[p] = accX;
    y0acc[p] = accY;
    z0acc[p] = accZ;

    x0jerk[p] = jerkX;
    y0jerk[p] = jerkY;
    z0jerk[p] = jerkZ;

    xposp[p] = xpos_p;
    yposp[p] = ypos_p;
    zposp[p] = zpos_p;

    xvelp[p] = xvel_p;
    yvelp[p] = yvel_p;
    zvelp[p] = zvel_p;
}

__kernel void hermiteIntegrator(__global const float* xpos, 
                                __global const float* ypos, 
                                __global const float* zpos, 
                                __global const float* xvel, 
                                __global const float* yvel, 
                                __global const float* zvel, 
                                __global const float* xposp, 
                                __global const float* yposp, 
                                __global const float* zposp, 
                                __global const float* xvelp,
                                __global const float* yvelp,
                                __global const float* zvelp, 
                                __global const float* xacc0,
                                __global const float* yacc0,
                                __global const float* zacc0,
                                __global const float* xjerk0,
                                __global const float* yjerk0,
                                __global const float* zjerk0,
                                __global const float* mass,
                                __global const int* particle_count,
                                __global float* xposres, 
                                __global float* yposres, 
                                __global float* zposres, 
                                __global float* xvelres, 
                                __global float* yvelres, 
                                __global float* zvelres ) {
    int p = get_global_id(0);
    
    float forcepX = 0, forcepY = 0, forcepZ = 0;
    float jerkpX = 0, jerkpY = 0, jerkpZ = 0;
    float dist = 0, pmass = mass[p];

    float x_comp = 0, y_comp = 0, z_comp = 0;
    /* ---- Calculate predicted acceleration and jerk ---- */
    for (int i = 0; i < particle_count; i++) {
        calcDistanceAndComponent(xposp[p], yposp[p], zposp[p], xposp[i], yposp[i], zposp[i], &dist, &x_comp, &y_comp, &z_comp);
        calcForceVector (dist, pmass, mass[i], x_comp, y_comp, z_comp, &forcepX, &forcepY, &forcepZ);
        

        float relvel [3] = {xvelp[i] - xvelp[p], yvelp[i] - yvelp[p], zvelp[i] - zvelp[p]};
        float vdotr = relvel[0] * (x_comp * dist) + relvel[1] * (y_comp * dist) + relvel[2] * (z_comp * dist);

        float sdist = dist + 1;
        float G = 6.67 * 0.00000000001;
        jerkpX += G * mass[i] * (relvel[0] / (sdist * sdist * sdist) - 3 * (vdotr) * (x_comp * sdist) / (sdist*sdist*sdist*sdist*sdist));
        jerkpY += G * mass[i] * (relvel[1] / (sdist * sdist * sdist) - 3 * (vdotr) * (y_comp * sdist) / (sdist*sdist*sdist*sdist*sdist));
        jerkpZ += G * mass[i] * (relvel[2] / (sdist * sdist * sdist) - 3 * (vdotr) * (z_comp * sdist) / (sdist*sdist*sdist*sdist*sdist));
    }
    float accpX = forcepX / pmass, accpY = forcepY / pmass, accpZ = forcepZ / pmass;

    float xvel_f = xvel[p] + 0.5 * (xacc0[p] + accpX) * dt + 1.0 / 12.0 * (xjerk0[p] - jerkpX) * dt * dt;
    float yvel_f = yvel[p] + 0.5 * (yacc0[p] + accpY) * dt + 1.0 / 12.0 * (yjerk0[p] - jerkpY) * dt * dt;
    float zvel_f = zvel[p] + 0.5 * (zacc0[p] + accpZ) * dt + 1.0 / 12.0 * (zjerk0[p] - jerkpZ) * dt * dt;

    float xpos_f = xpos[p] + 1.0 / 2.0 * (xvel[p] + xvel_f) * dt + 1.0 / 12.0 * (xacc0[p] - accpX) * dt * dt;
    float ypos_f = ypos[p] + 1.0 / 2.0 * (yvel[p] + yvel_f) * dt + 1.0 / 12.0 * (yacc0[p] - accpY) * dt * dt;
    float zpos_f = zpos[p] + 1.0 / 2.0 * (zvel[p] + zvel_f) * dt + 1.0 / 12.0 * (zacc0[p] - accpZ) * dt * dt;

    xposres[p] = xpos_f;
    yposres[p] = ypos_f;
    zposres[p] = zpos_f;

    xvelres[p] = xvel_f;
    yvelres[p] = yvel_f;
    zvelres[p] = zvel_f;
}