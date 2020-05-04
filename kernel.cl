void calcDistanceAndComponent (double x1, double y1, double z1, double x2, double y2, double z2, double* dist, double* x_comp, double* y_comp, double* z_comp);
void calcForceVector (double dist, double mass1, double mass2, double x_comp, double y_comp, double z_comp, double* forceX, double* forceY, double* forceZ);

void calcDistanceAndComponent (double x1, double y1, double z1, double x2, double y2, double z2, double* dist, double* x_comp, double* y_comp, double* z_comp) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    double dz = z2 - z1;

    *dist = sqrt(dx*dx + dy*dy + dz*dz);
    *x_comp = dx / *dist;
    *y_comp = dy / *dist;
    *z_comp = dz / *dist;
}

void calcForceVector (double dist, double mass1, double mass2, double x_comp, double y_comp, double z_comp, double* forceX, double* forceY, double* forceZ) {
    double G = 6.67 * 0.00000000001;
    double total_force = G * mass1 * mass2 / (dist) / (dist);
    *forceX += total_force * x_comp;
    *forceY += total_force * y_comp;
    *forceZ += total_force * z_comp;
}

__kernel void forcePredict (__global double* xpos, 
                            __global double* ypos, 
                            __global double* zpos, 
                            __global double* xvel, 
                            __global double* yvel, 
                            __global double* zvel, 
                            __global double* mass,
                            __global int* particle_count,
                            __global double* xposp,
                            __global double* yposp,
                            __global double* zposp,
                            __global double* xvelp,
                            __global double* yvelp,
                            __global double* zvelp,
                            __global double* x0acc,
                            __global double* y0acc,
                            __global double* z0acc,
                            __global double* x0jerk,
                            __global double* y0jerk,
                            __global double* z0jerk) {
    int p = get_global_id(0);
    double pmass = mass[p];
    double dt = 0.5;

    double dist = 0;
    double x_comp = 0;
    double y_comp = 0;
    double z_comp = 0;
    
    double forceX = 0, forceY = 0, forceZ = 0;
    double jerkX = 0, jerkY = 0, jerkZ = 0;
    double G = 6.67 * 0.00000000001;

    for (int i = 0; i < *particle_count; i++) {
        if (i == p) continue;
        calcDistanceAndComponent(xpos[p], ypos[p], zpos[p], xpos[i], ypos[i], zpos[i], &dist, &x_comp, &y_comp, &z_comp);
        double sdist = dist + 10000;
        calcForceVector (sdist, pmass, mass[i], x_comp, y_comp, z_comp, &forceX, &forceY, &forceZ);
        

        double relvel [3] = {xvel[i] - xvel[p], yvel[i] - yvel[p], zvel[i] - zvel[p]};
        double vdotr = relvel[0] * (x_comp * sdist) + relvel[1] * (y_comp * sdist) + relvel[2] * (z_comp * sdist);

        jerkX += G * mass[i] * (relvel[0] / (sdist * sdist * sdist) - 3 * (vdotr) * (x_comp * sdist) / (sdist*sdist*sdist*sdist*sdist));
        jerkY += G * mass[i] * (relvel[1] / (sdist * sdist * sdist) - 3 * (vdotr) * (y_comp * sdist) / (sdist*sdist*sdist*sdist*sdist));
        jerkZ += G * mass[i] * (relvel[2] / (sdist * sdist * sdist) - 3 * (vdotr) * (z_comp * sdist) / (sdist*sdist*sdist*sdist*sdist));
    }

    double accX = forceX / pmass, accY = forceY / pmass, accZ = forceZ / pmass;

    /* ---- Now calculate Taylor expansion for position and velocity from acceleration and jerk ---- */

    double xpos_p = xpos[p] + xvel[p] * dt + 0.5 * accX * dt * dt + 1.0 / 6.0 * jerkX * dt * dt * dt;
    double ypos_p = ypos[p] + yvel[p] * dt + 0.5 * accY * dt * dt + 1.0 / 6.0 * jerkY * dt * dt *dt;
    double zpos_p = zpos[p] + zvel[p] * dt + 0.5 * accZ * dt * dt + 1.0 / 6.0 * jerkZ * dt * dt *dt;

    double xvel_p = xvel[p] + accX * dt + 0.5 * jerkX * dt * dt;
    double yvel_p = yvel[p] + accY * dt + 0.5 * jerkY * dt * dt;
    double zvel_p = zvel[p] + accZ * dt + 0.5 * jerkZ * dt * dt;

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

__kernel void hermiteIntegrator(__global const double* xpos, 
                                __global const double* ypos, 
                                __global const double* zpos, 
                                __global const double* xvel, 
                                __global const double* yvel, 
                                __global const double* zvel, 
                                __global const double* xposp, 
                                __global const double* yposp, 
                                __global const double* zposp, 
                                __global const double* xvelp,
                                __global const double* yvelp,
                                __global const double* zvelp, 
                                __global const double* xacc0,
                                __global const double* yacc0,
                                __global const double* zacc0,
                                __global const double* xjerk0,
                                __global const double* yjerk0,
                                __global const double* zjerk0,
                                __global const double* mass,
                                __global const int* particle_count,
                                __global double* xposres, 
                                __global double* yposres, 
                                __global double* zposres, 
                                __global double* xvelres, 
                                __global double* yvelres, 
                                __global double* zvelres ) {
    int p = get_global_id(0);

    double dt = 0.5;

    double forcepX = 0, forcepY = 0, forcepZ = 0;
    double jerkpX = 0, jerkpY = 0, jerkpZ = 0;
    double dist = 0, pmass = mass[p];

    double x_comp = 0, y_comp = 0, z_comp = 0;
    /* ---- Calculate predicted acceleration and jerk ---- */
    
    for (int i = 0; i < *particle_count; i++) {
        if(i == p) continue;
        calcDistanceAndComponent(xposp[p], yposp[p], zposp[p], xposp[i], yposp[i], zposp[i], &dist, &x_comp, &y_comp, &z_comp);
        double sdist = dist + 10000;
        calcForceVector (sdist, pmass, mass[i], x_comp, y_comp, z_comp, &forcepX, &forcepY, &forcepZ);
        

        double relvel [3] = {xvelp[i] - xvelp[p], yvelp[i] - yvelp[p], zvelp[i] - zvelp[p]};
        double vdotr = relvel[0] * (x_comp * sdist) + relvel[1] * (y_comp * sdist) + relvel[2] * (z_comp * sdist);

        double G = 6.67 * 0.00000000001;
        jerkpX += G * mass[i] * (relvel[0] / (sdist * sdist * sdist) - 3 * (vdotr) * (x_comp * sdist) / (sdist*sdist*sdist*sdist*sdist));
        jerkpY += G * mass[i] * (relvel[1] / (sdist * sdist * sdist) - 3 * (vdotr) * (y_comp * sdist) / (sdist*sdist*sdist*sdist*sdist));
        jerkpZ += G * mass[i] * (relvel[2] / (sdist * sdist * sdist) - 3 * (vdotr) * (z_comp * sdist) / (sdist*sdist*sdist*sdist*sdist));
    }
    double accpX = forcepX / pmass, accpY = forcepY / pmass, accpZ = forcepZ / pmass;

    double xvel_f = xvel[p] + 0.5 * (xacc0[p] + accpX) * dt + 1.0 / 12.0 * (xjerk0[p] - jerkpX) * dt * dt;
    double yvel_f = yvel[p] + 0.5 * (yacc0[p] + accpY) * dt + 1.0 / 12.0 * (yjerk0[p] - jerkpY) * dt * dt;
    double zvel_f = zvel[p] + 0.5 * (zacc0[p] + accpZ) * dt + 1.0 / 12.0 * (zjerk0[p] - jerkpZ) * dt * dt;
    
    double xpos_f = xpos[p] + 0.5 * (xvel[p] + xvel_f) * dt + 1.0 / 12.0 * (xacc0[p] - accpX) * dt * dt;
    double ypos_f = ypos[p] + 0.5 * (yvel[p] + yvel_f) * dt + 1.0 / 12.0 * (yacc0[p] - accpY) * dt * dt;
    double zpos_f = zpos[p] + 0.5 * (zvel[p] + zvel_f) * dt + 1.0 / 12.0 * (zacc0[p] - accpZ) * dt * dt;

    xposres[p] = xpos_f;
    yposres[p] = ypos_f;
    zposres[p] = zpos_f;
    
    xvelres[p] = xvel_f;
    yvelres[p] = yvel_f;
    zvelres[p] = zvel_f;
}