#include <stdlib.h>
#include <math.h>
#include "rng.h"

static int ready = 0;

void clear_rng_state ()
{
    ready = 0;
}

static cl_float normal_01 ()
{
    static cl_float tmp = 0;
    cl_float x,y,rsq,f,val;

    if (ready) {
        ready = 0;
        return tmp;
    }

    do {
        x = 2.0 * rand() / (cl_float)RAND_MAX - 1.0;
        y = 2.0 * rand() / (cl_float)RAND_MAX - 1.0;
        rsq = x * x + y * y;
    } while( rsq >= 1. || rsq == 0. );

    f = sqrtf (-2.0 * logf(rsq) / rsq);
    tmp = x * f;
    val = y * f;
    ready = 1;

    return val;
}

cl_float normal (cl_float mean, cl_float sigma)
{
    return mean + sigma*normal_01();
}

cl_float2 normal2 (cl_float mean, cl_float sigma)
{
    cl_float norm = normal (mean, sigma);
    cl_float phi = 2 * M_PI * (cl_float)rand() / (cl_float)RAND_MAX;

    cl_float2 res;
    res.x = norm * sinf(phi);
    res.y = norm * cosf(phi);
    return res;
}

cl_float2 uniform2 (cl_float mean, cl_float halflen)
{
    cl_float2 res;
    cl_float x = 2 * (cl_float)rand() / (cl_float)RAND_MAX - 1;
    cl_float y = 2 * (cl_float)rand() / (cl_float)RAND_MAX - 1;
    
    res.x = mean + x*halflen;
    res.y = mean + y*halflen;

    return res;
}
