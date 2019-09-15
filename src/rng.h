#ifndef _NORMAL_RNG_H_
#define _NORMAL_RNG_H_
#include <CL/cl.h>

cl_float normal (cl_float mean, cl_float sigma);
cl_float2 normal2 (cl_float mean, cl_float sigma);
cl_float2 uniform2 (cl_float mean, cl_float len);
void clear_rng_state ();
#endif
