#ifndef _CL_STATE_H_
#define _CL_STATE_H_
#include <CL/cl.h>

#define MAP_MASS 0
#define MAP_POSITION 1
#define MAP_VELOCITY 2

struct cl_state;

struct cl_state* create_cl_state (const char *solver, cl_float delta);
void destroy_cl_state(struct cl_state *state);

size_t initialize_memory (struct cl_state *state, size_t n);
void* map_gpu_memory (struct cl_state *state, int which, cl_map_flags flags);
void unmap_gpu_memory (struct cl_state *state, int which, void *ptr);
void take_step (struct cl_state *state);
cl_float kinetic_energy (struct cl_state *state);
cl_float potential_energy (struct cl_state *state);
int save_gpu_memory (struct cl_state *state, int which, const char *name);
int restore_gpu_memory (struct cl_state *state, int which, const char *name);

#endif
