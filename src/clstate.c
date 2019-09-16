#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <program-map.h>

#include <program_loc.h>
#include "clstate.h"

#define KernelHelper(place, name) do {                          \
    place = clCreateKernel (state->program, name, NULL);        \
    if (place == NULL) {                                        \
        fprintf (stderr, "Cannot create kernel: %s\n", name);   \
        goto bad;                                               \
    }                                                           \
    } while (0)

struct cl_state {
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    cl_kernel step;
    cl_kernel kinetic_energy;
    cl_kernel potential_energy;
    cl_kernel reduce1;
    cl_kernel reduce2;

    size_t nbodies;
    cl_mem mass;
    cl_mem pos;
    cl_mem velocity;
    cl_mem energy;
    cl_float delta;

    size_t group_size;
};

void destroy_cl_state(struct cl_state *state)
{
    if (state == NULL) return;
    if (state->energy != NULL) clReleaseMemObject (state->energy);
    if (state->mass != NULL) clReleaseMemObject (state->mass);
    if (state->pos != NULL) clReleaseMemObject (state->pos);
    if (state->velocity != NULL) clReleaseMemObject (state->velocity);
    if (state->step != NULL) clReleaseKernel (state->step);
    if (state->kinetic_energy != NULL) clReleaseKernel (state->kinetic_energy);
    if (state->potential_energy != NULL) clReleaseKernel (state->potential_energy);
    if (state->reduce1 != NULL) clReleaseKernel (state->reduce1);
    if (state->reduce2 != NULL) clReleaseKernel (state->reduce2);
    if (state->program != NULL) clReleaseProgram (state->program);
    if (state->queue != NULL) clReleaseCommandQueue (state->queue);
    if (state->context != NULL) clReleaseContext (state->context);

    free (state);
}

struct cl_state* create_cl_state (const char *solver, cl_float delta)
{
    cl_context_properties properties[3];
    cl_uint num_of_platforms=0;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_uint num_of_devices=0;
    struct cl_state *state = NULL;
    size_t group_size;
    const char *solver_kernel;

    if (strcmp (solver, "rk2") == 0) solver_kernel = "take_step_rk2";
    else if (strcmp (solver, "euler") == 0) solver_kernel = "take_step_euler";
    else {
        fprintf (stderr, "No such solver: %s\n", solver);
        goto bad;
    }

    // retreives a list of platforms available
    if (clGetPlatformIDs (1, &platform_id, &num_of_platforms)!= CL_SUCCESS) {
        fprintf(stderr, "Unable to get platform_id\n");
        goto bad;
    }

    // try to get a supported GPU device
    if (clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
                        &num_of_devices) != CL_SUCCESS) {
        fprintf(stderr, "Unable to get device_id\n");
        goto bad;
    }

    // Get optimal group size
    if (clGetDeviceInfo (device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof (size_t),
                         &group_size, NULL) != CL_SUCCESS) {
        fprintf (stderr, "Cannot get the optimal group size\n");
        goto bad;
    }

    // context properties list - must be terminated with 0
    properties[0]= CL_CONTEXT_PLATFORM;
    properties[1]= (cl_context_properties) platform_id;
    properties[2]= 0;

    state = malloc (sizeof(*state));
    memset (state, 0, sizeof(*state));
    state->group_size = group_size;
    state->delta = delta;

    state->context = clCreateContext (properties, 1, &device_id, NULL, NULL, NULL);
    if (state->context == NULL) {
        fprintf (stderr, "Cannot create context\n");
        goto bad;
    }

    state->queue = clCreateCommandQueue (state->context, device_id, 0, NULL);
    if (state->queue == NULL) {
        fprintf (stderr, "Cannot create command queue\n");
        goto bad;
    }

    struct pm_program_handler ph;
    if (!pm_map_program (&ph, PROCESS_PATH)) {
        perror (pm_get_error ());
        fprintf (stderr, "Cannot load GPU program\n");
        goto bad;
    }

    state->program = clCreateProgramWithSource (state->context, 1, (const char **)
                                                &(ph.ph_space), NULL, NULL);
    pm_unmap_program (&ph);
    if (state->program == NULL) {
        fprintf (stderr, "Cannot create program\n");
        goto bad;
    }

    if (clBuildProgram (state->program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
        fprintf(stderr, "Error building program\n");
        char buffer[4096];
        size_t length;
        clGetProgramBuildInfo(state->program, device_id, CL_PROGRAM_BUILD_LOG,
                              sizeof(buffer), buffer, &length);
        fprintf(stderr, "%s\n",buffer);
        goto bad;
    }

    KernelHelper (state->step, solver_kernel);
    KernelHelper (state->kinetic_energy, "kinetic_energy");
    KernelHelper (state->potential_energy, "potential_energy");
    KernelHelper (state->reduce1, "reduce");
    KernelHelper (state->reduce2, "reduce");

    return state;

bad:
    destroy_cl_state (state);
    return NULL;
}

size_t initialize_memory (struct cl_state *state, size_t n)
{
    size_t rem = n % state->group_size;
    n -= rem;
    state->nbodies = n;

    cl_ulong red2_size = state->group_size;
    cl_ulong red1_size = n;

    state->mass = clCreateBuffer (state->context, CL_MEM_READ_ONLY, n*sizeof(cl_float), NULL, NULL);
    if (state->mass == NULL) return 0;

    state->pos = clCreateBuffer (state->context, CL_MEM_READ_WRITE, n*sizeof(cl_float2), NULL, NULL);
    if (state->pos == NULL) return 0;

    state->velocity = clCreateBuffer (state->context, CL_MEM_READ_WRITE, n*sizeof(cl_float2), NULL, NULL);
    if (state->velocity == NULL) return 0;

    state->energy = clCreateBuffer (state->context, CL_MEM_READ_WRITE, n*sizeof(cl_float), NULL, NULL);
    if (state->energy == NULL) return 0;

    clSetKernelArg (state->step, 0, sizeof(cl_mem), &state->mass);
    clSetKernelArg (state->step, 1, sizeof(cl_mem), &state->pos);
    clSetKernelArg (state->step, 2, sizeof(cl_mem), &state->velocity);
    clSetKernelArg (state->step, 3, sizeof(cl_float), &state->delta);
    clSetKernelArg (state->step, 4, sizeof(cl_float) * state->group_size, NULL);
    clSetKernelArg (state->step, 5, sizeof(cl_float2) * state->group_size, NULL);

    clSetKernelArg (state->kinetic_energy, 0, sizeof(cl_mem), &state->mass);
    clSetKernelArg (state->kinetic_energy, 1, sizeof(cl_mem), &state->velocity);
    clSetKernelArg (state->kinetic_energy, 2, sizeof(cl_mem), &state->energy);

    clSetKernelArg (state->potential_energy, 0, sizeof(cl_mem), &state->mass);
    clSetKernelArg (state->potential_energy, 1, sizeof(cl_mem), &state->pos);
    clSetKernelArg (state->potential_energy, 2, sizeof(cl_float) * state->group_size, NULL);
    clSetKernelArg (state->potential_energy, 3, sizeof(cl_float2) * state->group_size, NULL);
    clSetKernelArg (state->potential_energy, 4, sizeof(cl_mem), &state->energy);

    clSetKernelArg (state->reduce1, 0, sizeof(cl_mem), &state->energy);
    clSetKernelArg (state->reduce1, 1, sizeof(float) * state->group_size, NULL);
    clSetKernelArg (state->reduce1, 2, sizeof(cl_ulong), &red1_size);

    clSetKernelArg (state->reduce2, 0, sizeof(cl_mem), &state->energy);
    clSetKernelArg (state->reduce2, 1, sizeof(float) * state->group_size, NULL);
    clSetKernelArg (state->reduce2, 2, sizeof(cl_ulong), &red2_size);

    return n;
}

void* map_gpu_memory (struct cl_state *state, int which, cl_map_flags flags)
{
    cl_mem buffer;
    size_t size;

    switch (which) {
    case MAP_MASS:
        buffer = state->mass;
        size = sizeof(cl_float) * state->nbodies;
        break;
    case MAP_POSITION:
        buffer = state->pos;
        size = sizeof(cl_float2) * state->nbodies;
        break;
    case MAP_VELOCITY:
        buffer = state->velocity;
        size = sizeof(cl_float2) * state->nbodies;
        break;
    }
    void *res = clEnqueueMapBuffer (state->queue, buffer, CL_TRUE, flags, 0, size, 0, NULL, NULL, NULL);
    return res;
}

void unmap_gpu_memory (struct cl_state *state, int which, void *ptr)
{
    cl_mem buffer;

    switch (which) {
    case MAP_MASS:
        buffer = state->mass;
        break;
    case MAP_POSITION:
        buffer = state->pos;
        break;
    case MAP_VELOCITY:
        buffer = state->velocity;
        break;
    }

    clEnqueueUnmapMemObject (state->queue, buffer, ptr, 0, NULL, NULL);
    clFinish (state->queue);
}

void take_step (struct cl_state *state)
{
    size_t n = state->nbodies;
    size_t nloc = state->group_size;
    clEnqueueNDRangeKernel (state->queue, state->step, 1, NULL, &n, &nloc, 0, NULL, NULL);
    clFinish (state->queue);
}

int save_gpu_memory (struct cl_state *state, int which, const char *name)
{
    FILE *handle;
    cl_float2 *vector;
    cl_float *scalar;
    void *map;
    size_t i;
    int scalar_map = (which == MAP_MASS);
    int res = 0;

    handle = fopen (name, "w");
    if (handle == NULL) goto done;

    map = map_gpu_memory (state, which, CL_MAP_READ);
    if (map == NULL) goto done;
    scalar = map;
    vector = map;

    for (i=0; i<state->nbodies; i++) {
        if (scalar_map)
            fprintf (handle, "%.10f\n", scalar[i]);
        else
            fprintf (handle, "%.10f %.10f\n", vector[i].x, vector[i].y);
    }
    unmap_gpu_memory (state, which, map);
    res = 1;

done:
    if (handle != NULL) fclose (handle);
    return res;
}

int restore_gpu_memory (struct cl_state *state, int which, const char *name)
{
    FILE *handle;
    cl_float2 *vector;
    cl_float *scalar;
    void *map = NULL;
    size_t i;
    int scalar_map = (which == MAP_MASS);
    int res = 0;

    handle = fopen (name, "r");
    if (handle == NULL) goto done;

    map = map_gpu_memory (state, which, CL_MAP_WRITE);
    if (map == NULL) goto done;
    scalar = map;
    vector = map;
    for (i=0; i<state->nbodies; i++) {
        if (scalar_map) {
            res = fscanf (handle, "%f\n", &scalar[i]);
            if (res != 1) {
                res = 0;
                goto done;
            }
        } else {
            res = fscanf (handle, "%f %f\n", &vector[i].x, &vector[i].y);
            if (res != 2) {
                res = 0;
                goto done;
            }
        }
    }
    res = 1;

done:
    if (handle != NULL) fclose (handle);
    if (map != NULL) unmap_gpu_memory (state, which, map);
    return res;
}

cl_float kinetic_energy (struct cl_state *state)
{
    size_t n = state->nbodies;
    size_t nloc = state->group_size;
    size_t reduce1n = nloc * nloc;
    cl_float res;

    clEnqueueNDRangeKernel (state->queue, state->kinetic_energy, 1, NULL, &n, NULL, 0, NULL, NULL);
    clFinish (state->queue);
    clEnqueueNDRangeKernel (state->queue, state->reduce1, 1, NULL, &reduce1n, &nloc, 0, NULL, NULL);
    clFinish (state->queue);
    clEnqueueNDRangeKernel (state->queue, state->reduce2, 1, NULL, &nloc, &nloc, 0, NULL, NULL);
    clFinish (state->queue);
    clEnqueueReadBuffer (state->queue, state->energy, CL_TRUE, 0, sizeof (cl_float), &res, 0, NULL, NULL);

    return res;
}

cl_float potential_energy (struct cl_state *state)
{
    size_t n = state->nbodies;
    size_t nloc = state->group_size;
    size_t reduce1n = nloc * nloc;
    cl_float res;

    clEnqueueNDRangeKernel (state->queue, state->potential_energy, 1, NULL, &n, &nloc, 0, NULL, NULL);
    clFinish (state->queue);
    clEnqueueNDRangeKernel (state->queue, state->reduce1, 1, NULL, &reduce1n, &nloc, 0, NULL, NULL);
    clFinish (state->queue);
    clEnqueueNDRangeKernel (state->queue, state->reduce2, 1, NULL, &nloc, &nloc, 0, NULL, NULL);
    clFinish (state->queue);
    clEnqueueReadBuffer (state->queue, state->energy, CL_TRUE, 0, sizeof (cl_float), &res, 0, NULL, NULL);

    return res;
}
