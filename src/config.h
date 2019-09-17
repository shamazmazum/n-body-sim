#ifndef _CONFIG_H_
#define _CONFIG_H_
#include <CL/cl.h>

struct config_parameters {
    cl_float mass_mean;
    cl_float mass_std;

    cl_float velocity_norm_mean;
    cl_float velocity_norm_std;

    cl_float position_norm_mean;
    cl_float position_norm_std;

    cl_float delta;

    size_t nbodies;
    unsigned int snapshot_steps;
    unsigned int check_energy;
    const char *solver;
    const char *prefix;
};

void parse_config (struct config_parameters *config, const char *name);
void print_config (const struct config_parameters *config);

#endif
