#ifndef _CONFIG_H_
#define _CONFIG_H_

struct config_parameters {
    float mass_mean;
    float mass_std;

    float velocity_norm_mean;
    float velocity_norm_std;

    float position_norm_mean;
    float position_norm_std;

    float delta;

    size_t nbodies;
    unsigned int snapshot_steps;
};

void parse_config (struct config_parameters *config, const char *name);
void print_config (const struct config_parameters *config);

#endif
