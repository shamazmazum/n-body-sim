#include <iniparser.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "config.h"

static struct config_parameters conf_default = {
    .mass_mean = 2000000,
    .mass_std = 10000,

    .velocity_norm_mean = 80000,
    .velocity_norm_std = 400,

    .position_norm_mean = 0,
    .position_norm_std = 2000,

    .delta = 0.001,
    .nbodies = 15360,
    .snapshot_steps = 1000
};

void parse_config (struct config_parameters *config, const char *name)
{
    memcpy (config, &conf_default, sizeof(struct config_parameters));
    if (name == NULL) return;
    dictionary *dict = iniparser_load (name);
    if (dict == NULL) return;

    config->mass_mean = iniparser_getdouble (dict, "mass:mean", config->mass_mean);
    config->mass_std = iniparser_getdouble (dict, "mass:std", config->mass_std);

    config->velocity_norm_mean = iniparser_getdouble (dict, "velocity:mean", config->velocity_norm_mean);
    config->velocity_norm_std = iniparser_getdouble (dict, "velocity:std", config->velocity_norm_std);

    config->position_norm_mean = iniparser_getdouble (dict, "position:mean", config->position_norm_mean);
    config->position_norm_std = iniparser_getdouble (dict, "position:std", config->position_norm_std);

    config->delta = iniparser_getdouble (dict, "general:delta", config->delta);
    int nbodies = iniparser_getint (dict, "general:nbodies", config->nbodies);
    if (nbodies < 0 || (nbodies & 255) != 0) {
        fprintf (stderr, "nbodies must be positive multiple of 256, setting to default\n");
        nbodies = config->nbodies;
    }
    config->nbodies = nbodies;

    int snapshot_steps = iniparser_getint (dict, "general:snapshot_steps", config->snapshot_steps);
    if (snapshot_steps < 0) {
        fprintf (stderr, "snapshot_steps must be positive\n");
        snapshot_steps = config->snapshot_steps;
    }
    config->snapshot_steps = snapshot_steps;

    iniparser_freedict (dict);
}

void print_config (const struct config_parameters *config)
{
    printf ("mass: mean=%f, std=%f\n", config->mass_mean, config->mass_std);
    printf ("velocity norm: mean=%f, std=%f\n",
            config->velocity_norm_mean,
            config->velocity_norm_std);
    printf ("position norm: mean=%f, std=%f\n",
            config->position_norm_mean,
            config->position_norm_std);
    printf ("delta=%f\n", config->delta);
    printf ("nbodies=%lu\n", config->nbodies);
    printf ("snapshot every %u steps\n", config->snapshot_steps);
    
}
