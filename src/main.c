#include <sys/param.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <errno.h>
#include "rng.h"
#include "clstate.h"
#include "config.h"

#define CHECK_MAP(ptr) do {                             \
        if ((ptr) == NULL) {                            \
            fprintf (stderr, "Cannot map memory.\n");   \
            goto done;                                  \
        }} while(0)

static volatile int do_loop = 1;

static void cleanup (int sig)
{
    do_loop = 0;
}

static void save_snapshot (struct cl_state *state, unsigned int postfix)
{
    char filename[MAXPATHLEN];
    sprintf (filename, "%06i", postfix);
    save_gpu_memory (state, MAP_POSITION, filename);
}

static unsigned int find_starting_num (unsigned int step)
{
    unsigned int i = 0;
    char path[MAXPATHLEN];
    struct stat s;
    int res;

    while (1) {
        snprintf (path, MAXPATHLEN, "%06i", i);
        res = stat (path, &s);
        if (res != 0) break;
        i += step;
    }
    if (errno != ENOENT) i = 0;

    return i;
}

static int restore_state (struct cl_state *state)
{
    int res;

    res = restore_gpu_memory (state, MAP_POSITION, "state_position");
    if (!res) return 0;

    res = restore_gpu_memory (state, MAP_VELOCITY, "state_velocity");
    if (!res) return 0;

    res = restore_gpu_memory (state, MAP_MASS, "state_mass");
    if (!res) return 0;

    return 1;
}

static int save_state (struct cl_state *state)
{
    size_t i;
    int res;

    res = save_gpu_memory (state, MAP_MASS, "state_mass");
    if (!res) return 0;

    res = save_gpu_memory (state, MAP_POSITION, "state_position");
    if (!res) return 0;

    res = save_gpu_memory (state, MAP_VELOCITY, "state_velocity");
    if (!res) return 0;

    return 1;
}

int main (int argc, char *argv[])
{
    unsigned int i;
    int res;
    char ch;
    char *config_name = NULL;
    int restore = 0;
    int save = 0;

    while ((ch = getopt (argc, argv, "srf:")) != -1) {
        switch (ch) {
        case 'f':
            config_name = optarg;
            break;
        case 's':
            save = 1;
            break;
        case 'r':
            restore = 1;
            break;
        }
    }

    argc -= optind;
    argv += optind;

    struct config_parameters config;
    parse_config (&config, config_name);
    print_config (&config);
    size_t nbodies = config.nbodies;
    
    struct cl_state *state = create_cl_state (config.solver, config.delta);

    if (state == NULL) {
        fprintf (stderr, "Cannot initialize OpenCL state\n");
        return 1;
    }

    nbodies = initialize_memory (state, nbodies);
    if (!nbodies) {
        fprintf (stderr, "Cannot initialize memory\n");
        goto done;
    }

    if (restore) {
        res = restore_state (state);
        if (!res) {
            fprintf (stderr, "Cannot restore state, exiting\n");
            goto done;
        }

        i = find_starting_num (config.snapshot_steps);
    } else {
        clear_rng_state();
        cl_float *mass = map_gpu_memory (state, MAP_MASS, CL_MAP_WRITE);
        CHECK_MAP (mass);
        for (i=0; i<nbodies; i++) {
            mass[i] = fmaxf (0, normal (config.mass_mean, config.mass_std));
        }
        unmap_gpu_memory (state, MAP_MASS, mass);

        clear_rng_state();
        cl_float2 *velocity = map_gpu_memory (state, MAP_VELOCITY, CL_MAP_WRITE);
        CHECK_MAP (velocity);
        for (i=0; i<nbodies; i++) {
            velocity[i] = normal2 (config.velocity_norm_mean, config.velocity_norm_std);
        }
        unmap_gpu_memory (state, MAP_VELOCITY, velocity);

        clear_rng_state();
        cl_float2 *position = map_gpu_memory (state, MAP_POSITION, CL_MAP_WRITE);
        CHECK_MAP (position);
        for (i=0; i<nbodies; i++)
            position[i] = normal2 (config.position_norm_mean, config.position_norm_std);
        unmap_gpu_memory (state, MAP_POSITION, position);

        i = 0;
    }

    signal (SIGINT, cleanup);
    signal (SIGTERM, cleanup);

    while (do_loop) {
        if (i % 100 == 0) printf ("%i\n", i);
        if (i % config.snapshot_steps == 0) save_snapshot (state, i);
        if (i % config.check_energy == 0) {
            cl_float kin = kinetic_energy (state);
            cl_float pot = potential_energy (state);
            printf ("Kinetic energy=%.5e, potential energy=%.5e, total energy=%.5e\n",
                    kin, pot, kin+pot);
        }
        take_step (state);
        i++;
    }

    if (save) {
        res = save_state (state);
        if (!res) fprintf (stderr, "Cannot save state\n");
    }

done:
    destroy_cl_state (state);
    return 0;
}
