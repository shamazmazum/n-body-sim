#include <sys/param.h>
#include <sys/stat.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <errno.h>
#include <assert.h>
#include "clstate.h"

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

static struct config_parameters {
    size_t nbodies;
    unsigned int output_steps;
    unsigned int invariant_steps;
    cl_float delta;
    int no_update;

    const char *solver;
    const char *out_prefix;
    const char *energy_file;

    const char *state_position;
    const char *state_velocity;
    const char *state_mass;
} config = {
    .nbodies         = 0,
    .output_steps    = 100,
    .invariant_steps = 5000,
    .delta           = 0.00001,
    .no_update       = 0,

    .solver      = "rk2",
    .out_prefix  = "out",
    .energy_file = NULL
};

enum {
    SET_OUTPUT_PREFIX = 1000,
    SET_NO_UPDATE,
    SET_ENERGY_OUT
};
const struct option long_opts[] = {
    { "output-steps",    required_argument, NULL,               'o' },
    { "invariant-steps", required_argument, NULL,               'i' },
    { "output-prefix",   required_argument, NULL, SET_OUTPUT_PREFIX },
    { "no-update",       no_argument,       NULL, SET_NO_UPDATE     },
    { "output-energy",   required_argument, NULL, SET_ENERGY_OUT    },
    {NULL,                               0, NULL,                0  }
};

static void print_config (const struct config_parameters *config)
{
    printf ("Config parameters:\n");
    printf ("nbodies=%lu\n", config->nbodies);
    printf ("output steps=%u, invariant_steps=%u\n",
            config->output_steps,
            config->invariant_steps);
    printf ("delta=%f\n", config->delta);
    printf ("solver=%s\n", config->solver);

    if (config->energy_file) {
        printf ("Output energy to %s\n", config->energy_file);
    }
}

static void save_position (struct cl_state *state, const char *prefix, unsigned int n)
{
    char filename[MAXPATHLEN];
    assert (prefix != NULL);
    snprintf (filename, MAXPATHLEN, "%s%06i", prefix, n);
    save_gpu_memory (state, MAP_POSITION, filename);
}

static unsigned int find_starting_num (const char *prefix, unsigned int step)
{
    unsigned int i = 0;
    char path[MAXPATHLEN];
    struct stat s;
    int res;

    assert (prefix != NULL);
    while (1) {
        snprintf (path, MAXPATHLEN, "%s%06i", prefix, i);
        res = stat (path, &s);
        if (res != 0) break;
        i += step;
    }
    if (errno != ENOENT) i = 0;

    return i;
}

static int load_state (struct cl_state *state, const struct config_parameters *config)
{
    int res;

    assert (config->state_position != NULL &&
            config->state_velocity != NULL &&
            config->state_mass != NULL);
    res = restore_gpu_memory (state, MAP_POSITION, config->state_position);
    if (!res) return 0;

    res = restore_gpu_memory (state, MAP_VELOCITY, config->state_velocity);
    if (!res) return 0;

    res = restore_gpu_memory (state, MAP_MASS, config->state_mass);
    if (!res) return 0;

    return 1;
}

static int save_state (struct cl_state *state, const struct config_parameters *config)
{
    int res;

    assert (config->state_position != NULL &&
            config->state_velocity != NULL);

    res = save_gpu_memory (state, MAP_POSITION, config->state_position);
    if (!res) return 0;

    res = save_gpu_memory (state, MAP_VELOCITY, config->state_velocity);
    if (!res) return 0;

    return 1;
}

static void usage()
{
    fprintf (stderr,
             "n-body-sim -n nbodies [-o|--output-steps steps]\n"
             "[-i|--invariant-steps steps] [--output-prefix prefix]\n"
             "[-d delta] [-s solver] [--output-energy out] [--no-update]\n"
             "position velocity mass\n");
    exit(1);
}

#define parse_unsigned(str, place) do {             \
        long val;                                   \
        char *endptr;                               \
        val = strtol (str, &endptr, 10);            \
        if (*endptr != '\0' || val < 0) usage();    \
        place = val;                                \
    } while(0)

#define parse_float(str, place) do {             \
        double val;                              \
        char *endptr;                            \
        val = strtod (str, &endptr);             \
        if (*endptr != '\0' || val < 0) usage(); \
        place = val;                             \
    } while(0)

int main (int argc, char *argv[])
{
    unsigned int i;
    int opt;
    FILE *energy_out = NULL;
    struct cl_state *state = NULL;

    while ((opt = getopt_long (argc, argv, "n:d:s:o:i:", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'n':
            parse_unsigned (optarg, config.nbodies);
            break;
        case 'o':
            parse_unsigned (optarg, config.output_steps);
            break;
        case 'i':
            parse_unsigned (optarg, config.invariant_steps);
            break;
        case 'd':
            parse_float (optarg, config.delta);
            break;
        case 's':
            config.solver = optarg;
            break;
        case SET_OUTPUT_PREFIX:
            config.out_prefix = optarg;
            break;
        case SET_NO_UPDATE:
            config.no_update = 1;
            break;
        case SET_ENERGY_OUT:
            config.energy_file = optarg;
        }
    }

    argc -= optind;
    argv += optind;
    if (argc != 3 || config.nbodies == 0) usage();

    config.state_position = argv[0];
    config.state_velocity = argv[1];
    config.state_mass     = argv[2];

    print_config (&config);
    size_t nbodies = config.nbodies;

    if (config.energy_file != NULL) {
        energy_out = fopen (config.energy_file, "w");
        if (energy_out == NULL) {
            perror ("Cannot open file for energy output");
            goto done;
        }
    }

    state = create_cl_state (config.solver, config.delta);
    if (state == NULL) {
        fprintf (stderr, "Cannot initialize OpenCL state\n");
        goto done;
    }

    nbodies = initialize_memory (state, nbodies);
    if (!nbodies) {
        fprintf (stderr, "Cannot initialize memory\n");
        goto done;
    }

    if (!load_state (state, &config)) {
        fprintf (stderr, "Cannot load state, exiting\n");
        goto done;
    }

    i = find_starting_num (config.out_prefix, config.output_steps);

    signal (SIGINT, cleanup);
    signal (SIGTERM, cleanup);

    while (do_loop) {
        if (i % config.output_steps == 0) save_position (state, config.out_prefix, i);
        if (i % config.invariant_steps == 0) {
            cl_float kin = kinetic_energy (state);
            cl_float pot = potential_energy (state);
            printf ("\nKinetic energy=%.10e, potential energy=%.10e, total energy=%.10e, "
                    "angular momentum=%.10e\n",
                    kin, pot, kin+pot, angular_momentum (state));
            if (energy_out != NULL) {
                fprintf (energy_out, "%.10e %.10e %.10e\n",
                         kin, pot, kin+pot);
            }
        }
        if (i % 100 == 0) {
            printf ("%i... ", i);
            fflush (stdout);
        }

        take_step (state);
        i++;
    }
    printf ("\n");

    if (!config.no_update) {
        if (!save_state (state, &config))
            fprintf (stderr, "Cannot save state\n");
    }

done:
    if (state != NULL) {
        destroy_cl_state (state);
    }

    if (energy_out != NULL) {
        fclose (energy_out);
    }

    return 0;
}
