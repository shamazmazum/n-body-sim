N-Body-Sim
=========
[![Build Status](https://api.cirrus-ci.com/github/shamazmazum/n-body-sim.svg)](https://cirrus-ci.com/github/shamazmazum/n-body-sim)

**n-body-sim** is a simple program which simulates the movement of N
bodies under the influence of their gravitational forces in two
dimensional space. It uses simple O(n^2) algorithm executed on GPU,
faster O(n*log n) will be implemented later.

The gravitational force used in the program is:
$\vec{F} = G \frac{m_1 m_2}{({\left|\vec{r} \right|}^2 + \epsilon )^{3/2}} \vec{r}$
where ε=1 and G=100.

Usage:
```
n-body-sim -n nbodies [-o steps] [-i steps] [-d delta] [-s solver]
[--output-prefix prefix] [--no-update] position velocity mass
```

The flags mean following:
* `-n nbodies` Set the number of simulated bodies to `nbodies`. This
  is automatically rounded to be a multiple of GPU working group size
  (see below).
* `-o steps` Write an output file each `steps` iteration. The output
  file is a text file containing coordinates of each body, one
  coordinate per line. X and Y coordinates are separated by space.
* `-i steps` Calculate motion invariants each `steps`
  iteration. Invariants include total energy and angular momentum.
* `-d delta` A measure of time between two iterations. Smaller value
  will result in more accurate computations. When `delta` is good
  enough, the invariants of movement will change only insignificantly.
* `-s solver` A solver for the equation of movement. Three solvers are
  supported: `euler`, `rk2` and `rk4`. `rk2` and `rk4` are 2nd- and
  4th- order Runge-Kutta methods. `euler` is the fastest among them
  and `rk4` is the most accurate.
* `--output-prefix prefix` The names of output files will begin with
  this prefix.
* `--no-update` Do not update input files on exit (see below).

The last three mandatory arguments are paths to files containing
initial coordinates, velocities and masses of particles in the
system. All of them must contain exactly `nbodies` lines. For files
containing positions and velocities there must be a vector on each
line, X and Y coordinates must be separated by space or many spaces.
For a file which contains masses, there must be exactly one number on
each line. One set of these files is included in the `examples`
directory of this repository.

When the program receives `SIGINT` or `SIGTERM` signal (when you press
`^C` or `kill` it), it overwrites `position` and `velocity` files with
updated values unless `--no-update` option is specified.

Current restrictions: the number of bodies must be a multiple of
maximal GPU working group size (usually 256, see output of
`clinfo`). Wrong number of bodies is rounded to the biggest possible
number which is less than the number specified. A good number to start
the simulation with is around 10k.

Output files can be converted to png (or jpeg) pictures with `gnuplot`
and a script like this:

```
#!/bin/sh

gnuplot << EOF
size = 45000

set terminal png size 1920,1080 enhanced font "Helvetica,20"
set xrange [-size: size]
set yrange [-size: size]

set output "$1.png"
plot "$1" with dots
EOF
```

Installation
-----------
You need these dependencies installed:
*  OpenCL
*  [program-map](https://github.com/shamazmazum/program-map) (a small helper for loading OpenCL programs)
* Cmake for building

The building procedure is following:
```
mkdir build
cd build
cmake ..
make
make install
```
