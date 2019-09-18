N-Body-Sim
=========

**n-body-sim** is a simple program which simulates the movement of N
bodies under the influence of their gravitational forces in two
dimensional space. It uses simple O(n^2) algorithm executed on GPU,
faster O(n*log n) will be implemented later.

Using: `n-body-sim [-c config] -s -r`
The flags mean following:
* `-c` Specify configuration file. Look at `examples/config` for more
  info.
* `-s` Save execution state then receiving `SIGINT` (as when pressing
  `^C`) or `SIGTERM` (as when using `kill` or `killall`).
* `-r` When launching restore the state saved with `-s` before. Note,
  that you **MUST** specify the save config file as before.

Current restrictions: the number of bodies must be a multiple of
maximal working group size (usually 256, see output of
`clinfo`). Wrong number of bodies is rounded to the biggest possible
number which is less than the number specified. A good number to start
the simulation with is around 10k.

The output of the program consists of multiple files in the working
directory. Each of them has the coordinates of moving particles (one
coordinate on each line, single values separated by space). Plus, when
using with `-s` option, three additional files will be created upon
exit: `state_mass`, `state_velocity` and `state_position`.

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
*  [Iniparser](https://github.com/ndevilla/iniparser) (present in FreeBSD ports as devel/iniparser)
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

Problems
--------
* When initial speed is zero, angular momentum is not constant. However it is almost constant otherwise.
