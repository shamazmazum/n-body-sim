Examples
--------

This is an example set of 10240 bodies with the equal mass 0.05
distributed in the circle with center at (0,0) and radius 10. At the
time t=0 all the bodies have initial speed 0. You can try the
simulation with the following parameters:

`n-body-sim -n 10240 -o 20 -d 0.0001 -s rk4 example_position example_velocity`

You can convert the output into images with gnuplot and a script
`plotscript.sh` included in this repo. You can later convert these
images to a video or to a GIF animation. Or you can just look at my
own:

![Generated animation](world.gif)
