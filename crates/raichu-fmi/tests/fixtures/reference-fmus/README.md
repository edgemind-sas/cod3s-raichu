# Reference FMU test fixtures

The six `.fmu` files and `LICENSE.txt` come from the Modelica Association's
[Reference FMUs v0.0.41 release](https://github.com/modelica/Reference-FMUs/releases/tag/v0.0.41)
under the accompanying BSD-2-Clause licence. They are kept as complete FMU
archives, including binaries for the supported release platforms.

The two Dahlquist CSV files are independent co-simulation results from
[`fmusim` v0.12.0](https://github.com/modelica/fmusim/releases/tag/v0.12.0):

```sh
fmusim simulate --interface-type cs --start-time 0 --stop-time 1 \
  --output-interval 0.1 --fixed-step-size 0.1 --output-variable x \
  --output-file dahlquist-fmi3-fmusim-0.12.0.csv 3.0/Dahlquist.fmu
```

The FMI 2 file uses the same command with `2.0/Dahlquist.fmu` and its own
output filename. The `reader` test compares every communication point to
these fixed goldens.
