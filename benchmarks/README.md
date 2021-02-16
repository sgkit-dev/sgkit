# sgkit Benchmarks

Benchmarking sgkit with Airspeed Velocity.

## Usage

Airspeed Velocity manages building the environment via conda itself. The recipe for the same in
defined in the `asv.conf.json` configuration file.

* To run the benchmark suite:

```bash
asv run --config benchmarks/asv.conf.json
```

## Benchmark Results

The benchmark results are uploaded to the [benchmarks repository][https://github.com/pystatgen/sgkit-benchmarks] via
`benchmark.yml` Github Actions.

They can be viewed locally by running the following commands:


* Generate html

```bash
asv publish --config benchmarks/asv.conf.json -v
```

* Run local server

```bash
asv preview --config benchmarks/asv.conf.json -v
```

The benchmarks over time can be seen here: https://pystatgen.github.io/sgkit-benchmarks


## Writing benchmarks

Benchmarks should be written in the `benchmarks/` directory as `.py` files. For more information on different
types of benchmarks see the documentation here: https://asv.readthedocs.io/en/stable/writing_benchmarks.html#writing-benchmarks


## Benchmark Machine

The benchmark machine is the Github Actions machine, which has roughly the following configurations:

```json
{
    "arch": "x86_64",
    "cpu": "Intel(R) Xeon(R) Platinum 8272CL CPU @ 2.60GHz",
    "machine": "fv-az183-669",
    "num_cpu": "2",
    "os": "Linux 5.4.0-1039-azure",
    "ram": "7121276",
    "version": 1
}
```

The configuration above does changes slightly in every run, for example we could get a machine with different
cpu like say the one with 2.30GHz or the one with slightly less RAM (not a huge deviation from above though).
As of now it is not possible to fix this, unless we use a custom machine for benchmarking, hence minor deviation
in benchmarks performance should be consumed with a pinch of salt.
