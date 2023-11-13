# Overview
A garden planter that uses a genetic algorithm to find the most ideal planting schedule

# Details
There are three main files:

* `garden_calc.py`: This is what you run to output your planting schedule
* `garden_info.yaml`: The database used by `garden_calc.py` to know the planting criteria for a given plant. Explained more below.
* `schedule.yaml`: This is the file that `garden_calc.py` reads. Explained more below.

## The `garden_info.yaml` file
Thers is a lot here, but in general follow the given format if you want to add/change anything (submit as a PR to this repo if you want others to benefit).

One key piece of information - in `schedule.yaml` for any given `desired_plant`, the _type_ (like 'apple', 'tomato', 'bean') must be defined here. The name of the plant as defined in `schedule.yaml` is free-form, and is intended to be used to specify the given variety.

The `dimensions` field is in units. A unit is the smallest granularity that `garden_calc.py` sees. For example, most home gardens may have the unit be just big enough to fit the average plant (maybe 2' x 2'). If it's on a larger scale, it could be meters/yards in size (but everything within that unit is seen as 'one plant').

## The `schedule.yaml` file

There are two main root keys: `simulation` and `garden`. Each is below:

The `simulation` key:
  * `mutation_chance`: 0.2 is default, change this if you want more variance in generations
  * `iterations`: 400 is the default. Increase this number to give a better chance of a schedule with a higher fitness. This can dramatically increase CPU time for running the simulation.
  * `start_date`: Run time date is the default, but you can put a M/D or M/D/Y formatted string, and the simulation will start from that date. This is useful if you want to work on the coming season.
  * `data_file`: garden_info.yaml is default. You can define multiple data files for a given custom schdule if needed.

The `garden` key:
  * `zone`: The USDA zone. Sub-zones (like `8a`) are supported, but not defined as of this writing.
  * `boxes`: This is a list of boxes. Each box can have a `name` key (optional) which is free-form to help you identify the planting area. The size key (required) is based on `units` as defined above.
  * `desired_plants`: This is a list of plants the user wants to plant. There are several keys relevant to this:
    * `name`: Not required, but recommended. This is a free-form line that usually would describe the variety of the plant.
    * `type`: What is defined here _must_ be defined in `garden_info.yaml` file.
    * `max_in_ground_count`: By default (undefined), it's as much as can be planted. This may not be desired as you may want to spread out your harvest over the course of the season.
    * `max_plantings_per_interval`: An `interval` is the time between date checks that the simulation looks for things to sow or harvest. The interval is currently 7 days.
    * `number_wanted`: This is required, and indicates the number of units dedciated to a plant. For home gardens, one unit roughly equals one plant, so they are the same, but on large scale plots, one unit may hold 100 plants.


# Notes
This program uses [Numpy](https://numpy.org/). Install it via your package manager, or install it via pip:

```shell
$ python3 -m pip install --user numpy
```
# Future

* This program sows plants in mostly random locations.  If I have 5 plants, and 5 boxes, there could be one plant per box, and that could reduce fitness. Having it plant what it can in as few of boxes as possible seems like a great feature.
* Expanding on `garden_info.yaml` to add additional plants/features is an on-going goal. Contributions absolutely welcome.
