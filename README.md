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

# Example Output

Below is the output from the data defined in the repo at the time of this writing:
```shell
$ ./garden_calc.py
Fitness: 289
* On Wednesday, February 01, 2023:
   For 4x4 box: 'Closest to House':
        1: At 0x2, sow 'Cherry Tomato' (2550389979).
   For 4x4 box: 'Near Shed':
        2: At 0x1, sow 'Brandywine Tomato' (618076297).
        3: At 0x0, sow 'Brandywine Tomato' (1161215676).
        4: At 1x1, sow 'Brandywine Tomato' (4073726003).
        5: At 1x0, sow 'Brandywine Tomato' (1963134027).
        6: At 1x2, sow 'Brandywine Tomato' (2352062229).
        7: At 2x1, sow 'Brandywine Tomato' (2013941066).
        8: At 2x0, sow 'Cherry Tomato' (1490495288).
        9: At 0x2, sow 'Cherry Tomato' (4205018738).
   For 8x8 box: '3328514497':
       10: At 4x4, sow 'Brandywine Tomato' (2890773338).

* On Wednesday, March 01, 2023:
   For 4x4 box: 'Near Shed':
       11: At 3x1, sow 'Green Beans' (1525350292).
       12: At 3x0, sow 'Green Beans' (1634174451).
       13: At 2x2, sow 'Green Beans' (889438129).

* On Wednesday, March 15, 2023:
   For 4x4 box: 'Closest to House':
       14: At 0x1, sow 'None' (827751293).
       15: At 0x0, sow 'None' (3264362877).
       16: At 1x0, sow 'None' (1325011333).
       17: At 1x1, sow 'Canteloupe' (3174933468).
   For 8x8 box: '3328514497':
       18: At 5x3, sow 'Canteloupe' (2463015866).
       19: At 3x0, sow 'Canteloupe' (3709911670).

* On Wednesday, April 05, 2023:
   For 4x4 box: 'Closest to House':
       20: At 0x2, reap 'Cherry Tomato' (2550389979). Plant was 63 days old.
   For 4x4 box: 'Near Shed':
       21: At 0x1, reap 'Brandywine Tomato' (618076297). Plant was 63 days old.
       22: At 0x0, reap 'Brandywine Tomato' (1161215676). Plant was 63 days old.
       23: At 1x1, reap 'Brandywine Tomato' (4073726003). Plant was 63 days old.
       24: At 1x0, reap 'Brandywine Tomato' (1963134027). Plant was 63 days old.
       25: At 1x2, reap 'Brandywine Tomato' (2352062229). Plant was 63 days old.
       26: At 2x1, reap 'Brandywine Tomato' (2013941066). Plant was 63 days old.
       27: At 2x0, reap 'Cherry Tomato' (1490495288). Plant was 63 days old.
       28: At 0x2, reap 'Cherry Tomato' (4205018738). Plant was 63 days old.
   For 8x8 box: '3328514497':
       29: At 4x4, reap 'Brandywine Tomato' (2890773338). Plant was 63 days old.

* On Wednesday, April 26, 2023:
   For 4x4 box: 'Near Shed':
       30: At 3x1, reap 'Green Beans' (1525350292). Plant was 56 days old.
       31: At 3x0, reap 'Green Beans' (1634174451). Plant was 56 days old.
       32: At 2x2, reap 'Green Beans' (889438129). Plant was 56 days old.

* On Wednesday, May 10, 2023:
   For 4x4 box: 'Closest to House':
       33: At 0x1, reap 'None' (827751293). Plant was 56 days old.
       34: At 0x0, reap 'None' (3264362877). Plant was 56 days old.
       35: At 1x0, reap 'None' (1325011333). Plant was 56 days old.

* On Wednesday, May 24, 2023:
   For 4x4 box: 'Closest to House':
       36: At 1x1, reap 'Canteloupe' (3174933468). Plant was 70 days old.
   For 8x8 box: '3328514497':
       37: At 5x3, reap 'Canteloupe' (2463015866). Plant was 70 days old.
       38: At 3x0, reap 'Canteloupe' (3709911670). Plant was 70 days old.

* On Wednesday, September 20, 2023:
   For 4x4 box: 'Near Shed':
       39: At 1x2, sow 'Jalapeno Pepper' (652467628).
       40: At 1x1, sow 'Jalapeno Pepper' (2387044683).
   For 8x8 box: '3328514497':
       41: At 6x3, sow 'Jalapeno Pepper' (3335117289).

* On Wednesday, November 22, 2023:
   For 4x4 box: 'Near Shed':
       42: At 1x2, reap 'Jalapeno Pepper' (652467628). Plant was 63 days old.
       43: At 1x1, reap 'Jalapeno Pepper' (2387044683). Plant was 63 days old.
   For 8x8 box: '3328514497':
       44: At 6x3, reap 'Jalapeno Pepper' (3335117289). Plant was 63 days old.
```
