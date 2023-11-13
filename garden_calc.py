#!/bin/env python3
# -*- coding: utf-8 -*-

###################################################
# Author: Eric Bullen
# Date: 2016-07-09
# Description: This program generates a planting
# schedule based off of plant data stored in a yaml
# file.
###################################################

import locale
import logging
import multiprocessing as mp
import time
from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from itertools import chain
from uuid import uuid4

import numpy as np
import yaml

locale.setlocale(locale.LC_ALL, ('en_US', 'utf-8'))

# Log to the screen
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s: "%(name)s" (line: %(lineno)d) - %(levelname)s %(message)s'))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(stream_handler)

class Garden:
    def __init__(self, garden_info_file, start_date, usda_zone, boxes, sim_interval_days=None, debug=None):
        self.usda_zone = usda_zone
        self.start_date = start_date

        self.planting_history = defaultdict(deque)
        self.planter_box_plants = defaultdict(dict)
        self._planter_boxes = dict()

        self.planting_log = list()

        if debug is None:
            self.debug = False
        else:
            self.debug = debug

        # This is the day interval for when to check for
        # what plants should be planted, or reaped. 7 days
        # is probably granular enough- it's unlikely a plant
        # needs sub-week planting/reaping accuracy
        if sim_interval_days is None:
            self.sim_interval_days = 7
        else:
            self.sim_interval_days = sim_interval_days

        self.raw_boxes = boxes

        self.initialize_garden()
        self.best_garden_layout = None

        self.garden_data = read_yaml(path=garden_info_file)

        self.plant_list = PlantList(start_date=self.start_date, usda_zone=self.usda_zone, garden_data=self.garden_data)

    def __repr__(self):
        return f"Garden: Zone: {self.usda_zone}, Fitness: {self.fitness}"

    @property
    def planting_schedule(self):
        output = list()
        planting_log = defaultdict(list)

        [planting_log[entry_date].append((planter_box, coords, entry_log)) for entry_date, coords, planter_box, entry_log in self.planting_log]

        old_date_key = None
        log_num = 1

        for date_key in [entry[0] for entry in self.planting_log]:
            if old_date_key == date_key:
                continue
            else:
                old_date_key = date_key

            output.append(f"* On {date_key.strftime('%A, %B %d, %Y')}:")

            curr_box = None

            for planter_box, (x0, y0), log_entry in sorted(planting_log[date_key], key=lambda x: x[0].id):
                if not curr_box or curr_box.id != planter_box.id:
                    output.append(f"   For {planter_box.width}x{planter_box.length} box: '{planter_box.name}':")
                    output.append(f"      {log_num:>3}: At {x0}x{y0}, {log_entry}")
                    curr_box = planter_box
                else:
                    output.append(f"      {log_num:>3}: At {x0}x{y0}, {log_entry}")

                log_num += 1

            output.append("")

        return "\n".join(output)

    @property
    def fitness(self):
        return sum([self.score_planter(planter_box) for planter_box in self.planter_boxes])

    @property
    def layout(self):
        result = list()

        for planter_box in self.planter_boxes:
            pb_fitness = self.score_planter(planter_box)
            result.append(f"Planter Box Fitness: {pb_fitness}")
            result.append(f"{planter_box.layout}")

        return "\n".join(result)

    @property
    def state(self):
        pb_states = list()
        plants_states = list()

        [pb_states.append((pb_id, planter_box.state)) for pb_id, planter_box in self._planter_boxes.items()]

        for pb_id in self.planter_box_plants:
            for plant_id in self.planter_box_plants[pb_id]:
                plant = self.planter_box_plants[pb_id][plant_id]
                plants_states.append((pb_id, plant_id, plant.state))

        return pb_states, self.plant_list.state, plants_states

    @state.setter
    def state(self, data):
        pb_states, plant_list_state, plants_states = data

        for pb_id, pb_state in pb_states:
            self._planter_boxes[pb_id].state = pb_state

        self.plant_list.state = plant_list_state

        self.planter_box_plants = defaultdict(dict)

        for pb_id, plant_id, plant_state in plants_states:
            plant_state, planter_box_id = plant_state

            plant = self.plant_list.all_plants[plant_id]
            plant.state = plant_state
            plant.planter_box = self._planter_boxes[planter_box_id]
            self.planter_box_plants[pb_id][plant_id] = plant

    @property
    def planter_boxes(self):
       return list(self._planter_boxes.values())

    def initialize_garden(self):
        self._planter_boxes = dict()

        for box in self.raw_boxes:
            name = box.get("name", "")
            x, y = map(int, box["size"].split("x"))

            planter_box = PlanterBox(x, y, name)
            self._planter_boxes[planter_box.id] = planter_box

    def add_plants(self, plants):
        for plant in plants:
            self.plant_list.add_plant(plant)

    def find_space(self, plant):
        # Don't want to beat up on a single planter
        # box every time, so shuffling it (permutations)
        for planter_box in np.random.permutation(self.planter_boxes):
            planter_box_length, planter_box_width = planter_box.shape

            # Makes the check start in random spots in the garden
            # box, so that each generation doesn't put things back
            # in the same position
            end_x = planter_box_width - plant.width
            end_y = planter_box_length - plant.length

            start_x = np.random.randint(0, end_x)
            start_y = np.random.randint(0, end_y)

            y_chain = chain(range(start_y, end_y + 1), range(0, start_y))
            x_chain = chain(range(start_x, end_x + 1), range(0, start_x))

            for y in y_chain:
                y1 = y + plant.length

                for x in x_chain:
                    x1 = x + plant.width

                    # Short circuit the 'and' below to see if it has a value first
                    # before doing a 2-d comnparison
                    if not any(chain.from_iterable(planter_box.box[y:y1, x:x1])):
                        return planter_box, x, y

        # No spots found
        return None, None, None

    def bounding_box(self, plant, planter_box):
        planter_box_length, planter_box_width = planter_box.shape

        x0 = plant.x0 - 1
        y0 = plant.y0 - 1
        x1 = plant.x0 + plant.width
        y1 = plant.y0 + plant.length

        x0 = 0 if x0 < 0 else x0
        y0 = 0 if y0 < 0 else y0
        x1 = planter_box_width - 1 if x1 >= planter_box_width - 1 else x1
        y1 = planter_box_length - 1 if y1 >= planter_box_length - 1 else y1

        return x0, y0, x1, y1

    def neighbors(self, plant):
        bbox_x0, bbox_y0, bbox_x1, bbox_y1 = self.bounding_box(plant, plant.planter_box)

        data = plant.planter_box.box[bbox_y0:bbox_y1 + 1, bbox_x0:bbox_x1 + 1]

        return [self.plant_list.all_plants[plant_id] for plant_id in set(data[np.where((data != plant.id) & (data != 0))])]

    def pull_history(self, plant, max_lookback=None):
        history = list()

        if max_lookback is None:
            max_lookback = 3

        x0, y0, x1, y1 = plant.x0, plant.y0, plant.x0 + plant.width, plant.y0 + plant.length

        for index, entry in enumerate(self.planting_history[plant.planter_box.id]):
            reap_date, plant_x0, plant_y0, plant_x1, plant_y1, plant = entry

            if ((plant_x0 <= x0 <= plant_x1 and plant_y0 <= y0 <= plant_y1) or
                (plant_x0 <= x1 <= plant_x1 and plant_y0 <= y1 <= plant_y1)):
                history.append((reap_date, plant))

            if index >= max_lookback:
                break

        return history

    def check_neighbor_score(self, plant, neighbor_plants):
        fitness = 0

        for neighbor_plant in neighbor_plants:
            if neighbor_plant is None:
                fitness += 1
            else:
                if plant.is_combative(neighbor_plant):
                    fitness -= 2
                elif plant.is_companion(neighbor_plant):
                    fitness += 2
                else:
                    fitness += 1

        return fitness

    def check_history_score(self, plant):
        fitness = 0

        # elapsed_days = (datetime.now() - planting_date).days
        history = self.pull_history(plant)

        if history:
            age1_plant_family = history[0][1].plant_family

            try:
                age2_plant_family = history[1][1].plant_family
            except IndexError:
                age2_plant_family = None

            try:
                age3_plant_family = history[2][1].plant_family
            except IndexError:
                age3_plant_family = None

            # now, the logic
            if age1_plant_family == plant.plant_family:
                fitness -= 1

                if age2_plant_family == plant.plant_family:
                    fitness -= 5

                    if age3_plant_family == plant.plant_family:
                        fitness -= 10

            # Bean family
            elif age1_plant_family == "fabaceae":
                fitness += 5

            else:
                fitness += 1

        return fitness

    def score_planter(self, planter_box):
        fitness = 0

        for plant in self.planter_box_plants[planter_box.id].values():
            fitness += self.check_neighbor_score(plant, self.neighbors(plant))
            fitness += self.check_history_score(plant)

        # Add the number of unique plant types to the fitness score;
        # not having all of the same type should be a bonus
        fitness += planter_box.plant_count

        return fitness

    def planting_priority_list(self, check_date, mutation_chance):
        planting_list = self.plant_list.planting_priority_list(check_date)

        if not planting_list:
            return list()

        if not mutation_chance:
            return list(zip(*sorted(planting_list, key=lambda entry: entry[1], reverse=True)))[0]

        plant_list, weights = map(list, zip(*planting_list))

        old_mutation_method = False

        if old_mutation_method:
            # This builds the plant list doing a random sample where
            # the weights are the scores normalized between 0 and 1.
            #
            # This way is about 30% slower
            #
            plant_list = np.random.choice(plant_list, replace=False, size=len(plant_list), p=weights)
        else:
            # This builds the plant list based on a probability for
            # each element to be swapped with some other element
            index = list(range(len(plant_list)))
            shuffled_index = np.random.permutation(index[:])

            for index, shuffled_index in zip(index, shuffled_index):
                if np.random.random() <= mutation_chance:
                    plant_list[index], plant_list[shuffled_index] = plant_list[shuffled_index], plant_list[index]

        return plant_list

    def sow(self, check_date, plant, planter_box, x, y):
        self.plant_list.sow(check_date, plant, planter_box, x, y)
        self.planter_box_plants[planter_box.id][plant.id] = plant

        self.planting_log.append((check_date,
                                 (plant.x0, plant.y0),
                                 plant.planter_box,
                                 f"sow '{plant.name}' ({plant.id})."))

    def reap(self, check_date, plant):
        # Store the plant in history
        self.planting_history[plant.planter_box.id].appendleft((check_date,
                                                                plant.x0,
                                                                plant.y0,
                                                                plant.x0 + plant.width,
                                                                plant.y0 + plant.length,
                                                                plant))

        # Delete the mapping
        self.plant_list.reap(check_date, plant)
        del self.planter_box_plants[plant.planter_box.id][plant.id]

        self.planting_log.append((check_date,
                                 (plant.x0, plant.y0),
                                 plant.planter_box,
                                 f"reap '{plant.name}' ({plant.id}). Plant was {(check_date - plant.sow_date).days} days old."))

    def reap_plants(self, check_date):
        reap_count = 0
        reaped_planting_log = list()

        reapable_plants = self.plant_list.reapable_plants(check_date)

        if reapable_plants:
            orig_planting_log = self.planting_log
            self.planting_log = list()

            for plant in reapable_plants:
                self.reap(check_date, plant)
                reap_count += 1

            reaped_planting_log = self.planting_log
            self.planting_log = orig_planting_log

        return reap_count, reaped_planting_log

    def sow_plants(self, check_date, generations, mutation_chance=None):
        max_fitness = 0
        max_sow_count = 0
        best_planting_log = list()
        best_garden_layout = None

        if mutation_chance is None:
            mutation_chance = 0.0
        else:
            # Since I'm swapping variables, divide the value by 2
            mutation_chance /= 2.0

        # Save the data
        orig_state = self.state
        orig_planting_log = self.planting_log

        for i in range(generations):
            # Reset to before the simulation
            sow_count = 0
            self.planting_log = list()

            # Plant what we can
            for plant in self.planting_priority_list(check_date=check_date, mutation_chance=mutation_chance):
                max_plantings_per_interval = self.plant_list.planting_limits[plant.name]
                interval_planting_count = self.plant_list.planting_counter[plant.name]

                max_inground_count = self.plant_list.max_in_ground_count[plant.name]
                inground_count = self.plant_list.in_ground_count[plant.name]

                if inground_count >= max_inground_count or interval_planting_count >= max_plantings_per_interval:
                    continue

                planter_box, x, y = self.find_space(plant)

                if planter_box:
                    self.sow(check_date, plant, planter_box, x, y)
                    sow_count += 1

            # Now score it
            fitness = self.fitness

            if fitness > max_fitness:
                logger.debug(f"Old: {max_fitness} New: {fitness} Iterations: {i}/{generations}")
                max_fitness = fitness
                max_sow_count = sow_count
                best_garden_layout = self.state
                best_planting_log = self.planting_log

            self.state = orig_state

        # Tried X generations, now set the box to the best score.
        self.state = orig_state
        self.planting_log = orig_planting_log

        return max_sow_count, max_fitness, best_garden_layout, best_planting_log

    def build_schedule(self, local_garden, random_seed, output_queue, working_date, mutation_chance, schedule_generations):
        np.random.seed(random_seed)

        max_fitness = 0
        top_garden_log = list()
        top_state = None

        start_date = working_date

        year_boundary = start_date + timedelta(days=365 * 2)
        initial_state = local_garden.state

        for i in range(schedule_generations):
            fitness = 0
            garden_log = list()

            # Was this a bad generation?
            bad_generation = False

            while local_garden.plant_list.unsown_plants or local_garden.plant_list.unreaped_plants:
                if working_date > year_boundary:
                    logger.debug("WARNING: Too much time has elapsed (still unsown or unreaped plants). Failing this scenario.")
                    bad_generation = True
                    break

                logger.debug(f"Checking Date: {date.strftime(working_date, '%m/%d/%y')}")

                # Reap anything in ALL planter boxes
                reaped_count, reaped_planting_log = local_garden.reap_plants(working_date)

                if reaped_count:
                    logger.debug(f"Reaped {reaped_count} plants.")
                    garden_log.extend(reaped_planting_log)
                else:
                    reaped_count = 1

                # This is for throttling how much is planted per interval. I picked '80' because
                # the fitness score doesn't improve much beyond 1000 generations w/ 13 plants, so
                # set the value to 80 (1000 / 13).
                if any(local_garden.plant_list.sowable_plants(working_date)):
                    sow_count, best_fitness, best_state, best_planting_log = local_garden.sow_plants(check_date=working_date,
                                                                                                     generations=reaped_count * 80,
                                                                                                     mutation_chance=mutation_chance)

                    local_garden.state = best_state
                    garden_log.extend(best_planting_log)

                    if sow_count:
                        fitness += best_fitness

                else:
                    logger.debug("Nothing to sow...")

                working_date += timedelta(days=local_garden.sim_interval_days)

            # Rewind back one week as it gets incremented at the end regardless
            working_date -= timedelta(days=local_garden.sim_interval_days)

            # After the year is over, OR if there is nothing left to sow/reap
            if not bad_generation and fitness > max_fitness:
                max_fitness = fitness
                top_state = local_garden.state
                top_garden_log = garden_log

            local_garden.state = initial_state
            working_date = start_date

        # Make the state of the garden the best state
        if top_state:
            local_garden.state = top_state
            local_garden.planting_log = top_garden_log

        output_queue.put((max_fitness, self.planting_schedule))

    def run_simulation(self, iterations, mutation_chance):
        results = list()
        output_queue = mp.Queue()

        if self.debug:
            worker_count = 1
            self.build_schedule(self,
                                np.random.randint(0, 2 ** 32),
                                output_queue,
                                self.start_date,
                                mutation_chance,
                                int(iterations / worker_count))
        else:
            worker_count = mp.cpu_count()

            for i in range(worker_count):
                mp.Process(target=self.build_schedule,
                           args=(self,
                                 np.random.randint(0, 2 ** 32),
                                 output_queue,
                                 self.start_date,
                                 mutation_chance,
                                 int(iterations / worker_count))).start()

        for i in range(worker_count):
            results.append(output_queue.get())

        return sorted(results, key=lambda entry: entry[0], reverse=True)[0]


class PlanterBox:
    def __init__(self, width, length, name=None):
        self.id = uuid4().time_low

        self.width = width
        self.length = length
        self.plant_count = 0
        self._shape = None

        if not name:
            self.name = self.id
        else:
            self.name = name

        # Should the fill type be a deque to store
        # history for each section?
        self.box = np.zeros((self.length, self.width), dtype=np.uint32)

    def __repr__(self):
        return f"Planter Box ID: {self.name}, Size: {self.width}x{self.length}, {self.plant_count} plants"

    @property
    def state(self):
        return self.__dict__.copy(), np.copy(self.box)

    @state.setter
    def state(self, data):
        obj_dict, box = data

        self.__dict__.update(obj_dict)
        self.box = np.copy(box)

    @property
    def layout(self):
        return f"Planter Box ID: {self.name}, Size: {self.width}x{self.length}, {self.plant_count} plants\n{self.box}\n"

    @property
    def shape(self):
        if self._shape is None:
            self._shape = self.box.shape

        return self._shape


class PlantList:
    def __init__(self, start_date, garden_data, usda_zone):
        self.all_plants = dict()

        self.start_date = start_date
        self.garden_data = garden_data
        self.usda_zone = usda_zone

        self._unsown_plants = None
        self._unreaped_plants = None

        self.planting_limits = dict()
        self.max_in_ground_count = dict()
        self.in_ground_count = defaultdict(int)
        self.planting_counter = defaultdict(int)
        self.number_wanted = defaultdict(int)
        self.number_planted = defaultdict(int)

        self.number_harvested = 0

    def __repr__(self):
        return (f"Plant List: Total Plants: {len(self.all_plants.keys())}, "
                f"Wanted: {len(self.number_wanted.keys())}, "
                f"Planted: {len(self.number_planted.keys())}, "
                f"Harvested: {self.number_harvested}")

    @property
    def state(self):
        plant_state = [(plant_id, plant.state) for plant_id, plant in self.all_plants.items()]

        counters = [self.planting_limits.copy(),
                    self.planting_counter.copy(),
                    self.number_planted.copy(),
                    self.number_wanted.copy(),
                    self.in_ground_count.copy(),
                    self.max_in_ground_count.copy()]

        return self.__dict__.copy(), plant_state, counters

    @state.setter
    def state(self, data):
        # 'data' could be emptty
        if data:
            obj_dict, plant_state, counters = data

            self.__dict__.update(obj_dict)

            [self.all_plants[plant_id].__dict__.update(new_plant_state[0]) for plant_id, new_plant_state in plant_state]

            self.planting_limits = dict()
            self.max_in_ground_count = dict()
            self.in_ground_count = defaultdict(int)
            self.planting_counter = defaultdict(int)
            self.number_planted = defaultdict(int)
            self.number_wanted = defaultdict(int)

            self.planting_limits.update(counters[0])
            self.planting_counter.update(counters[1])
            self.number_planted.update(counters[2])
            self.number_wanted.update(counters[3])
            self.in_ground_count.update(counters[4])
            self.max_in_ground_count.update(counters[5])

    @property
    def unsown_plants(self):
        if not self._unsown_plants:
            self._unsown_plants = [plant for plant in self.all_plants.values() if not plant.sow_date]

        return self._unsown_plants

    @property
    def unreaped_plants(self):
        if not self._unreaped_plants:
            self._unreaped_plants = [plant for plant in self.all_plants.values() if plant.sow_date and not plant.reap_date]

        return self._unreaped_plants

    def sowable_plants(self, check_date):
        return [plant for plant in self.unsown_plants if plant.can_sow(check_date)]

    def reapable_plants(self, check_date):
        return [plant for plant in self.unreaped_plants if plant.can_reap(check_date)]

    def priority_score(self, plant, check_date):
        number_wanted = self.number_wanted[plant.name]
        number_planted = self.number_planted[plant.name]
        total_plants = number_wanted + number_planted

        number_planted = 1 if not number_planted else number_planted

        total_days, days_left = plant.get_sow_date_info(check_date)

        return (total_plants / number_planted) * (total_days / float(days_left))

    def planting_priority_list(self, check_date):
        result = list()

        unordered_plants = [(plant, self.priority_score(plant, check_date)) for plant in self.sowable_plants(check_date)]

        if unordered_plants:
            total_score = sum(list(zip(*unordered_plants))[1])

            # Now to bound the score between 0 and 1 inclusive
            unordered_plants = [(plant, score / total_score) for plant, score in unordered_plants]

            result = sorted(unordered_plants, key=lambda entry: entry[1], reverse=True)

        return result

    def add_plant(self, plant):
        name = plant.get("name")
        plant_type = plant["type"]
        number_wanted = plant.get("number_wanted", 1)
        max_in_ground_count = plant.get("max_in_ground_count", float("+Inf"))
        max_plantings_per_interval = plant.get("max_plantings_per_interval", float("+Inf"))

        for i in range(number_wanted):
            plant = Plant(start_date=self.start_date,
                          name=name,
                          plant_type=plant_type,
                          garden_data=self.garden_data,
                          usda_zone=self.usda_zone)

            self.all_plants[plant.id] = plant
            self.planting_limits[plant.name] = max_plantings_per_interval
            self.max_in_ground_count[plant.name] = max_in_ground_count

        self.number_wanted[plant.name] += number_wanted

    def reap(self, check_date, plant):
        plant.reap(check_date)

        # clear the cache
        self._unreaped_plants = None

        self.number_harvested += 1
        self.in_ground_count[plant.name] -= 1

    def sow(self, check_date, plant, planter_box, x, y):
        plant.sow(check_date, planter_box, x, y)

        # clear the cache
        self._unreaped_plants = None
        self._unsown_plants = None

        self.number_wanted[plant.name] -= 1
        self.number_planted[plant.name] += 1
        self.planting_counter[plant.name] += 1
        self.in_ground_count[plant.name] += 1


class Plant:
    def __init__(self, start_date, plant_type, name, garden_data, usda_zone):
        self.id = uuid4().time_low

        self.start_date = start_date
        self.garden_data = garden_data
        self.name = name
        self.plant_type = plant_type

        usda_zone = f"zone_{usda_zone}"

        if usda_zone not in self.garden_data["plants"][self.plant_type]["zone_data"]["usda_zone"]:
            raise ValueError(f"Zone \'{usda_zone.removeprefix('zone_')}\' not a known usda zone for plant '{self.plant_type}'.")

        self.usda_zone = usda_zone

        self.x0 = None
        self.y0 = None
        self.planter_box = None

        self.sow_date = None
        self.reap_date = None
        self.maturity_date = None

        self.combative_plants = set()
        self.companion_plants = set()
        self.plant_family = None

        self._sow_cache = dict()

        self.initialize()

    def __repr__(self):
        if self.planter_box:
            return f"{self.name} (id: {self.id}) @ {self.planter_box.name} @ coords: {self.x0}x{self.y0}"
        else:
            return f"{self.name} (id: {self.id}) @ (not planted)"

    def initialize(self):
        plant = self.garden_data["plants"][self.plant_type]

        self.width = plant["dimensions"]["width"]
        self.length = plant["dimensions"]["width"]
        self.maturity_days = plant["zone_data"]["usda_zone"][self.usda_zone]["maturity_days"]

        self.sow_dates = self.set_sow_dates(usda_zone=self.usda_zone)

        self.combative_plants = set(plant["combative_plants"])
        self.companion_plants = set(plant["companion_plants"])
        self.plant_family = plant["classification"]["family"]

    @property
    def state(self):
        pb_id = None

        if self.planter_box is not None:
            pb_id = self.planter_box.id

        return self.__dict__.copy(), pb_id

    @state.setter
    def state(self, data):
        self.__dict__.update(data)

    def get_sow_date_info(self, check_date):
        if check_date not in self._sow_cache:
            for start_date, end_date in [(start_date, end_date) for start_date, end_date in self.sow_dates if start_date <= check_date <= end_date]:
                total_days = (end_date - start_date).days
                days_left = (end_date - check_date).days

                # I'm setting this to .01 so that the priority score is very high
                days_left = .01 if days_left < 1 else days_left

                self._sow_cache[check_date] = total_days, days_left
                break

        return self._sow_cache[check_date]

    def reap(self, check_date):
        self.reap_date = check_date

        # Clear out the plot for the next plant
        self.planter_box.box[self.y0:self.y0 + self.length, self.x0:self.x0 + self.width] = 0
        self.planter_box.plant_count -= 1

    def sow(self, check_date, planter_box, x, y):
        self.sow_date = check_date
        self.maturity_date = self.sow_date + timedelta(days=self.maturity_days)

        self.planter_box = planter_box
        self.x0 = x
        self.y0 = y

        self.planter_box.box[self.y0:self.y0 + self.length, self.x0:self.x0 + self.width] = self.id
        self.planter_box.plant_count += 1

    def can_reap(self, check_date):
        return self.sow_date and check_date >= self.maturity_date

    def can_sow(self, check_date):
        return not self.sow_date and any([start_date <= check_date <= end_date for start_date, end_date in self.sow_dates])

    def set_sow_dates(self, usda_zone):
        result = list()

        sow_dates = self.garden_data["plants"][self.plant_type]["zone_data"]["usda_zone"][usda_zone]["planting_dates"]

        for date_range in sow_dates:
            start_date, end_date = [date.fromtimestamp(time.mktime(time.strptime(check_date.strip(), "%m/%d"))) for check_date in date_range.split("-")]

            start_date = start_date.replace(year=self.start_date.year)
            end_date = end_date.replace(year=self.start_date.year)

            if start_date < self.start_date:
                start_date = start_date.replace(year=self.start_date.year + 1)
                end_date = end_date.replace(year=self.start_date.year + 1)

            result.append((start_date, end_date))

        return result

    def is_companion(self, other_plant):
        return other_plant.plant_type in self.companion_plants

    def is_combative(self, other_plant):
        return other_plant.plant_type in self.combative_plants


def read_yaml(path):
    result = None

    with open(path, "r", encoding="utf8") as fh:
        result = yaml.safe_load(fh)

    return result



if __name__ == "__main__":
    now = datetime.now()

    conf_file = "schedule.yaml"
    schedule = read_yaml(conf_file)

    zone = schedule["garden"]["zone"]
    boxes = schedule["garden"]["boxes"]
    data_file = schedule.get("simulation", {}).get("data_file", "garden_info.yaml")
    start_date = schedule.get("simulation", {}).get("start_date", date.today().strftime("%m/%d/%Y"))
    mutation_chance = schedule.get("simulation", {}).get("mutation_chance", 0.2)
    iterations = schedule.get("simulation", {}).get("iterations", 400)

    try:
        start_date = datetime.strptime(start_date, "%m/%d/%Y").date()
    except ValueError:
        start_date = datetime.strptime(start_date, "%m/%d").replace(year=now.year).date()

    garden = Garden(garden_info_file=data_file,
                    start_date=start_date,
                    usda_zone=zone,
                    boxes=boxes)

    garden.add_plants(plants=schedule["garden"]["desired_plants"])

    # Raisig the interations gives it a better chance at a
    # higher fitness at a cost of more CPU
    fitness, planting_schedule = garden.run_simulation(iterations=iterations,
                                                        mutation_chance=mutation_chance)
    print(f"Fitness: {fitness}")
    print(planting_schedule)
