import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import random
import math
import os
import time

# ------------------------- INITIALIZATION ------------------------- 

# df = pd.read_csv('Test_Data-cleaned.csv')
# df = pd.read_csv('Final-Test-Data-1.csv')
df = pd.read_csv('Final-Test-Data-2.csv')
# df = pd.read_csv('Final-Test-Data-3.csv')
# df = pd.read_csv('Final-Test-Data-4.csv')
locations_df = pd.read_csv('Locations.csv')

df.set_index('Student Number', inplace=True)

NUM_OF_STUDENTS = len(df)

def getLocations():
    file = open("Locations.csv", 'r')

    lines = file.readlines()

    locations = {line[:-1]:[] for line in lines}
    return locations

LOCAITON_COUNT = len(getLocations())
CAPACITY = math.ceil(NUM_OF_STUDENTS / LOCAITON_COUNT)


# generate a random solution
def generate_random_solution(df):
    locationsList = getLocations()

    for id, student in df.iterrows():
        
        random_number = random.randint(1, 4)

        if random_number == 1:
            selected_loc = df.at[id, 'P1']

        elif random_number == 2:
            selected_loc = df.at[id, 'P2']

        elif random_number == 3:
            selected_loc = df.at[id, 'P3']

        elif random_number == 4:
            selected_loc = df.at[id, 'P4']

        locationsList[selected_loc] += [id]

        for locs, values in locationsList.items():
            locationsList[locs] = list(set(values))
    
    return locationsList

# ------------------------- END OF INITIALIZATION ------------------------- 

# ------------------------- CALCULATION ------------------------- 

# returns a dictionary of { student_number : location }
def display_assigned_locations(solution, df):
    assignment = {}

    for id, student in df.iterrows():
        for locs, values in solution.items():
            if id in values:
                assignment[id] = locs
                
    return assignment

# PRIORITY SCORE
def calculate_priority_score(solution, df):
    preferred_locs = {}
    priority = {}

    assignment = display_assigned_locations(solution, df)

    id = 0
    while id < NUM_OF_STUDENTS:
        id += 1
        p1 = df.at[id, 'P1']
        p2 = df.at[id, 'P2']
        p3 = df.at[id, 'P3']
        p4 = df.at[id, 'P4']

        preferred_locs[id] = [p1, p2, p3, p4]

        if assignment[id] in preferred_locs[id]:
            priority[id] = 1
        else: 
            priority[id] = 0

    prio_score = sum(priority.values()) / len(priority.values())
    
    return prio_score

# GENDER DISTRIBUTION SCORE
def calculate_gender_score(solution, df):
    gender_dict = {}
    male_count = 0
    female_count = 0

    for locs, values in solution.items():
        for id in values:
            gender = df.at[id, 'Gender']

            if gender == 'Male':
                male_count += 1
            elif gender == 'Female':
                female_count += 1

            total = male_count + female_count
            
            male_percent = (male_count / total)
            female_percent = (female_count / total)

            difference = round(abs(male_percent - female_percent), 2)

        gender_dict[locs] = difference
        
        male_count = 0
        female_count = 0

    gender_average = sum(gender_dict.values()) / LOCAITON_COUNT
    
    return 1 - gender_average

# GRADE LEVEL DISTRIBUTION SCORE
def calculate_grade_level_score(solution, df):
    grade_dict = {}
    g9 = 0
    g10 = 0
    g11 = 0
    g12 = 0

    for locs, values in solution.items():
        for id in values:
            grade_level = df.at[id, 'Grade']

            if grade_level == 'Grade 9':
                g9 += 1
            elif grade_level == 'Grade 10':
                g10 += 1
            elif grade_level == 'Grade 11':
                g11 += 1
            elif grade_level == 'Grade 12':
                g12 += 1

            total = g9 + g10 + g11 + g12

            g9_percent = (g9 / total)
            g10_percent = (g10 / total)
            g11_percent = (g11 / total)
            g12_percent = (g12 / total)

            difference = abs(0.25 - g9_percent) + abs(0.25 - g10_percent) + abs(0.25 - g11_percent) + abs(0.25 - g12_percent)    

        grade_dict[locs] = difference
        
        g9 = 0
        g10 = 0
        g11 = 0
        g12 = 0

    grade_average = sum(grade_dict.values()) / LOCAITON_COUNT   
    
    return 1 - grade_average

# CAPACITY SCORE
def calcualte_capacity_score(solution, df):
    capacity_dict = {}
    for locs, values in solution.items():
        empty_slots = CAPACITY - len(values)

        capacity_dict[locs] = empty_slots

    cap_sum = 0
    for num in capacity_dict.values():
        if num < 0:
            cap_sum += num

    cap_percent = abs(cap_sum) / NUM_OF_STUDENTS
    
    return 1 - cap_percent

# FITNESS FUNCTION
def calculate_fitness(solution, df):
    
    gender_average = calculate_gender_score(solution, df)
    grade_average = calculate_grade_level_score(solution, df)   
    cap_percent = calcualte_capacity_score(solution, df)    
    prio_score = calculate_priority_score(solution, df)
        
    weighted_gender = gender_average * 25
    weighted_grade = grade_average * 25
    weighted_cap = cap_percent * 25
    weighted_prio = prio_score * 25

    # the fitness function
    fitness_score = ((gender_average * 100) + (grade_average * 100) + (cap_percent * 100) + (prio_score * 100)) / 4

    weighted_accuracy = weighted_gender + weighted_grade + weighted_cap + weighted_prio
    
    return fitness_score, weighted_accuracy

def results_table(solution, df):
    gender_average = calculate_gender_score(solution, df)
    grade_average = calculate_grade_level_score(solution, df)
    cap_percent = calcualte_capacity_score(solution, df)
    prio_score = calculate_priority_score(solution, df)

    weighted_gender = gender_average * 25
    weighted_grade = grade_average * 25
    weighted_cap = cap_percent * 25
    weighted_prio = prio_score * 25

    fitness_score, weighted_accuracy = calculate_fitness(solution, df)

    score = {
        'Criteria' : ['Gender Distribution', 'Grade Level Distribution', 'Capacity', 'Priority', 'Fitness Score'],
        'Score %' : [round(gender_average * 100, 2), round(grade_average * 100, 2), round(cap_percent * 100, 2), round(prio_score * 100, 2), fitness_score],
        'Weight %' : ['25%', '25%', '25%', '25%', '100%'],
        'Weighted Score' : [weighted_gender, weighted_grade, weighted_cap, weighted_prio, weighted_accuracy]
    }

    score_df = pd.DataFrame(score)

    print(score_df)

# ------------------------- END OF CALCULATION ------------------------- 

# ------------------- SETTING UP DATA FOR THE ALGORITHM --------------------

# CONVERT TO MATRIX

# Expected format: 
# { 
#   location_1 : [0, 1, 0, 0, 0,... 0], 
#   location_2 : [1, 0, 0, 0, 0,... 0], 
#   location_3 : [0, 0, 0, 1, 0,... 0], 
#   location_4 : [0, 0, 0, 0, 0,... 1], 
#   ...
#   location_n : [0, 0, 0, 0, 1,... 0], 
# }

def solution_to_matrix(solution):

    # first creates a matrix of 32 rows and 780 columns filled with zeros
    np_array = np.zeros((LOCAITON_COUNT, NUM_OF_STUDENTS), int)

    matrix = np.asmatrix(np_array)

    fill_matrix(matrix, solution)
    
    return matrix

def fill_matrix(matrix, solution):
    index = 0
    for loc, values in solution.items():
        
        for id in values:
            matrix[index, id - 1] = 1
            
        index += 1

def matrix_to_solution(matrix):
    locations = getLocations()

    result = np.where(matrix == 1)
    coords = list(zip(result[0], result[1] + 1))

    current_loc = list(locations.keys())

    index = 0
    for loc, values in locations.items():
        for i in coords:
            if i[0] == index:
                locations[current_loc[index]] += [i[1]]
        index += 1

    return locations

# ------------------- END OF SETTING UP DATA FOR THE ALGORITHM --------------------

# ------------------------------ CONVERT TO CSV -------------------------------

def dict_to_csv(solution):
    with open('assigned-locations.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in solution.values():
            columns = [c.strip() for c in row.strip(', ').split(',')]
            writer.writerow(columns)

def dict_to_txt(solution):
    with open("locations.txt", 'w') as f: 
        for key, value in solution.items(): 
            f.write('%s:%s\n' % (key, value))

# ------------------------- GENETIC ALGORITHM --------------------------

# GENERATE INITIAL POPULATION
def generate_initial_population(size, df):
    population = {}
    fitness_scores = []

    for i in range(size):
        print(f'[Solution {i + 1}]')
        sol = generate_random_solution(df)
        mat = solution_to_matrix(sol)
        fitness = calculate_fitness(sol, df)
        fitness_scores.append(round(fitness[0], 5))
        population[fitness[0]] = mat

    return population


# SELECTION
def roulette_selection(population):
    keys = population.keys()
    keys_list = [key for key in keys]
    
    total_count = sum(keys_list)
    probabilities = [(value)/total_count for value in keys_list]

    random_number = random.uniform(0, 1)

    cumulative_probability = [probabilities[0]]
    for value in probabilities[1:]:
        cumulative_probability.append(value + cumulative_probability[-1])

    choose = 0
    count = 0
    for value in probabilities:
        choose += value
        count += 1
        if choose >= random_number:
            selected_score = keys_list[probabilities.index(value)]
            return selected_score

def get_average_parent(population):
    keys = population.keys()
    keys_list = [key for key in keys]
    average = np.average(keys_list)
    
    parent = min(keys, key=lambda x:abs(x-average))
    return parent
    
# this is where we get the matrices for each of the selected parents
def select_parents(population):
    print('Selecting Parents....')

    parent_1 = roulette_selection(population)
    parent_2 = get_average_parent(population)

    # if selected parents match
    if parent_1 == parent_2:
        parent_1 = roulette_selection(population)

    return population[parent_1], population[parent_2]


# CROSSOVER
def crossover(parent_1, parent_2):
    print('Performing Crossover....')

    # generate Crosover Mask
    mask = np.random.randint(2, size=NUM_OF_STUDENTS)

    offspring_1 = np.zeros((LOCAITON_COUNT, NUM_OF_STUDENTS), dtype=int)
    offspring_2 = np.zeros((LOCAITON_COUNT, NUM_OF_STUDENTS), dtype=int)

    for loc_index in range(LOCAITON_COUNT):
        for id_index in range(NUM_OF_STUDENTS):
            if mask[id_index] == 1:
                offspring_1[loc_index][id_index] = parent_1[loc_index, id_index]
                offspring_2[loc_index][id_index] = parent_2[loc_index, id_index]

            elif mask[id_index] == 0:
                offspring_1[loc_index][id_index] = parent_2[loc_index, id_index]
                offspring_2[loc_index][id_index] = parent_1[loc_index, id_index]

    return offspring_1, offspring_2


# MUTATION
def mutate(offspring_1, offspring_2, mutation_rate):
    print('Mutating Offspring...')

    if random.uniform(0, 1) < mutation_rate:
        new_offspring_1 = mutate_row(offspring_1)
    else:
        new_offspring_1 = offspring_1
    
    if random.uniform(0, 1) < mutation_rate:
        new_offspring_2 = mutate_row(offspring_2)
    else:
        new_offspring_2 = offspring_2

    return new_offspring_1, new_offspring_2

def mutate_row(offspring):
    row_mutation_rate = 1
    if random.uniform(0, 1) < row_mutation_rate:
        # chooses a random row
        chosen_row = random.randint(0, len(offspring) - 1)

        # finds all the indexes of the 1s in the row
        ones = np.where(offspring[chosen_row] == 1)[0]

        # selects a random index
        chosen_element = ones[random.randint(0, len(ones) - 1)]

        # selects a row to swap to
        chosen_row_2 = random.randint(0, len(offspring) - 1)

        # the off chance where the two randomly selected rows are the same, choose another again
        if chosen_row == chosen_row_2:
            chosen_row_2 = random.randint(0, len(offspring) - 1)

        # swap the two elements
        offspring[chosen_row][chosen_element], offspring[chosen_row_2][chosen_element] = offspring[chosen_row_2][chosen_element], offspring[chosen_row][chosen_element]

    return offspring


# MERGING
def merge(population, offspring_1, offspring_2):
    
    print('Adding Offspring to Population....')
    offspring_1_solution = matrix_to_solution(offspring_1)
    offspring_1_fitness = calculate_fitness(offspring_1_solution, df)

    offspring_2_solution = matrix_to_solution(offspring_2)
    offspring_2_fitness = calculate_fitness(offspring_2_solution, df)

    population[offspring_1_fitness[0]] = offspring_1
    population[offspring_2_fitness[0]] = offspring_2

# Genetic Algorithm Loop
population_size = 20
def perform_generation():
    os.system('cls')
    print('Generating Initial Population....')
    main_population = generate_initial_population(population_size, df)

    best_fitness = 0
    previous_fitness = 0
    threshold = 100.0

    num_of_generations = 0
    max_generations = 5000

    # line plot variables
    x = []
    y = []

    no_improvement_count = 0
    max_no_improvement_count = -1 if max_generations < 250 else (max_generations / 10) if max_generations < 10000 else 1000

    # uncomment the line below if you want to run the program for however long you wish
    # max_no_improvement_count = -1

    mutation_rate = 0.10
    
    start = time.time()
    
    while True:

        num_of_generations += 1
        print(f'\n[Generation {num_of_generations}]\n')
        x.append(num_of_generations)

        # SELECTION
        parent_1, parent_2 = select_parents(main_population)

        # CROSSOVER
        offspring_1, offspring_2 = crossover(parent_1, parent_2)

        # MUTATION
        offspring_1, offspring_2 = mutate(offspring_1, offspring_2, mutation_rate)

        # MERGING
        merge(main_population, offspring_1, offspring_2)
            
        # EVALUATION
        main_population = dict(sorted(main_population.items(), reverse=True))
        print('Killing off the weaklings....')

        if len(main_population) > population_size:
            main_population.popitem()
            main_population.popitem()

        keys = main_population.keys()
        keys_list = [key for key in keys]
        average = np.average(keys_list)

        best_fitness = float(keys_list[0])
        y.append(best_fitness)

        # TERMINATION
        if previous_fitness == best_fitness:
            no_improvement_count += 1
        else:
            previous_fitness = best_fitness
            no_improvement_count = 0

        if no_improvement_count == max_no_improvement_count:
            print('\nNo more improvement in fitness scores. Terminating Loop.\n')
            break

        if num_of_generations == max_generations:
            print('\nMaximum number of generations reached. Terminating Loop.\n')
            break

        if best_fitness > threshold:
            print('\nThreshold reached. Terminating Loop.\n')
            break
                
        print(f'\nCurrent Best Fitness: {best_fitness}')
        print(f'Current Population Average Fitness: {average}\n')

        os.system('cls')

    print(f'Current Best Fitness: {best_fitness}')
    print(f'Current Population Average Fitness: {average}\n')

    end = time.time()
    print(f'TIME ELAPSED: {end - start}\n')

    best_solution_matrix = main_population[best_fitness]
    best_solution = matrix_to_solution(best_solution_matrix)

    print('Best Solution Found:\n')
    results_table(best_solution, df)

    assigned_locations = display_assigned_locations(best_solution, df)
    dict_to_csv(assigned_locations)
    dict_to_txt(best_solution)

    plt.title('Fitness Scores by Generation')
    plt.xlabel('Generations')
    plt.ylabel('Fitness Score')
    plt.plot(x, y)
    plt.show()

perform_generation()


# ------------------------- END OF GENETIC ALGORITHM --------------------------