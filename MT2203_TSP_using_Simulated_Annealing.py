import numpy as np
import random
import math
import matplotlib.pyplot as plt
import random


# Function to Calulate Distance of the Route

def calculate_total_distance(solution, distance_matrix):
    total_distance = 0
    num_cities = len(solution)
    
    for i in range(num_cities - 1):
        total_distance += distance_matrix[solution[i]][solution[i + 1]]
    total_distance += distance_matrix[solution[-1]][solution[0]]  
    
    return total_distance


# Function to Generate New Path either by Reversing the section or by shifting the section

def generate_neighbor_solution(current_solution):
    choices = ['reverse', 'shift']
    method_to_change_path = random.choice(choices)

    if method_to_change_path == 'reverse':
        # Swap two random cities to generate a neighbor solution
        neighbor_solution = current_solution.copy()
        i, j = sorted(random.sample(range(len(current_solution)), 2))
        neighbor_solution[i:j+1] = reversed(neighbor_solution[i:j+1])
    
        return neighbor_solution

    elif method_to_change_path == 'shift':
        neighbor_solution = current_solution.copy()
        i, j = sorted(random.sample(range(len(current_solution)), 2))
        cut_section = neighbor_solution[i:j]
        new_soln = neighbor_solution[:i]+ neighbor_solution[j:] 
        k = random.sample(range(len(new_soln)),1)[0]
        neighbor_solution = new_soln[:k]+ cut_section + new_soln[k:]
        return neighbor_solution


# Function to calculate optimum route using Initail Temperature, and 100*N iterations or 10*N scuessfull iterations. 

def simulated_annealing_TSP(distance_matrix, initial_solution, initial_temperature, cooling_rate, stopping_condition):
    # Initialize current and best solutions
    current_solution = initial_solution
    best_solution = initial_solution

    # Calculate the total distance for the initial solution
    current_distance = calculate_total_distance(current_solution, distance_matrix)
    best_distance = current_distance
    current_temperature = initial_temperature
    # Initialize list to keep track of distance history
    dist = [current_distance]

    # Main annealing loop: continue until temperature drops below a certain threshold
    while current_temperature > 5:
        stopping_condition = False
        iterations = 0
        sucessfull_iterations = 0
        
        # Inner loop: explore neighboring solutions until stopping condition is met
        while not stopping_condition:
            new_solution = generate_neighbor_solution(current_solution)
            new_distance = calculate_total_distance(new_solution, distance_matrix)
            
            distance_difference = new_distance - current_distance

            # If the new solution is better (shorter distance), accept it
            if distance_difference < 0:
                current_solution = new_solution
                current_distance = new_distance
                dist.append(current_distance)
                iterations += 1
                sucessfull_iterations += 1
                if new_distance < best_distance:
                    best_solution = new_solution
                    best_distance = new_distance
            
            # If the new solution is worse, accept it with a probability -(E1 - E0)/kT   k=1
            else:
                acceptance_probability = math.exp(-distance_difference / current_temperature)
                if random.random() < acceptance_probability:
                    current_solution = new_solution
                    current_distance = new_distance
                    dist.append(current_distance)
                    iterations += 1
    
            # Cut-off conditions to stop the inner loop
            if sucessfull_iterations > 10*num_cities:
                stopping_condition = True
            elif iterations > 100*num_cities:
                stopping_condition = True
                
        # Reduce the temperature for the next iteration    
        current_temperature *= cooling_rate
        
            
    
    return best_solution, best_distance , dist


# Set number of nodes
num_cities = 50
random.seed(42)

#Generate a random distance matrix (replace this with your actual data)
distance_matrix = np.random.randint(1, 100, size=(num_cities, num_cities))


# distance_matrix = [[29 ,81, 75, 46, 10, 94, 26, 43, 35,  5],
#  [96 ,38, 29, 62, 90,  2, 17, 82, 99, 67],
#  [86 ,10 ,75, 69, 29, 53, 44, 67, 10, 33],
#  [91 ,76, 72, 46, 71, 82, 28,  2, 91, 92],
#  [75 ,38, 91, 58, 34, 58, 23,  5, 69, 53],
#  [68 ,77, 31, 12, 14 ,10 ,13, 19, 28 ,93],
#  [84, 37, 55, 63, 80, 95, 57, 92, 47, 82],
#  [64 ,90, 42, 44, 52, 41, 22, 31, 37,  9],
#  [37 ,58, 40,  2,  2, 73, 14, 14, 99, 37],
#  [36, 77, 43, 53 ,21 ,64 ,64 ,84, 71 ,56]]


# Setting Initial Conditons and Printing results
initial_solution = list(range(num_cities))  
initial_temperature = 1000
cooling_rate = 0.95
stopping_condition = False

best_solution, best_distance , dist = simulated_annealing_TSP(distance_matrix, initial_solution, initial_temperature, cooling_rate, stopping_condition)

print("Best solution:", best_solution)
print("Best distance:", best_distance)
print('Total iterations:',len(dist))


plt.plot(dist)
plt.title('Iteration vs Loss Function')
plt.show()
