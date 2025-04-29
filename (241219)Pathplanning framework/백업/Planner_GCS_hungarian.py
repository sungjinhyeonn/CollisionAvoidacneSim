import numpy as np
from deap import creator, base, tools, algorithms
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment

class Planner:
    def __init__(self, agents, tasks, num_agents):
        self.agents = agents
        self.tasks = tasks
        self.num_agents = num_agents

        # DEAP setup
        creator.create("FitnessMax", base.Fitness, weights=(-1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", np.random.randint, 0, self.num_agents, len(self.tasks))
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attribute)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.fitness_func)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutUniformInt, low=0, up=self.num_agents-1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def fitness_func(self, individual):
        assignment = np.array(individual)
        energy_cost = 0
        task_completion = 0

        for i in range(self.num_agents):
            agent_pos = self.agents[i]
            task_indices = np.where(assignment == i)[0]
            task_positions = self.tasks[task_indices]
            distances = np.linalg.norm(task_positions - agent_pos, axis=1)
            energy_cost += np.sum(distances)
            task_completion += len(task_indices)  # Each task completion gives a point

        return (energy_cost, task_completion)

    def initialize_goals(self):
        # Initialize and run the genetic algorithm
        population = self.toolbox.population(n=300)
        NGEN = 40
        for gen in range(NGEN):
            offspring = algorithms.varAnd(population, self.toolbox, cxpb=0.5, mutpb=0.2)
            fits = self.toolbox.map(self.toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            population = self.toolbox.select(offspring, k=len(population))

        best_ind = tools.selBest(population, 1)[0]

        # Allocate tasks based on the best individual found by GA
        self.agent_assignments = {i: self.tasks[np.where(np.array(best_ind) == i)] for i in range(self.num_agents)}
        
        # Solve TSP for each agent's assigned tasks
        for agent in self.agent_assignments:
            if len(self.agent_assignments[agent]) > 1:
                self.agent_assignments[agent] = self.solve_tsp(self.agent_assignments[agent])

    def solve_tsp(self, tasks):
        if len(tasks) < 2:
            return tasks
        dist_matrix = squareform(pdist(tasks))
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        return tasks[col_ind]

# Example usage
agents = np.array([[10, 80], [20, 50], [5, 20], [10, 90], [90, 30]])
tasks = np.array([
    [60, 35], [30, 60], [60, 80], [80, 60], [80, 25],
    [40, 80], [50, 10], [70, 70], [20, 40], [55, 45]
])
planner = Planner(agents, tasks, len(agents))
planner.initialize_goals()
