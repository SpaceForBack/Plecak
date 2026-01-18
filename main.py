import os
import random
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

# MODEL 

class Item:
    def __init__(self, item_id: int, value: int, weight: int):
        self.item_id = item_id
        self.value = value
        self.weight = weight

    def __repr__(self):
        return f"Item(id={self.item_id}, v={self.value}, w={self.weight})"

class KnapsackProblem:
    def __init__(self):
        self.items: List[Item] = []
        self.capacity: int = 0
        self.num_items: int = 0
        self.optimal_value: Optional[int] = None

    def load_problem(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Nie znaleziono pliku: {filepath}")
        with open(filepath, 'r') as f:
            lines = f.readlines()
        header = lines[0].strip().split()
        self.num_items = int(header[0])
        self.capacity = int(header[1])
        self.items = []
        for i, line in enumerate(lines[1:]):
            parts = line.strip().split()
            if len(parts) >= 2:
                val = int(parts[0])
                w = int(parts[1])
                self.items.append(Item(i, val, w))

    def load_optimum(self, filepath: str):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                content = f.read().strip()
                if content:
                    self.optimal_value = int(content)

class Individual:
    """Reprezentuje jedno rozwiązanie."""
    def __init__(self, num_genes: int):
        # Średnio losujemy tylko 2-3 przedmioty na start, niezależnie od wielkości problemu.
        # To gwarantuje, że nie przeładujemy plecaka na początku.
        prob = 2.5 / max(num_genes, 1)
        self.chromosome = [1 if random.random() < prob else 0 for _ in range(num_genes)]
        self.fitness = 0.0
        self.total_weight = 0

    def calculate_fitness(self, problem: KnapsackProblem):
        weight_sum = 0
        value_sum = 0
        for i, gene in enumerate(self.chromosome):
            if gene == 1:
                item = problem.items[i]
                weight_sum += item.weight
                value_sum += item.value
        self.total_weight = weight_sum
        
        if weight_sum <= problem.capacity:
            self.fitness = float(value_sum)
        else:
            self.fitness = 0.0 # Kara śmierci

# SILNIK GENETYCZNY

class GeneticAlgorithm:
    def __init__(self, problem: KnapsackProblem, population_size=50, 
                 generations=100, crossover_prob=0.8, mutation_prob=0.01,
                 selection_method="roulette", crossover_method="one_point"):
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.population: List[Individual] = []
        self.best_solution: Optional[Individual] = None
        self.fitness_history = []

    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            ind = Individual(self.problem.num_items)
            ind.calculate_fitness(self.problem)
            self.population.append(ind)
        self.best_solution = max(self.population, key=lambda x: x.fitness)

    def select_parent_roulette(self) -> Individual:
        total_fitness = sum(ind.fitness for ind in self.population)
        if total_fitness == 0: return random.choice(self.population)
        pick = random.uniform(0, total_fitness)
        current = 0
        for ind in self.population:
            current += ind.fitness
            if current > pick: return ind
        return self.population[-1]

    def select_parent_tournament(self, tournament_size=3) -> Individual:
        candidates = random.sample(self.population, tournament_size)
        return max(candidates, key=lambda x: x.fitness)

    def select_parent(self) -> Individual:
        if self.selection_method == "tournament": return self.select_parent_tournament()
        else: return self.select_parent_roulette()

    def crossover_one_point(self, p1: Individual, p2: Individual):
        point = random.randint(1, self.problem.num_items - 1)
        c1, c2 = Individual(self.problem.num_items), Individual(self.problem.num_items)
        c1.chromosome = p1.chromosome[:point] + p2.chromosome[point:]
        c2.chromosome = p2.chromosome[:point] + p1.chromosome[point:]
        return c1, c2

    def crossover_two_point(self, p1: Individual, p2: Individual):
        if self.problem.num_items < 3: return self.crossover_one_point(p1, p2)
        pt1 = random.randint(1, self.problem.num_items - 2)
        pt2 = random.randint(pt1 + 1, self.problem.num_items - 1)
        c1, c2 = Individual(self.problem.num_items), Individual(self.problem.num_items)
        c1.chromosome = p1.chromosome[:pt1] + p2.chromosome[pt1:pt2] + p1.chromosome[pt2:]
        c2.chromosome = p2.chromosome[:pt1] + p1.chromosome[pt1:pt2] + p2.chromosome[pt2:]
        return c1, c2

    def crossover(self, p1: Individual, p2: Individual):
        if random.random() > self.crossover_prob:
            c1, c2 = Individual(self.problem.num_items), Individual(self.problem.num_items)
            c1.chromosome, c2.chromosome = p1.chromosome[:], p2.chromosome[:]
            return c1, c2
        if self.crossover_method == "two_point": return self.crossover_two_point(p1, p2)
        else: return self.crossover_one_point(p1, p2)

    def mutate(self, ind: Individual):
        for i in range(len(ind.chromosome)):
            if random.random() < self.mutation_prob:
                ind.chromosome[i] = 1 - ind.chromosome[i]

    def run(self) -> Individual:
        self.initialize_population()
        for _ in range(self.generations):
            new_population = []
            current_best = max(self.population, key=lambda x: x.fitness)
            elite = Individual(self.problem.num_items)
            elite.chromosome = current_best.chromosome[:]
            elite.calculate_fitness(self.problem)
            new_population.append(elite)
            
            while len(new_population) < self.population_size:
                p1 = self.select_parent()
                p2 = self.select_parent()
                c1, c2 = self.crossover(p1, p2)
                self.mutate(c1)
                self.mutate(c2)
                c1.calculate_fitness(self.problem)
                c2.calculate_fitness(self.problem)
                if len(new_population) < self.population_size: new_population.append(c1)
                if len(new_population) < self.population_size: new_population.append(c2)
            
            self.population = new_population
            global_best = max(self.population, key=lambda x: x.fitness)
            self.fitness_history.append(global_best.fitness)
            if global_best.fitness > self.best_solution.fitness:
                self.best_solution = global_best
        return self.best_solution

# 3. AUTOMATYZACJA

if __name__ == "__main__":
    base_folder = "C:/Users/SpaceOn/Desktop/Studia Projekty itp/Algorytmy/Plecakowy/dane AG"
    output_folder = "wyniki_eksperymentu"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    tasks = [
        ("low-dimensional", "f1_l-d_kp_10_269"),
        ("low-dimensional", "f3_l-d_kp_4_20"),
        ("low-dimensional", "f8_l-d_kp_23_10000"),
        ("low-dimensional", "f10_l-d_kp_20_879"),
        ("large_scale", "knapPI_1_100_1000_1"),
        ("large_scale", "knapPI_1_200_1000_1") 
    ]

    print(f"--- START AUTOMATU ---")
    for idx, (category, filename) in enumerate(tasks):
        print(f"\n[{idx+1}/{len(tasks)}] Przetwarzanie: {filename} ...")
        
        problem_path = os.path.join(base_folder, category, filename)
        optimum_path = os.path.join(base_folder, category + "-optimum", filename)

        try:
            kp = KnapsackProblem()
            kp.load_problem(problem_path)
            kp.load_optimum(optimum_path)

            pop_size = 100 if category == "large_scale" else 50
            gens = 200 if category == "large_scale" else 100
            
            # Dynamiczna mutacja (1 gen na raz)
            # Jeśli mamy 100 przedmiotów -> mutacja 0.01. Jeśli 20 -> 0.05.
            dynamic_mutation = 1.0 / kp.num_items

            # CLASSIC
            ga_classic = GeneticAlgorithm(kp, pop_size, gens, 0.9, dynamic_mutation, "roulette", "one_point")
            ga_classic.run()

            # PRO
            ga_pro = GeneticAlgorithm(kp, pop_size, gens, 0.9, dynamic_mutation, "tournament", "two_point")
            ga_pro.run()

            plt.figure(figsize=(10, 6))
            plt.plot(ga_classic.fitness_history, label='Classic (Ruletka)', linestyle='--', color='blue', alpha=0.6)
            plt.plot(ga_pro.fitness_history, label='Pro (Turniej)', linewidth=2, color='green')
            
            if kp.optimal_value:
                plt.axhline(y=kp.optimal_value, color='red', linestyle=':', label=f'Optimum ({kp.optimal_value})')

            plt.title(f"Porównanie: {filename}")
            plt.xlabel("Generacja")
            plt.ylabel("Fitness")
            plt.legend()
            plt.grid(True, alpha=0.3)

            output_path = os.path.join(output_folder, f"Wykres_{filename}.png")
            plt.savefig(output_path)
            plt.close()
            print(f"   -> OK: {output_path}")

        except Exception as e:
            print(f"   [!] Błąd: {e}")

    print("\n--- ZAKOŃCZONO ---")