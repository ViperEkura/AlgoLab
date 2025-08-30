import numpy as np
import random
from typing import Tuple


class Problem:
    def __init__(
        self, 
        cost_matrix: np.ndarray,
        supply: np.ndarray,
        capacity: np.ndarray,
    ):
        self.cost_matrix = cost_matrix  # shape: (n_suppliers, n_carriers)
        self.supply = supply            # shape: (n_suppliers,)
        self.capacity = capacity        # shape: (n_carriers,)
        self.n_suppliers = len(supply)
        self.n_carriers = len(capacity)

    def evaluate(self, individual: np.ndarray) -> float:
        total_cost = 0.0
        carrier_load = np.zeros(self.n_carriers)

        for i, j in enumerate(individual):
            total_cost += self.cost_matrix[i, j] * self.supply[i]
            carrier_load[j] += self.supply[i]

        overload = np.sum(np.maximum(carrier_load - self.capacity, 0)) * 1000
        return total_cost + overload

        


class GeneticAlgorithm:
    def __init__(
        self, 
        problem: Problem,
        population_size: int = 100,
        max_generations: int = 200,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8
    ):
        self.problem = problem
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.population = []
        self.fitnesses = []


    def initialize_population(self):
        """随机生成初始种群"""
        self.population = []
        for _ in range(self.population_size):
            # 每个供应商随机分配一个转运商
            individual = np.random.randint(0, self.problem.n_carriers, size=self.problem.n_suppliers)
            self.population.append(individual)

    def evaluate_population(self):
        """评估整个种群的适应度"""
        self.fitnesses = [self.problem.evaluate(ind) for ind in self.population]

    def select(self) -> np.ndarray:
        """锦标赛选择（Tournament Selection），推荐用于组合优化"""
        k = 3  # 锦标赛规模
        selected = random.choices(range(self.population_size), k=k)
        selected_fitness = [self.fitnesses[i] for i in selected]
        winner_idx = selected[np.argmin(selected_fitness)]  # 最小化问题
        return self.population[winner_idx].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """单点交叉"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def mutate(self, individual: np.ndarray):
        """随机突变：以概率 mutation_rate 改变某个供应商的分配"""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = random.randint(0, self.problem.n_carriers - 1)

    def run(self) -> Tuple[np.ndarray, float]:
        """运行遗传算法"""
        self.initialize_population()
        best_individual = None
        best_fitness = float('inf')

        for gen in range(self.max_generations):
            self.evaluate_population()
            
            # 找出当前最优
            min_fitness_idx = np.argmin(self.fitnesses)
            if self.fitnesses[min_fitness_idx] < best_fitness:
                best_fitness = self.fitnesses[min_fitness_idx]
                best_individual = self.population[min_fitness_idx].copy()

            # 输出进度
            if gen % 1 == 0:
                print(f"Generation {gen}: Best Cost = {best_fitness:.2f}")

            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self.select()
                parent2 = self.select()
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            self.population = new_population

        return best_individual, best_fitness
    
if __name__ == '__main__':
    np.random.seed(42)
    n_suppliers = 10
    n_carriers = 3

    cost_matrix = np.random.rand(n_suppliers, n_carriers) * 10         # 随机成本
    supply = np.random.randint(5, 15, size=n_suppliers)                # 供应量
    capacity = np.array([80, 70, 60])                                  # 转运商容量

    problem = Problem(cost_matrix, supply, capacity)
    ga = GeneticAlgorithm(problem, population_size=100, max_generations=20, mutation_rate=0.5)

    best_solution, best_cost = ga.run()

    print("\n--- 最优分配方案 ---")
    for i, carrier in enumerate(best_solution):
        print(f"供应商 {i} → 转运商 {carrier} (供货量: {supply[i]})")

    print(f"\n总成本: {best_cost:.2f}")