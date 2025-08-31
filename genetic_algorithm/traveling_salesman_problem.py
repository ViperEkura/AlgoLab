import numpy as np
import matplotlib.pyplot as plt
from math import acos, cos, sin, pi
import random
from typing import Tuple


class TSPProblem:
    def __init__(self, xy: np.ndarray):
        self.xy = xy  # 所有点的坐标（弧度制）
        self.n = len(xy)  # 点的数量（包括起点和终点）
        self.distance_matrix = self._calculate_distance_matrix()
    
    def _calculate_distance_matrix(self) -> np.ndarray:
        """计算所有点之间的球面距离矩阵"""
        d = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i+1, self.n):
                # 使用球面距离公式计算两点间距离
                d[i, j] = 6370 * acos(cos(self.xy[i, 0]-self.xy[j, 0]) * cos(self.xy[i, 1]) * 
                            cos(self.xy[j, 1]) + sin(self.xy[i, 1]) * sin(self.xy[j, 1]))
        return d + d.T  # 使距离矩阵对称
    
    def evaluate(self, individual: np.ndarray) -> float:
        """评估路径的总距离（确保形成闭环）"""
        total_distance = 0.0
        # 计算路径上相邻点之间的距离
        for i in range(len(individual) - 1):
            total_distance += self.distance_matrix[individual[i], individual[i+1]]
        # 添加从终点回到起点的距离，形成闭环
        total_distance += self.distance_matrix[individual[-1], individual[0]]
        return total_distance

class GeneticAlgorithm:
    def __init__(
        self, 
        problem: TSPProblem,
        population_size: int = 50,
        max_generations: int = 100,
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
        """使用改良圈算法初始化种群"""
        self.population = []
        for _ in range(self.population_size):
            # 生成1-100的随机排列
            c = random.sample(range(1, self.problem.n - 1), self.problem.n - 2)
            individual = [0] + c + [self.problem.n - 1]  # 构建初始解（添加起点终点）
            
            # 改良圈算法优化路径
            for _ in range(self.problem.n):  # 最大优化次数
                flag = 0  # 优化标志
                for m in range(self.problem.n - 2):
                    for n in range(m + 2, self.problem.n - 1):
                        # 如果交换路径段能缩短总距离
                        if (self.problem.distance_matrix[individual[m], individual[n]] + 
                            self.problem.distance_matrix[individual[m+1], individual[n+1]] < 
                            self.problem.distance_matrix[individual[m], individual[m+1]] + 
                            self.problem.distance_matrix[individual[n], individual[n+1]]):
                            # 执行路径段交换(翻转链接)
                            individual[m+1:n+1] = individual[n:m:-1]
                            flag = 1  # 标记已优化
                if flag == 0:  # 如果没有优化则退出
                    break
            
            self.population.append(np.array(individual))

    def evaluate_population(self):
        """评估整个种群的适应度"""
        self.fitnesses = [self.problem.evaluate(ind) for ind in self.population]

    def select(self) -> np.ndarray:
        """锦标赛选择"""
        k = 3  # 锦标赛规模
        selected = random.choices(range(self.population_size), k=k)
        selected_fitness = [self.fitnesses[i] for i in selected]
        winner_idx = selected[np.argmin(selected_fitness)]  # 最小化问题
        return self.population[winner_idx].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """顺序交叉（OX）适用于TSP"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # 选择交叉点
        point1 = random.randint(1, len(parent1) - 3)
        point2 = random.randint(point1 + 1, len(parent1) - 2)
        
        # 创建子代
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        # 复制中间段
        child1[point1:point2+1] = parent1[point1:point2+1]
        child2[point1:point2+1] = parent2[point1:point2+1]
        
        # 填充剩余部分
        self._fill_remaining(child1, parent2, point1, point2)
        self._fill_remaining(child2, parent1, point1, point2)
        
        return child1, child2
    
    def _fill_remaining(self, child, parent, point1, point2):
        """填充顺序交叉的剩余部分"""
        current_pos = (point2 + 1) % len(child)
        parent_pos = (point2 + 1) % len(parent)
        
        while current_pos != point1:
            if parent[parent_pos] not in child[point1:point2+1]:
                child[current_pos] = parent[parent_pos]
                current_pos = (current_pos + 1) % len(child)
            parent_pos = (parent_pos + 1) % len(parent)

    def mutate(self, individual: np.ndarray):
        """交换变异：随机选择两个位置交换"""
        if random.random() < self.mutation_rate:
            # 避免交换起点和终点
            i = random.randint(1, len(individual) - 2)
            j = random.randint(1, len(individual) - 2)
            individual[i], individual[j] = individual[j], individual[i]

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
            if gen % 10 == 0:
                print(f"Generation {gen}: Best Distance = {best_fitness:.2f}")

            new_population = []
            # 保留精英
            elite_size = int(0.1 * self.population_size)
            elite_indices = np.argsort(self.fitnesses)[:elite_size]
            new_population.extend([self.population[i].copy() for i in elite_indices])
            
            # 生成新个体
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


def load_data():
    """硬编码数据并处理坐标"""

    data = np.array([
        [53.7121, 15.3046, 51.1758, 0.0322, 46.3253, 28.2753, 30.3313, 6.9348],
        [56.5432, 21.4188, 10.8198, 16.2529, 22.7891, 23.1045, 10.1584, 12.4819],
        [20.1050, 15.4562, 1.9451, 0.2057, 26.4951, 22.1221, 31.4847, 8.9640],
        [26.2418, 18.1760, 44.0356, 13.5401, 28.9836, 25.9879, 38.4722, 20.1731],
        [28.2694, 29.0011, 32.1910, 5.8699, 36.4863, 29.7284, 0.9718, 28.1477],
        [8.9586, 24.6635, 16.5618, 23.6143, 10.5597, 15.1178, 50.2111, 10.2944],
        [8.1519, 9.5325, 22.1075, 18.5569, 0.1215, 18.8726, 48.2077, 16.8889],
        [31.9499, 17.6309, 0.7732, 0.4656, 47.4134, 23.7783, 41.8671, 3.5667],
        [43.5474, 3.9061, 53.3524, 26.7256, 30.8165, 13.4595, 27.7133, 5.0706],
        [23.9222, 7.6306, 51.9612, 22.8511, 12.7938, 15.7307, 4.9568, 8.3669],
        [21.5051, 24.0909, 15.2548, 27.2111, 6.2070, 5.1442, 49.2430, 16.7044],
        [17.1168, 20.0354, 34.1688, 22.7571, 9.4402, 3.9200, 11.5812, 14.5677],
        [52.1181, 0.4088, 9.5559, 11.4219, 24.4509, 6.5634, 26.7213, 28.5667],
        [37.5848, 16.8474, 35.6619, 9.9333, 24.4654, 3.1644, 0.7775, 6.9576],
        [14.4703, 13.6368, 19.8660, 15.1224, 3.1616, 4.2428, 18.5245, 14.3598],
        [58.6849, 27.1485, 39.5168, 16.9371, 56.5089, 13.7090, 52.5211, 15.7957],
        [38.4300, 8.4648, 51.8181, 23.0159, 8.9983, 23.6440, 50.1156, 23.7816],
        [13.7909, 1.9510, 34.0574, 23.3960, 23.0624, 8.4319, 19.9857, 5.7902],
        [40.8801, 14.2978, 58.8289, 14.5229, 18.6635, 6.7436, 52.8423, 27.2880],
        [39.9494, 29.5114, 47.5099, 24.0664, 10.1121, 27.2662, 28.7812, 27.6659],
        [8.0831, 27.6705, 9.1556, 14.1304, 53.7989, 0.2199, 33.6490, 0.3980],
        [1.3496, 16.8359, 49.9816, 6.0828, 19.3635, 17.6622, 36.9545, 23.0265],
        [15.7320, 19.5697, 11.5118, 17.3884, 44.0398, 16.2635, 39.7139, 28.4203],
        [6.9909, 23.1804, 38.3392, 19.9950, 24.6543, 19.6057, 36.9980, 24.3992],
        [4.1591, 3.1853, 40.1400, 20.3030, 23.9876, 9.4030, 41.1084, 27.7149]
    ])

    x = data[:, 0:8:2].flatten()  # 提取所有x坐标（经度）
    y = data[:, 1:8:2].flatten()  # 提取所有y坐标（纬度）
    sj = np.column_stack((x, y))
    d1 = [70, 40]  # 基地坐标（起点/终点）
    xy = np.vstack((d1, sj, d1))  # 组合完整坐标
    return xy * pi / 180  # 将角度转换为弧度


def plot_path(xy, path):
    """绘制路径图"""
    xx = xy[path, 0]  # 路径经度序列
    yy = xy[path, 1]  # 路径纬度序列
    plt.plot(xx, yy, '-o')  # 绘制带标记点的路径线
    plt.show()


if __name__ == "__main__":
    xy = load_data()
    
    problem = TSPProblem(xy)
    ga = GeneticAlgorithm(
        problem=problem,
        population_size=50,
        max_generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    best_path, best_distance = ga.run()
    
    print(f"\n最优路径: {best_path}")
    print(f"最短距离: {best_distance:.2f} km")
    plot_path(xy, best_path)