import random
import numpy as np
import pandas as pd


# day_data = pd.read_csv('/Users/jojeonghyeon/Documents/WorkSpace/PYTHON/chaejugchwah/output.csv')
day_data = pd.read_csv('./chaejugchwah/output.csv')
nurse_data = pd.read_csv('./chaejugchwah/nurse_schedule.csv')
bf = []
# 간호사 스케줄링 문제를 해결하는 Genetic Algorithm 클래스
class NurseSchedulingGA:
    
    
    
    def __init__(self, num_nurses, num_days, mutation_rate=0.1, population_size=30):
        self.num_nurses = num_nurses
        self.num_days = num_days
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.population = []

    def initialize_population(self):
        for _ in range(self.population_size):
            schedule = np.random.randint(2, size=(self.num_nurses, self.num_days))
            self.population.append(schedule)

    def fitness(self, schedule, min_nightshift = 2):
        # 초기화
        total_fitness = 0
        # (각 날짜의 선호도) * (근무하면 1 아님 0)
        result = nurse_data.iloc[:,1:22].mul(schedule) ## 임의값
        total_fitness += result.values.sum()
        # 각 간호사 끼리의 선호도 고려 -> 단순곱
        df_schedule = pd.DataFrame(schedule)
        for i in range(20): ## 임의값
            work_nurse_day = df_schedule.iloc[:,i]
            pref = work_nurse_day.mul(nurse_data.iloc[:,22+i]) ## 임의값
            total_fitness += pref.values.sum()
            work_nurse_day.values.tolist()
            
    # 각 간호사에 대해 평가 패널티요소들
        for nurse_schedule in schedule:
            # 쉬프트 수에 따른 패널티 요소
            first_shift = 0
            second_shift = 0
            third_shift = 0
            work_day = [index for index, value in enumerate(nurse_schedule) if value == 1]
            for i in work_day:
                if i // 3 == 0:
                    first_shift += 1
                elif i//3 == 1:
                    second_shift += 1
                else:
                    third_shift += 1
            if third_shift < min_nightshift:
                total_fitness -= 30
            # 16시간 휴식 확인 이후 패널티
            # max_daily_hours = 8  # 하루 최대 근무 시간
            # min_rest_time = 2
            # daily_hours = np.sum(nurse_schedule)
            # if daily_hours > max_daily_hours:
            #     total_fitness -= (daily_hours - max_daily_hours)
            
            # 21shift중 야간 쉬프트 최소 2회
            
            

        return total_fitness
        # 이 부분에서 실제로는 스케줄링의 평가 지표를 정의해야 합니다.
        # 간호사들의 근무 조건, 휴가, 교대근무 등을 고려하여 평가합니다.
        

    def crossover(self, parent1, parent2):
        # 교차 연산을 수행하는 부분입니다.
        crossover_point = random.randint(1, self.num_days - 1)
        child1 = np.hstack((parent1[:, :crossover_point], parent2[:, crossover_point:]))
        child2 = np.hstack((parent2[:, :crossover_point], parent1[:, crossover_point:]))
        return child1, child2

    def mutate(self, schedule):
        # 돌연변이 연산을 수행하는 부분입니다.
        for i in range(self.num_nurses):
            for j in range(self.num_days):
                if random.uniform(0, 1) < self.mutation_rate:
                    schedule[i, j] = 1 - schedule[i, j]
        return schedule

    # def evolve(self, generations):
    #     self.initialize_population()

    #     for generation in range(generations):
    #         # 평가 및 선택
    #         fitness_scores = [self.fitness(schedule) for schedule in self.population]
    #         selected_indices = np.argsort(fitness_scores)[-self.population_size // 2:]
    #         selected_population = [self.population[i] for i in selected_indices]

    #         # 교차
    #         new_population = []
    #         for _ in range(self.population_size // 2):
    #             parent1, parent2 = random.choices(selected_population, k=2)
    #             child1, child2 = self.crossover(parent1, parent2)
    #             new_population.extend([child1, child2])

    #         # 돌연변이
    #         new_population = [self.mutate(schedule) for schedule in new_population]

    #         # 업데이트
    #         self.population = selected_population + new_population

    #         # 출력
    #         # best_5per = self.population[np.argsort(fitness_scores)]
    #         best_schedule = self.population[np.argmax(fitness_scores)]
    #         best_fitness = self.fitness(best_schedule)
    #         bf.append(best_fitness)
    #         print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")
        
    def evolve(self, generations, elitism_rate=0.05):
        self.initialize_population()

        for generation in range(generations):
            # 평가 및 선택
            fitness_scores = [self.fitness(schedule) for schedule in self.population]
            selected_indices = np.argsort(fitness_scores)[-self.population_size // 2:]
            selected_population = [self.population[i] for i in selected_indices]

            # 엘리트 선택
            num_elite = int(self.population_size * elitism_rate)
            elite_indices = np.argsort(fitness_scores)[-num_elite:]
            elite_population = [self.population[i] for i in elite_indices]

            # 교차
            new_population = []
            for _ in range(self.population_size // 2 - num_elite):
                parent1, parent2 = random.choices(selected_population, k=2)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([child1, child2])

            # 돌연변이
            new_population = [self.mutate(schedule) for schedule in new_population]

            # 엘리트 추가
            new_population.extend(elite_population)

            # 업데이트
            self.population = new_population

            # 출력
            best_schedule = max(self.population, key=lambda x: self.fitness(x))
            best_fitness = self.fitness(best_schedule)
            bf.append(best_fitness)
            print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

        # print(best_schedule)


# 예제로 간호사 스케줄링 문제 해결
num_nurses = 20
num_days = 21
ga = NurseSchedulingGA(num_nurses, num_days)
ga.evolve(generations=200)
print(bf)
