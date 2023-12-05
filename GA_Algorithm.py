import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
# from sklearn.preprocessing import MinMaxScaler



day_data = pd.read_csv('./chaejugchwah/output.csv')
nurse_data = pd.read_csv('./chaejugchwah/nurse_schedule.csv')
bf = []
# 간호사 스케줄링 문제를 해결하는 Genetic Algorithm 클래스
best_schedule_result = []
class NurseSchedulingGA:
    def __init__(self, num_nurses, num_days, mutation_rate=0.1, population_size=5):
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
        # print(schedule.shape)
        result = nurse_data.iloc[:,1:1+num_days].mul(schedule)
        # print(result.iloc[1,:])
        # print(result.shape)
        total_fitness += result.values.sum()
        # 각 간호사 끼리의 선호도 고려 -> 단순곱
        df_schedule = pd.DataFrame(schedule)
        ## 이부분 해결필요 Degree 사용
        pref_list = []
        nurse_degree = nurse_data.loc[:,'Degree']
        for i in range(num_nurses): ## 임의값
            work_nurse_day = df_schedule.iloc[:,i]
            # print(work_nurse_day)
            pref = work_nurse_day.mul(nurse_data.iloc[:,1+num_days+i]) ## 임의값
            asdf = pref.tolist()
            # print(pref.shape)
            pref_list.append(asdf)
            total_fitness += pref.values.sum()
            # work_nurse_day_list = work_nurse_day.values.tolist()
        # print(len(pref_list[0]))
        
        # for i in range(num_nurses):
        #     for j in range(num_days):
        #         total_fitness += (pd.DataFrame(pref_list)*result.iloc[i,j]).sum()

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
        # Degree 평가요소들
        
        return total_fitness
        # 이 부분에서 실제로는 스케줄링의 평가 지표를 정의해야 합니다.
        # 간호사들의 근무 조건, 휴가, 교대근무 등을 고려하여 평가합니다.
        
    def roulette_wheel_selection(self, fitness_values):
        # 루레-또 루르 입니다. 수정중입니다~
        total_fitness = sum(fitness_values)
        selection_probabilities = [fitness / total_fitness for fitness in fitness_values]
        # df_selection_probabilities = pd.DataFrame(selection_probabilities)
        # print(df_selection_probabilities)
        # scaler = MinMaxScaler()
        # a= scaler.fit_transform(df_selection_probabilities)
        # print(a)
        # print(a.tolist())
        # print(selection_probabilities)
        selected_index = np.random.choice(len(fitness_values), p=selection_probabilities)
        # print("룰렛룰 적용 테스트" + str(selected_index))
        return selected_index
    
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

    def evolve(self, generations, elitism_rate=0.05):
        self.initialize_population()

        for generation in range(generations):
            # print("{} 번째 제네레이션 시작".format(generation+1))
            # print(len(self.population))
            # 평가 및 선택
            fitness_scores = [self.fitness(schedule) for schedule in self.population]
            # print(fitness_scores)
            # print(np.argsort(fitness_scores))
            # selected_indices = np.argsort(fitness_scores)[-self.population_size // 2:] ## 이쪽 수정이 필요할듯 룰렛룰로 선택하도록 현재는 50% 이상인 것들을 선택해서 가져오는 방식을 취하는중 현재 룰렛룰로 대체완료
            # print(selected_indices)
            # selected_population = [self.population[i] for i in selected_indices]
            selected_population = []
            
            # print(len(selected_population))

            # Roulette wheel selection
            for _ in range(self.population_size):
                selected_index = self.roulette_wheel_selection(fitness_scores)
                selected_population.append(self.population[selected_index])
            # print("룰렛룰")
            # print(len(self.population))

            # 엘리트 선택 이쪽부분에서 population size가 커지는 문제점 발견됨. 수정 필요함 후에 100개로 조정하는 식으로 가져가도 괜춘할듯 ㅇㅇ 뭐여 수정했는데 95개로 작아짐;
            num_elite = int(self.population_size * elitism_rate)
            elite_indices = np.argsort(fitness_scores)[-num_elite:]
            elite_population = [self.population[i] for i in elite_indices]



            # 교차 이파트에서 부모와의 비교가 필요할듯 근데 갠취는 부모를 완전히 빼버리는게 취향
            new_population = []
            # for _ in range(self.population_size // 2 - num_elite):
            for _ in range((self.population_size-num_elite) // 2): ## 이렇게 만들었더니 홀수개일때 1개 줄어버리는 문제가 생김 이에 대한 질문이 필요
                parent1, parent2 = random.choices(selected_population, k=2)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([child1, child2])
            # print("Cross이후")
            # print(len(self.population))
            # print("Cross이후 new_pop길이")
            # print(len(new_population))

            # 돌연변이
            new_population = [self.mutate(schedule) for schedule in new_population]
            # print("Mutation이후")
            # print(len(new_population))

            # 엘리트 추가
            
            new_population.extend(elite_population)
            # print("Elite 이후")
            # print(len(new_population))
            
            # 같은 Population size로 맞춰주기
            # print(new_population)
            
            # 업데이트
            # print(len(new_population))
            # print(len(self.population))
            self.population = new_population

            # 출력
            best_schedule = max(self.population, key=lambda x: self.fitness(x))
            best_fitness = self.fitness(best_schedule)
            bf.append(-best_fitness)
            print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")
            plt.cla()
            plt.plot(bf)
            plt.pause(0.1)
        print("베스트 스케쥴입니다!")
        print(best_schedule)


# 예제로 간호사 스케줄링 문제 해결
num_nurses = 40
num_days = 120
plt.show()
ga = NurseSchedulingGA(num_nurses, num_days)
ga.evolve(generations=3)
plt.show()
print("베스트 스코어 리스트입니다! \n{}".format(bf))
# print("베스트 스코어를 달성한 거 \n{}".format(best_schedule))
# df_bf = pd.DataFrame(bf)
# scaler = MinMaxScaler()
# b =scaler.fit_transform(df_bf)
# print(b)
# plt.plot(bf)
# plt.show()