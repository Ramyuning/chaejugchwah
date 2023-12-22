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
    def __init__(self, num_nurses, num_days, mutation_rate=0.1, population_size=10):
        self.num_nurses = num_nurses
        self.num_days = num_days
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.population = []

    def initialize_population(self):
        for _ in range(self.population_size):
            while True:
                schedule = np.random.randint(2, size=(self.num_nurses, self.num_days))
                
                # Check hard constraints
                if not self.check_nurse_constraints(schedule) or not self.check_day_constraints(schedule):
                    continue  # If constraints are not met, regenerate the schedule
                else:
                    break  # If constraints are met, exit the loop

            self.population.append(schedule)
            # schedule = np.random.randint(2, size=(self.num_nurses, self.num_days))
            # self.population.append(schedule)
    #하드 제약조건 주 4일제
    def check_nurse_constraints(self, schedule):
        subset_size = 14 #이쪽에 스케쥴 Np니까 그거 제대로 조정하는 절차 필요할듯함
        for nurse_schedule in schedule:    
            for i in range(5):
                subset_start = i * subset_size
                subset_end = (i + 1) * subset_size
                if sum(nurse_schedule[subset_start:subset_end]) > 4: #이거 이상하다 한번 고쳐주는 시간이 필요할듯함
                    return False 
            return True
    #하드제약조건 최소 간호사 수
    def check_day_constraints(self, schedule):
        for i in range(self.num_days): #왜 self.num_days로 했는지 모르겠음
            if schedule[:, i].sum() < day_data.iloc[i, 1]: 
                return False
        return True
    
    
    def fitness(self, schedule, min_nightshift = 5):
        # 초기화
        total_fitness = 0
        df_schedule = pd.DataFrame(schedule)
        ###하드 제약사항
        
        for i in range(num_days):
        # 최소 간호사 수 평가요소
            if df_schedule.iloc[:,i].values.sum() < day_data.iloc[i,1]:
                total_fitness-=100
                break #하드젱략조건
            # 주 4일제 사용
            #Degree 최소 요건
            # else:
            #     work_nurse = [index for index, value in enumerate(df_schedule.iloc[:,i]) if value == 1]
            #     # print(work_nurse)
            #     Work_Degree_list = nurse_data.iloc[work_nurse,-1].values.tolist()
            #     # print(Work_Degree_list)
            #     Work_Degree_nums = {"A" : Work_Degree_list.count("A"), "B" : Work_Degree_list.count("B"), "C" : Work_Degree_list.count("C")}
            #     print(Work_Degree_nums)
        ### 하드 제약사향
        
        ### 소프트 제약사항 ###
        # (각 날짜의 선호도) * (근무하면 1 아님 0)
        result = nurse_data.iloc[:,1:1+num_days].mul(schedule)
        # 각 간호사 끼리의 선호도 고려 -> 단순곱
        ## 이부분 해결필요 Degree 사용
        pref_list = []
        # nurse_degree = nurse_data.loc[:,'Degree']
        for i in range(num_nurses): 
            pref = nurse_data.iloc[:,num_days+1+i].dot(df_schedule)
            pref_list.append(pref.values/num_days)
        pref_df = pd.DataFrame(pref_list)
        total_fit = pref_df * result.values
        total_fitness = total_fit.values.sum()
        
        
    # 각 간호사에 대해 평가 패널티요소들 간호사 1,2,3,4,... 으로 올라가면서 work_day 에서 일을하는 index값 반환함
        for nurse_schedule in schedule:
            # 쉬프트 수에 따른 패널티 요소
            first_shift = 0
            second_shift = 0
            third_shift = 0
            work_day = [index for index, value in enumerate(nurse_schedule) if value == 1]
            consecutive_count = 1
            for i in range(1, len(work_day)):
                if work_day[i] - work_day[i - 1] == 1:
                    consecutive_count += 1
                else:
                    consecutive_count = 1
                # Apply penalty if consecutive occurrences are three or more
                if consecutive_count >= 2:
                    total_fitness -= 5

            for i in work_day:
                if i // 3 == 0:
                    first_shift += 1
                elif i//3 == 1:
                    second_shift += 1
                else:
                    third_shift += 1
            if third_shift < min_nightshift:
                total_fitness -= 10
            # print(sum(nurse_schedule[0:num_days//4]))
            # print(num_days//4)
            # print(len(nurse_schedule))
            if sum(nurse_schedule[0:14]) > 4 :
                total_fitness -= 100
            if sum(nurse_schedule[14:14*2]) > 4 :
                total_fitness -= 100
            if sum(nurse_schedule[14*2:14*3]) > 4 :
                total_fitness -= 100
            if sum(nurse_schedule[14*3:14*4]) > 4 :
                total_fitness -= 100
            if sum(nurse_schedule[14*4:]) > 4 :
                total_fitness -= 100
            
            # 16시간 휴식 확인 이후 패널티
            # max_daily_hours = 8  # 하루 최대 근무 시간
            # min_rest_time = 2
            # daily_hours = np.sum(nurse_schedule)
            # if daily_hours > max_daily_hours:
            #     total_fitness -= (daily_hours - max_daily_hours)
            # 21shift중 야간 쉬프트 최소 2회
        
        # Degree 평가해서 넣기
        # Need Degree_A_B_C 소프트 제약조건 대략 0.2 , 0.3, 0.5 대략적으로 이정도로 잡고 만들어주기
        # 스케쥴 제약들 평가하는 함수 만들기-> 출력은 print()로 터미널에 띄워주기 5순위
        # 주 4일제 평가를 위하여 2교대 or 고정근무재오전에만 일을 하겠다 
        # 이런식으로 야간에만 일을 하겠다 즉, Shift수에 변화가 생길것. 
        # 20% 정도의 야간근무 희망자 nurse_csv.py 에서 야간근무를 80% 이상하는 사람의
        # 비율이 20% 인지 평가하는 함수 필요 -> 성보님
        # 16시간 연속근무 불가
        # 주 4일제는 연속 12시간 이상 근무 불가
        
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
num_days = 60
plt.show()
ga = NurseSchedulingGA(num_nurses, num_days)
ga.evolve(generations=3)
plt.show()
print("베스트 스코어 리스트입니다! \n{}".format(bf))

# print("베스트 스코어를 달성한 거 \n{}".format(best_schedule))
# df_bf = pd.DataFrame(bf)
# scaler = MinMaxScaler()
# plt.show()
# b =scaler.fit_transform(df_bf)
# print(b)
# plt.plot(bf)