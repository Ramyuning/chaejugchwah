import numpy as np
import pandas as pd
population = []
for i in range(3):
    schedule = np.random.randint(2, size=(20 , 21))
    population.append(schedule)
# print(population)
# print(schedule)
# asdf = pd.DataFrame(schedule)
# print(asdf)
# print(asdf.iloc[:,2].values.tolist())

# 값이 1인 항목의 인덱스 추출
# indices_of_ones = [index for index, value in enumerate(schedule[0]) if value == 1]

# 결과 출력
# print(indices_of_ones)
# for i in population:
#     for schedule in i:
#         print (schedule[0])
# dt = pd.read_csv('/Users/jojeonghyeon/Documents/WorkSpace/PYTHON/chaejugchwah/output.csv')
nurse_data = pd.read_csv('/Users/jojeonghyeon/Documents/WorkSpace/PYTHON/chaejugchwah/nurse_schedule.csv')
# print(ndt.iloc[0,:21]*[2 for _ in range(21)])
# print(ndt.iloc[:,0:21]*schedule)
# print(schedule.shape)
# print(ndt.iloc[:,0:21].shape)
# print(ndt.iloc[:,0:21].mul(schedule))
# result = ndt.iloc[:,1:22].mul(schedule)
# print(schedule)
# print("")
# print(result.values)
# print(result.values.sum())
# print(result.dtypes)
# total_fitness = 0
# total_fitness += np.sum(ndt.iloc[:,0:21]*schedule)
# p
# total_fitness += np.sum(ndt.iloc[:,0:21]*schedule)

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
nurse_degree = nurse_data.loc[:,'Degree']
print(nurse_degree)
# 각 간호사에 대해 평가 패널티요소들
for nurse_schedule in schedule:
    # 쉬프트 수에 따른 패널티 요소
    first_shift = 0
    second_shift = 0
    third_shift = 0
    min_nightshift = 2 ##최소 밤에 일해야 하는 쉬프트
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
print(total_fitness)