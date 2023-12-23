import pandas as pd
import random

# 간호사 수
num_nurses = 40

# Day, Pref, 직급 수
num_days = 90
num_prefs = 40  # 선호도를 20개로 늘림 간호사 수와 동일하도록 설정해주세요
num_ranks = 3  # A, B, C 세 개의 직급

# 각 직급의 최소 개수 서울대병원 2:5:13실제로
min_a_count = num_nurses*0.1
min_b_count = num_nurses*0.25
min_c_count = num_nurses*0.65

# 간호사 목록 생성
nurses = [f"Nurse {i+1}" for i in range(num_nurses)]

def generate_rank(rank_counts):
    max_attempts = 100  # 최대 시도 횟수
    attempt = 0
    stopping = False
    while stopping != True:
        if attempt >= max_attempts:
            stopping = True
            print("생성실패입니다")
        rank_list = []
        for _ in range(num_nurses):
            rank_prob = random.random()
            if rank_prob < 0.65:
                rank = 'C'
            elif rank_prob < 0.9:
                rank = 'B'
            else:
                rank = 'A'
            rank_list.append(rank)
            rank_counts[rank] += 1
            # if rank_counts[rank] < min_count:
            #     rank_counts[rank] += 1
            #     return rank
            # 직급의 최소 개수 조건을 충족하면 반환
        if rank_counts["A"] < min_a_count:
            attempt += 1
        elif rank_counts["B"] < min_b_count:
            attempt += 1
        elif rank_counts["C"] < min_c_count:
            attempt += 1
        else:

            stopping = True
            #통과
    return rank_list

# 무작위 데이터 생성
data = []
ranks = ['A', 'B', 'C']
rank_counts = {'A': 0, 'B': 0, 'C': 0}
rank_list = generate_rank(rank_counts)
print(rank_list)

for i, nurse in enumerate(nurses):
    row = [nurse]
        # 밤 선호도 랜덤 데이터 생성
    one_shift_preferences = []
    two_shift_preferences = []
    night_preferences = []
    nurse_prob = random.random()
    if nurse_prob >=0.781:
        for _ in range(num_days//3):
            one_shift_pref = random.choice([1,2,3,4,5])
            two_shift_pref = random.choice([1,2,3,4,5])
            night_pref_prob = random.random()
            if night_pref_prob >= 0.85:
                night_preference = 5
            elif night_pref_prob<= 0.7:
                night_preference = 4 
            else :
                night_preference = random.choice([1,2,3])
            night_preferences.append(night_preference)
            one_shift_preferences.append(one_shift_pref)
            two_shift_preferences.append(two_shift_pref)
    else:
        for _ in range(num_days//3):
            one_shift_pref = random.choice([1,2,3,4,5])
            two_shift_pref = random.choice([1,2,3,4,5])
            night_preference = random.choice([1,2,3,4,5])
            
            night_preferences.append(night_preference)
            one_shift_preferences.append(one_shift_pref)
            two_shift_preferences.append(two_shift_pref)
    for shift_number in range(num_days//3):    
        row.extend([one_shift_preferences[ shift_number]])
        row.extend([two_shift_preferences[ shift_number]])
        row.extend([night_preferences[ shift_number]])
    # 무작위 Day 데이터 생성 이부분 5에 대한 선호도가 높도록 수정 필요함!!
    # row.extend([random.randint(1, 5) for _ in range(num_days)])
    
    # 선호도 데이터 생성
    preferences = []
    for j in range(num_prefs):
        pref_prob = random.random()
        if pref_prob < 0.5:
            preference = 3
        else :
            preference = random.choice([1, 2])
        preferences.append(preference)
    preferences[i] = 0  # 자기자신의 선호도를 0으로 설정
    row.extend(preferences)
    # 직급 데이터 생성
    # rank = generate_rank(rank_counts) 
    rank = rank_list[i]
    row.extend([rank])
    data.append(row)
    # print(data)


# 컬럼명 생성
columns = ["Nurse"]
columns.extend([f"Day_{i}" for i in range(1, num_days + 1)])
columns.extend([f"Pref_{i}" for i in range(1, num_prefs + 1)])
columns.extend(["Degree"])

# 데이터프레임 생성
df = pd.DataFrame(data, columns=columns)

# CSV 파일로 저장
df.to_csv("./chaejugchwah/nurse_schedule.csv", index=False)
