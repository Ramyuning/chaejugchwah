# import pandas as pd
# import random

# # 간호사 수
# num_nurses = 20

# # Day, Pref, 직급 수
# num_days = 21
# num_prefs = 20  # 선호도를 20개로 늘림
# num_ranks = 3  # A, B, C 세 개의 직급

# # 각 직급의 최소 개수
# min_a_count = 3
# min_b_count = 3
# min_c_count = 3

# # 간호사 목록 생성
# nurses = [f"Nurse {i+1}" for i in range(num_nurses)]

# # 무작위 데이터 생성
# data = []
# ranks = ['A', 'B', 'C']
# rank_counts = {'A': 0, 'B': 0, 'C': 0}

# for i, nurse in enumerate(nurses):
#     row = [nurse]

#     # 무작위 Day 데이터 생성
#     row.extend([random.randint(1, 5) for _ in range(num_days)])

#     # 선호도 데이터 생성
#     preferences = []
#     for j in range(num_prefs):
#         pref_prob = random.random()
#         if pref_prob < 0.5:
#             preference = 3
#         else:
#             preference = random.choice([1, 2])

#         preferences.append(preference)
#     preferences[i] = 0  # 특정 간호사의 선호도를 0으로 설정
#     row.extend(preferences)

#     # 직급 데이터 생성
#     rank_prob = random.random()
#     if rank_prob < 0.5:
#         rank = 'B'
#     elif rank_prob < 0.75:
#         rank = 'A'
#     else:
#         rank = 'C'

#     # 직급의 최소 개수 조건을 충족할 때까지 다시 무작위로 선택
#     while rank_counts[rank] >= locals()[f'min_{rank.lower()}_count']:
#         rank_prob = random.random()
#         if rank_prob < 0.5:
#             rank = 'B'
#         elif rank_prob < 0.75:
#             rank = 'A'
#         else:
#             rank = 'C'

#     rank_counts[rank] += 1
#     row.extend([rank])

#     data.append(row)

# # 컬럼명 생성
# columns = ["Nurse"]
# columns.extend([f"Day_{i}" for i in range(1, num_days + 1)])
# columns.extend([f"Pref_{i}" for i in range(1, num_prefs + 1)])
# columns.extend(["Degree"])

# # 데이터프레임 생성
# df = pd.DataFrame(data, columns=columns)

# # CSV 파일로 저장
# df.to_csv("간호사_일정.csv", index=False)
# import pandas as pd
# import random

# # 간호사 수
# num_nurses = 20

# # Day, Pref, 직급 수
# num_days = 21
# num_prefs = 20  # 선호도를 20개로 늘림
# num_ranks = 3  # A, B, C 세 개의 직급

# # 각 직급의 최소 개수
# min_a_count = 5
# min_b_count = 5
# min_c_count = 5

# # 간호사 목록 생성
# nurses = [f"간호사 {i+1}" for i in range(num_nurses)]

# def generate_rank(rank_counts, min_count):
#     max_attempts = 10  # 최대 시도 횟수
#     attempt = 0

#     while attempt < max_attempts:
#         rank_prob = random.random()
#         if rank_prob < 0.5:
#             rank = 'B'
#         elif rank_prob < 0.75:
#             rank = 'A'
#         else:
#             rank = 'C'

#         # 직급의 최소 개수 조건을 충족하면 반환
#         if rank_counts[rank] < min_count:
#             rank_counts[rank] += 1
#             return rank

#         attempt += 1

#     # 최대 시도 횟수를 초과하면 'B'를 기본값으로 반환
#     return 'B'

# # 무작위 데이터 생성
# data = []
# ranks = ['A', 'B', 'C']
# rank_counts = {'A': 0, 'B': 0, 'C': 0}

# for i, nurse in enumerate(nurses):
#     row = [nurse]

#     # 무작위 Day 데이터 생성
#     row.extend([random.randint(1, 5) for _ in range(num_days)])

#     # 선호도 데이터 생성
#     preferences = []
#     for j in range(num_prefs):
#         pref_prob = random.random()
#         if pref_prob < 0.5:
#             preference = 3
#         elif pref_prob < 0.75:
#             preference = random.choice([1, 2])
#         else:
#             preference = 0.125  # 나머지는 0.125
#         preferences.append(preference)
#     preferences[i] = 0  # 특정 간호사의 선호도를 0으로 설정
#     row.extend(preferences)

#     # 직급 데이터 생성
#     rank = generate_rank(rank_counts, locals()[f'min_{rank.lower()}_count'])
#     row.extend([rank])

#     data.append(row)

# # 컬럼명 생성
# columns = ["Nurse"]
# columns.extend([f"Day_{i}" for i in range(1, num_days + 1)])
# columns.extend([f"Pref_{i}" for i in range(1, num_prefs + 1)])
# columns.extend(["직급"])

# # 데이터프레임 생성
# df = pd.DataFrame(data, columns=columns)

# # CSV 파일로 저장
# df.to_csv("간호사_일정.csv", index=False)
import pandas as pd
import random

# 간호사 수
num_nurses = 20

# Day, Pref, 직급 수
num_days = 21
num_prefs = 20  # 선호도를 20개로 늘림
num_ranks = 3  # A, B, C 세 개의 직급

# 각 직급의 최소 개수
min_a_count = 3
min_b_count = 3
min_c_count = 5

# 간호사 목록 생성
nurses = [f"Nurse {i+1}" for i in range(num_nurses)]

def generate_rank(rank_counts, min_count):
    max_attempts = 10  # 최대 시도 횟수
    attempt = 0

    while attempt < max_attempts:
        rank_prob = random.random()
        if rank_prob < 0.5:
            rank = 'B'
        elif rank_prob < 0.75:
            rank = 'A'
        else:
            rank = 'C'

        # 직급의 최소 개수 조건을 충족하면 반환
        if rank_counts[rank] < min_count:
            rank_counts[rank] += 1
            return rank

        attempt += 1

    # 최대 시도 횟수를 초과하면 'B'를 기본값으로 반환
    return 'B'

# 무작위 데이터 생성
data = []
ranks = ['A', 'B', 'C']
rank_counts = {'A': 0, 'B': 0, 'C': 0}

for i, nurse in enumerate(nurses):
    row = [nurse]

    # 무작위 Day 데이터 생성
    row.extend([random.randint(1, 5) for _ in range(num_days)])

    # 선호도 데이터 생성
    preferences = []
    for j in range(num_prefs):
        pref_prob = random.random()
        if pref_prob < 0.5:
            preference = 3
        else :
            preference = random.choice([1, 2])
        preferences.append(preference)
    preferences[i] = 0  # 특정 간호사의 선호도를 0으로 설정
    row.extend(preferences)

    # 직급 데이터 생성
    rank = generate_rank(rank_counts, min_b_count)  # 수정: min_b_count를 기본값으로 사용
    row.extend([rank])

    data.append(row)

# 컬럼명 생성
columns = ["Nurse"]
columns.extend([f"Day_{i}" for i in range(1, num_days + 1)])
columns.extend([f"Pref_{i}" for i in range(1, num_prefs + 1)])
columns.extend(["Degree"])

# 데이터프레임 생성
df = pd.DataFrame(data, columns=columns)

# CSV 파일로 저장
df.to_csv("nurse_schedule.csv", index=False)
