import csv
import random

# CSV 파일의 헤더 정의
header = ["Day", "Need_person", "Need_Degree_A", "Need_Degree_B", "Need_Degree_C"]

# CSV 파일의 행 데이터 생성
rows = []
for day in range(1, 22):
    need_person = random.randint(10, 20)
    need_degree_a = random.randint(1, 5)
    need_degree_b = random.randint(1, 5)
    need_degree_c = random.randint(1, 5)
    
    row = [day, need_person, need_degree_a, need_degree_b, need_degree_c]
    rows.append(row)

# CSV 파일 쓰기
with open("output.csv", mode="w", newline='') as file:
    writer = csv.writer(file)
    
    # 헤더 쓰기
    writer.writerow(header)
    
    # 행 데이터 쓰기
    writer.writerows(rows)

print("CSV 파일이 생성되었습니다: output.csv")