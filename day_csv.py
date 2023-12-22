import csv
import random

num_day = 90
num_nurses = 40
# CSV 파일의 헤더 정의
header = ["Day", "Need_person", "Need_Degree_A", "Need_Degree_B", "Need_Degree_C"]

# CSV 파일의 행 데이터 생성
rows = []
for day in range(1, 1+num_day): 
    need_person = random.randint(int(num_nurses*0.2) , int(num_nurses*0.4)) ##임의값
    need_degree_a = 1 ##임의값 이부분에 대한 조정 필요!!
    # need_degree_a = random.randint(2) ##임의값 이부분에 대한 조정 필요!!
    need_degree_b = random.randint(int(need_person*0.1),int(need_person*0.15)) ##임의값 이부분에 대한 조정 필요!!
    need_degree_c = need_person-need_degree_b-need_degree_a ##임의값 이부분에 대한 조정 필요!!
    # need_degree_c = random.randint(need_person-need_degree_b-need_degree_c) ##임의값 이부분에 대한 조정 필요!!
    row = [day, need_person, need_degree_a, need_degree_b, need_degree_c]
    rows.append(row)

# CSV 파일 쓰기
with open("./chaejugchwah/output.csv", mode="w", newline='') as file:
    writer = csv.writer(file)
    
    # 헤더 쓰기
    writer.writerow(header)
    
    # 행 데이터 쓰기
    writer.writerows(rows)

print("CSV 파일이 생성되었습니다: output.csv")