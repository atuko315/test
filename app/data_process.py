import pandas as pd
import csv

def parse_data(data):
    name = data[0]
    day = data[1]
    level = data[2]
    limit = data[3]
    player = data[4]
    total_time = data[-1]
    count = 6
    tmp = data[count]
    times = []
    while tmp != -2:
        times.append(tmp)
        count += 1
        tmp = data[count]
    print(times)
    with open("data/time.csv", mode="a") as f:
            writer = csv.writer(f)
            writer.writerow(times)
    avg_time = sum(times) / len(times)

    count += 1
    result = data[count]
    count += 3
    tmp_data = data[count:len(data)-2]
    print(tmp_data)
    ftimes = []
    for i in range(0, len(tmp_data), 3):
        imp = tmp_data[i]
        time = tmp_data[i+1]
        with open("data/imp_time.csv", mode="a") as f:
            writer = csv.writer(f)
            writer.writerow([imp, time])
    
    with open("data/abstract.csv", mode="a") as f:

        writer = csv.writer(f)
        writer.writerow([name, day, level, limit, avg_time, total_time]) 

with open("/home/student/PARL/benchmark/torch/AlphaZero/app/csv/20231207133022.csv", mode="r") as f:
    data = f.readline()
    data = data.strip('\n').split(',')
    data = list(map(float, data))
    print(data)
    print(type(data))
    parse_data(data)