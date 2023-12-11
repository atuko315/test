import pandas as pd
import csv

def parse_data(data):
    name = data[0]
    if name == 0: # チェック用
        return 
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

    #print(times)
    with open("data/time.csv", mode="a") as f:
            writer = csv.writer(f)
            writer.writerow(times)
    avg_time = sum(times) / len(times)

    count += 1
    result = data[count]
    count += 3
    tmp_data = data[count:len(data)-1]
    #print(tmp_data)
    ftimes = []
    
    i = 0
    while i < len(tmp_data)-3:
        #print(tmp_data[i+1: i+4])
        if tmp_data[i]!= -2 or -2 in tmp_data[i+1: i+4]:
            i += 1
            continue


        mimp = tmp_data[i+1]
        imp = tmp_data[i+2]
        time = tmp_data[i+3]
        with open("data/imp_time.csv", mode="a") as f:
            writer = csv.writer(f)
            writer.writerow([mimp, imp, time])
        i += 4

    
    
    with open("data/abstract.csv", mode="a") as f:

        writer = csv.writer(f)
        writer.writerow([name, day, level, limit, player, avg_time, result, total_time]) 

def parse_choice(data):
    count = 0
    while count < len(data)-1:
        if data[count] == -2 and data[count+1] >= 0 and data[count+1] <= 6:
            rank = data[count+1]
            time = data[count+2]
            count += 3
            with open("data/abstract.csv", mode="a") as f:
                writer = csv.writer(f)
                writer.writerow([rank, time]) 
                continue
        count += 1


def __main__():
     
    with open("csv/data.csv", mode="r") as f:
        reader = csv.reader(f)
        for data in reader:
            #data = data.strip('\n').split(',')
            data = list(map(float, data))
            try:
                parse_data(data)
            except Exception:
                print(data)
                continue
    
    with open("csv/choices.csv", mode="r") as f:
        reader = csv.reader(f)
        for data in reader:
            #data = data.strip('\n').split(',')
            data = list(map(float, data))
            try:
                parse_choice(data)
            except Exception:
                print(data)
                continue

__main__()
