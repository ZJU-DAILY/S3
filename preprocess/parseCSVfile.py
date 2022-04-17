# timestamp;latitude;longitude;speed;direction
from matplotlib import pyplot as plt
import csv
res = []
i = 0
with open('../datasets/porto.csv', 'r') as f:

    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:
        pos = row[8]
        res.append(pos)

print(len(res))
with open('../datasets/porto.pos', 'w') as f:
    for it in res:
        i += 1
        # if i > 100000:
        #     break
        f.write(it + "\n")
