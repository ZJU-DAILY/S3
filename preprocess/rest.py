# from preprocess.SpatialRegionTools import cell2gps
# from matplotlib import pyplot as plt
# import pickle
#
# with open('pickle.txt', 'rb') as f:
#     var_a = pickle.load(f)
# region = pickle.loads(var_a)
#
# # with open("dataset/test", "r") as f:
# #     ss = f.readlines()
#
# # trg = ss[0].split(" ")
# # x = []
# # y = []
# # for p in trg:
# #     x_, y_ = cell2gps(region, int(p))
# #     x.append(x_)
# #     y.append(y_)
# # xx = []
# # yy = []
# #
# # src = str2trip(ss[1])
# # for p in src:
# #     xx.append(p[0])
# #     yy.append(p[1])
# # plt.plot(x, y, color='r')
# # plt.plot(xx, yy)
# # plt.show()
#
# with open("dataset/train.src", "r") as f:
#     ss = f.readlines()
# print(len(ss))
# i = 1
# for s in ss:
#     trg = s.split(" ")
#     x = []
#     y = []
#     if i == 10000:
#         break
#     i += 1
#     for p in trg:
#         x_, y_ = cell2gps(region, int(p))
#         x.append(x_)
#         y.append(y_)
#     plt.plot(x, y, color='b')
# # plt.show()
#
#
# with open("dataset/eval.src", "r") as f:
#     ss = f.readlines()
# print(len(ss))
# i = 1
# for s in ss:
#     trg = s.split(" ")
#     x = []
#     y = []
#     if i % 10000 == 0:
#         print("no",i)
#     i += 1
#     for p in trg:
#         x_, y_ = cell2gps(region, int(p))
#         x.append(x_)
#         y.append(y_)
#     plt.plot(x, y, color='r')
# plt.show()

with open("../datasets/eval.src", "r") as f:
    ss = f.readlines()
length = 30
cnt = 0
with open("../datasets/infer.src", "w") as f:
    for s in ss:
        s = s.strip("\n")
        s = s.split(" ")
        if len(s) >= length:
            res = ""
            for k in range(0, length):
                if k != length - 1:
                    res += s[k] + " "
                else:
                    res += s[k]
            res += "\n"
            cnt += 1
            f.write(res)
            if cnt >= 1000:
                break
print(cnt)
