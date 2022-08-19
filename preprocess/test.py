"2237,2008-02-02 13:47:30,0.0,0.0"
"2237,2008-02-02 13:52:22,116.30368,39.97505"
import os


def write2file(points):
    head = "2237,2008-02-02 13:47:30,"
    st = head + "0.0,0.0"
    res = [st]
    for p in points:
        lat = p[0]
        lng = p[1]
        s = head + str(lat) + "," + str(lng)
        res.append(s)
    with open("./tmp.trj", "w") as f:
        for r in res:
            f.write(r)
    return 1

def span_search(points,W):
    write2file(points)
    ratio = W / len(points)
    os.system(f'./tmp.trj	{len(points)}	{ratio}	4	2')
