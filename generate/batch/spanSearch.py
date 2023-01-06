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
    with open("/home/hch/Desktop/trjcompress/generate/batch/MinError/data/tmp.trj", "w") as f:
        for r in res:
            f.write(r + "\n")
    return 1


def span_search(points, W):
    ratio = W / len(points)
    with open("/home/hch/Desktop/trjcompress/generate/batch/MinError/config.txt", "w") as f:
        f.write(f'data/tmp.trj	{len(points)}	{ratio}	4	2')
    write2file(points)
    os.system('/home/hch/Desktop/trjcompress/generate/batch/e.sh > a.txt')
    with open("/home/hch/Desktop/trjcompress/generate/batch/MinError/stat.txt", "r") as f:
        res = f.readlines()
    err = float(res[0].strip("\n"))
    with open("/home/hch/Desktop/trjcompress/generate/batch/MinError/res.txt", "r") as f:
        res = f.readlines()
    idx = [int(s.strip("\n")) for s in res]
    if len(points) - 1 not in idx:
        idx.append(len(points) - 1)
    return idx, err

def error_search(points, W):
    ratio = round(W / len(points), 1)
    with open("/home/hch/Desktop/trjcompress/generate/batch/MinError/config.txt", "w") as f:
        f.write(f'data/tmp.trj	{len(points)}	{ratio}	3	2')
    write2file(points)
    os.system('/home/hch/Desktop/trjcompress/generate/batch/e.sh > a.txt')
    with open("/home/hch/Desktop/trjcompress/generate/batch/MinError/stat.txt", "r") as f:
        res = f.readlines()
    err = float(res[0].strip("\n"))
    with open("/home/hch/Desktop/trjcompress/generate/batch/MinError/res.txt", "r") as f:
        res = f.readlines()
    idx = [int(s.strip("\n")) for s in res]
    if len(points) - 1 not in idx:
        idx.append(len(points) - 1)

    return idx, err

if __name__ == '__main__':
    points = [(116.30368, 39.97505), (116.30113, 39.99632), (116.34702, 39.98575), (116.34909, 39.93935),
              (116.34924, 39.92385), (116.34652, 39.91962), (116.33505, 39.94455), (116.29119, 39.96872),
              (116.28234, 39.95188), (116.27636, 39.95222), (116.27139, 39.9677), (116.2813, 39.97752),
              (116.27008, 39.95727), (116.26868, 39.93307), (116.26868, 39.93307)]
    print(len(points))
    idx, err = span_search(points, 6)
    print(idx, err)
