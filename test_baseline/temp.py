with open("/home/v-yuancwang/AudioEditing/test_baseline/drop.txt", "w") as f:
    with open("/home/v-yuancwang/AudioEditing/test_baseline/add_0.txt", "r") as ff:
        lines = ff.readlines()
    for line in lines:
        p1, p2, t1, t2 = line.replace("\n", "").split("   ")
        t1 = t1.replace("_", " ", 5)
        t2 = t2.replace("_", " ", 5)
        t_e = "Drop: " + t2.lower()
        t_g = t1
        f.write(p2 + "   " + p1 + "   " + t_e + "   " + t_g + "\n")
    with open("/home/v-yuancwang/AudioEditing/test_baseline/add_1.txt", "r") as ff:
        lines = ff.readlines()
    for line in lines:
        p1, p2, t1, t2, tc = line.replace("\n", "").split("   ")
        t1 = t1.replace("_", " ", 5)
        t2 = t2.replace("_", " ", 5)
        t_e = "Drop: " + t2.lower()
        t_g = t1
        f.write(p2 + "   " + p1 + "   " + t_e + "   " + t_g + "\n")