with open("/home/v-yuancwang/AudioEditing/test_baseline/add.txt", "w") as f:
    with open("/home/v-yuancwang/AudioEditing/test_baseline/add_0.txt", "r") as ff:
        lines = ff.readlines()
    for line in lines:
        p1, p2, t1, t2 = line.replace("\n", "").split("   ")
        t1 = t1.replace("_", " ", 5)
        t2 = t2.replace("_", " ", 5)
        t_e = "Add: " + t2.lower()
        t_g = t1 + ", while " + t2.lower() + " in the background."
        f.write(p1 + "   " + p2 + "   " + t_e + "   " + t_g + "\n")
    with open("/home/v-yuancwang/AudioEditing/test_baseline/add_1.txt", "r") as ff:
        lines = ff.readlines()
    for line in lines:
        p1, p2, t1, t2, tc = line.replace("\n", "").split("   ")
        t1 = t1.replace("_", " ", 5)
        t2 = t2.replace("_", " ", 5)
        t_e = "Add: " + t2.lower()
        t_g = t1 + ", while " + t2.lower()
        if tc == '0':
            t_e += " in the beginning"
            t_g += " in the beginning"
        if tc == '1':
            t_e += " in the middle"
            t_g += " in the beginning"
        if tc == '2':
            t_e += " in the end"
            t_g += " in the beginning"
        f.write(p1 + "   " + p2 + "   " + t_e + "   " + t_g + "\n")