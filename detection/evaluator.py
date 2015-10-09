__author__ = 'mataevs'

def eval_performance(result_file):
    with open(result_file, "r") as inputFile:
        lines = inputFile.readlines()

    tp, fp, tn, fn = 0, 0, 0, 0
    for line in lines:
        words = line.split()
        tpw = words[1]
        fpw = words[2]
        fnw = words[3]

        tpv = int(tpw[tpw.index("=") + 1:])
        fpv = int(fpw[fpw.index("=") + 1:])
        fnv = int(fnw[fnw.index("=") + 1:])
        tnv = 0 if (tpv + fpv + fnv) != 0 else 1

        tp += tpv
        fp += fpv
        tn += tnv
        fn += fnv

    acc = int(float(tp + tn) / (tp + tn + fp + fn) * 100) / 100.0
    prec = int(float(tp) / (tp + fp) * 100) / 100.0
    recall = int(float(tp) / (tp + fn) * 100) / 100.0

    print result_file
    print "tp", tp, "fp", fp, "tn", tn, "fn", fn
    print "acc", acc, "prec", prec, "recall", recall
    print ""

    return tp, fp, tn, fn

if __name__ == "__main__":
    # eval_performance("results_icf_5000_2000icf.txt")
    # eval_performance("results_cascade_1d_5d_1h_5h_2kcascade.txt")
    # eval_performance("results_cascade_10_50_100cascade.txt")
    # eval_performance("results_cascade_100_500_2kcascade.txt")
    # eval_performance("results_cascade_2_cascade.txt")
    # eval_performance("results_oldhog_hog.txt")
    # eval_performance("results_oldhog_hog.txt")
    # eval_performance("results_new_hog.txt")
    # eval_performance("results_hoghog.txt")
    # eval_performance("results_sum_hog_hog.txt")
    eval_performance("results_cascade_5k_500_cascade.txt")