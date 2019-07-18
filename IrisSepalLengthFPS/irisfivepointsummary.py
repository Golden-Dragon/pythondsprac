import csv
import numpy as np
from scipy import stats


class IrisFivePointSummary:
    sepal_length_dataset = []

    with open("data/iris.csv") as csv_file:

        # PRE-PROCESS DATA

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
            else:
                sepal_length_dataset.append(float(row[0]))
            line_count += 1

        print(f'Processed {line_count} lines.')

        # CALCULATE FIVE POINT SUMMARY, MODE, VARIANCE & SD

        mean = np.mean(sepal_length_dataset)
        quartiles = np.percentile(sepal_length_dataset, [25, 50, 75])
        mode = stats.mode(sepal_length_dataset)
        variance = np.var(sepal_length_dataset)
        standard_deviation = np.std(sepal_length_dataset)
        data_min, data_max = np.asarray(sepal_length_dataset).min(), np.asarray(sepal_length_dataset).max()

        # PRINT DETAILS

        print("=========================================================")
        print("Min: %.3f" % data_min)
        print("Q1: %.3f" % quartiles[0])
        print("Median: %.3f" % quartiles[1])
        print("Q3: %.3f" % quartiles[2])
        print("Max: %.3f" % data_max)
        print("=========================================================")
        print("Mode Value: %.3f" % float(mode.mode))
        print("Mode Count: %.3f" % int(mode.count))
        print("Standard Deviation: %.3f" % standard_deviation)
        print("Variance: %.3f" % variance)
        print("=========================================================")
