import pandas as pd

if __name__ == '__main__':
    data_frame1 = pd.read_csv("../Datasets/avila-tr.csv", header=None)
    data_frame2 = pd.read_csv("../Datasets/avila-ts.csv", header=None)
    data_merge = data_frame1.append(data_frame2)
    data_merge.columns = ["intercolumnar distance", "upper margin", "lower margin",
                          "exploitation", "row number", "modular ratio", "interlinear spacing ",
                          "weight", "peak number", "modular ratio/interlinear spacing",
                          "Class"]
    encoder = {'Class': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
                         'I': 8, 'W': 9, 'X': 10, 'Y': 11}}
    data_merge.replace(encoder, inplace=True)
    print(data_merge.head)
    # data_merge.to_csv("../Datasets/avila_clean.csv", index=False, header=True)
