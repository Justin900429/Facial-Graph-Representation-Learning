import re
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Selected action units
AU_CODE = [1, 2, 4, 10, 12, 14, 15, 17, 25]
AU_DICT = {
    number: idx
    for idx, number in enumerate(AU_CODE)
}

# Create a blank matrix for counting the adjacent
count_matrix = np.zeros((9, 9))
# Create a blank list for counting the au
count_au = np.zeros(9)


def evaluate_adj(args):
    # Read in data
    df = pd.read_csv(args.csv_name)
    # Take out the `Action Units` Columns
    data = df.loc[:, "Action Units"]

    # Split the action list
    for idx, unit in enumerate(data):
        # Find only the digit
        au_list = re.findall(r"\d+", unit)

        # Filter the AU_CODE
        au_list = list(filter(lambda au: int(au) in AU_CODE, au_list))

        for i in range(len(au_list)):
            first_code = AU_DICT[int(au_list[i])]
            for j in range(i + 1, len(au_list)):
                second_code = AU_DICT[int(au_list[j])]

                count_matrix[first_code, second_code] += 1
                count_matrix[second_code, first_code] += 1

            # Count the total appear times
            count_au[first_code] += 1

    # Compute the adjancent matrix
    adj_matrix = count_matrix / count_au.reshape(-1, 1)

    # Show the information
    print("AU appers:\n", count_au)

    plt.matshow(adj_matrix, cmap="summer")
    for (i, j), z in np.ndenumerate(adj_matrix):
        plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

    plt.savefig(args.jpg_name, format="svg", dpi=1200)

    np.savez(args.npz_name,
             adj_matrix=adj_matrix)


if __name__ == "__main__":
    # Argument parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_name",
                        type=str,
                        required=True,
                        help="Filename")
    parser.add_argument("--npz_name",
                        type=str,
                        required=True,
                        help="The name for the npz file to be saved")
    parser.add_argument("--jpg_name",
                        type=str,
                        required=True,
                        help="The name for the jpg file to be saved")
    args = parser.parse_args()

    evaluate_adj(args)
