import re
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataloader import LOSO_sequence_generate

# Selected action units
AU_CODE = [1, 2, 4, 10, 12, 14, 15, 17, 25]
AU_DICT = {
    number: idx
    for idx, number in enumerate(AU_CODE)
}


def evaluate_adj(df, args):
    assert isinstance(df, (str, pd.DataFrame)), "Type not supported"
    if isinstance(df, str):
        # Read in data
        df = pd.read_csv(args.csv_name)
    
    # Take out the `Action Units` Columns
    data = df.loc[:, "Action Units"]

    # Create a blank matrix for counting the adjacent
    count_matrix = np.zeros((9, 9))
    
    # Create a blank list for counting the au
    count_au = np.zeros(9)

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
    
    # Replace 0 in count_au to 1
    count_au = np.where(count_au == 0.0, 1, count_au)

    # Compute the adjancent matrix
    adj_matrix = count_matrix / count_au.reshape(-1, 1)

    # Show the information
    print("AU appers:\n", count_au)

    if args["save_img"]:
        plt.matshow(adj_matrix, cmap="summer")
        for (i, j), z in np.ndenumerate(adj_matrix):
            plt.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')

        plt.savefig(args["jpg_name"], format="svg", dpi=1200)

    np.savez(args["npz_name"],
             adj_matrix=adj_matrix)


def save_LOSO_adj(args):
    data = pd.read_csv(args.csv_name)
    train_list, _ = LOSO_sequence_generate(data, "Subject")
    os.makedirs(args.npz_place, exist_ok=True)
    for idx, train_info in enumerate(train_list):
        evaluate_adj(df=train_info,
                     args={
                        "npz_name": f"{args.npz_place}/{idx}.npz",
                        "jpg_name": f"{args.image_place}/{idx}.svg",
                        "save_img": args.save_img
                     })


if __name__ == "__main__":
    # Argument parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_name",
                        type=str,
                        required=True,
                        help="Filename")
    parser.add_argument("--npz_place",
                        type=str,
                        required=True,
                        help="The root place for saving npz files")
    parser.add_argument("--save_img",
                        action="store_true",
                        default=False)
    parser.add_argument("--image_place",
                        type=str,
                        default=None,
                        help="The root place for saving images")
    args = parser.parse_args()

    save_LOSO_adj(args)
