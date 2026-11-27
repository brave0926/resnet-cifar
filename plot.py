import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")                 # save to file, no window needed
import matplotlib.pyplot as plt

# run folder -> (label, color = depth, line style = type)
RUNS = [
    ("plain20",  "plain-20",  "tab:blue", "--"),
    ("resnet20", "ResNet-20", "tab:blue", "-"),
    ("plain56",  "plain-56",  "tab:red",  "--"),
    ("resnet56", "ResNet-56", "tab:red",  "-"),
]


def read_log(path):
    epochs, train_err, test_err = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_err.append(float(row["train_err"]))
            test_err.append(float(row["test_err"]))
    return epochs, train_err, test_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs", help="folder with plain20/, resnet20/, ...")
    args = parser.parse_args()

    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(12, 4.5))
    rows = []
    for folder, label, color, style in RUNS:
        path = os.path.join(args.runs, folder, "log.csv")
        if not os.path.exists(path):
            print(f"skip {label}: {path} not found")
            continue
        epochs, train_err, test_err = read_log(path)
        if not epochs:
            print(f"skip {label}: {path} is empty")
            continue
        ax_train.plot(epochs, train_err, color=color, linestyle=style, label=label)
        ax_test.plot(epochs, test_err, color=color, linestyle=style, label=label)
        rows.append((label, epochs[-1], train_err[-1], test_err[-1]))   # FINAL epoch

    if not rows:
        print(f"No log.csv found in {args.runs}/")
        return

    for ax, title in ((ax_train, "Training error"), (ax_test, "Test error")):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("error (%)")
        ax.set_ylim(0, 40)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("CIFAR-10: dashed = plain, solid = ResNet; blue = 20 layers, red = 56 layers")
    plt.tight_layout()
    png_path = os.path.join(args.runs, "curves.png")
    plt.savefig(png_path, dpi=150)

    table = ["| Model | Epochs | Final train err (%) | Final test err (%) |",
             "|-------|--------|---------------------|--------------------|"]
    for label, epoch, train_e, test_e in rows:
        table.append(f"| {label} | {epoch} | {train_e:.2f} | {test_e:.2f} |")
    text = "\n".join(table) + "\n"
    md_path = os.path.join(args.runs, "results.md")
    with open(md_path, "w") as f:
        f.write(text)

    print(text)
    print(f"Saved {png_path} and {md_path}")


if __name__ == "__main__":
    main()
