import argparse 
from sklearn.metrics import mean_squared_error, cohen_kappa_score

def parse_args():
    parser = argparse.ArgumentParser(description="Compute classification performance metrics for verdict verification experiment.")
    parser.add_argument("--results_path", type=str, help="Path to the file where verdict verification experiment results are stored.")

def main(config=None):

    if config is None:
        config = parse_args()


if __name__ == "__main__":
    main()