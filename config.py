import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument("--project", type=str, help="Project Name", default="project")
    parser.add_argument("--method", type=str, help="Method to use", default="ridge")
    parser.add_argument("--seed", type=int, help="Seed for the experiment", default=0)
    parser.add_argument("--data_path", type=str, help="Path for data", default="./xe3_dataset_dft.xyz")

    return parser
