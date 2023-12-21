import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument("--project", type=str, help="Project Name", default="project")
    parser.add_argument("--seed", type=int, help="Seed for the experiment", default=0)

    return parser
