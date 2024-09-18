import argparse

def get_config():
    parser = argparse.ArgumentParser(
        description='dribblebot', formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--method", type=str, default='mlp', choices=["mlp", "aug", "emlp"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--load_run", type=str, help="Name of the run to load")
    parser.add_argument("--checkpoint", type=str, default="latest", help="Saved model checkpoint number")
    parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times")

    return parser
