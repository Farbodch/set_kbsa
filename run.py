import argparse

def main():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--variable_name", default="default_value", help="tooltip-here")
    args = parser.parse_args()
    print(f"{args.variable_name}")
    return 0

if __name__ == "__main__":
    main()