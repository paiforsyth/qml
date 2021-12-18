import argparse
import sys

from pur import update_requirements

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    messages = [
        x[0]["message"]
        for x in update_requirements(
            input_file="requirements.txt", dry_run=args.check
        ).values()
    ]
    if len(messages) > 0:
        sys.exit(1)
