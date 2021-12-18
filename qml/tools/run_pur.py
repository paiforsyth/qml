import argparse
import logging
import sys

from pur import update_requirements

logger = logging.getLogger(__name__)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = update_requirements(
        input_file="requirements.txt",
        dry_run=args.check,
        skip=["types-setuptools", "setuptools"],
    )
    updated = [x for x in result.values() if x[0]["updated"]]
    num_updated = len(updated)
    if num_updated > 0:
        logger.error(f"Stale packages: {updated}")
        sys.exit(1)
