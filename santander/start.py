import sys
import logging

from time import sleep


def main():
    while True:
        logger.info("Iterating ...")

        sleep(5)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Add handler
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    #
    #
    #
    #
    main()
