#!/usr/bin/env python
import json
import sys


def main() -> None:
    with open(sys.argv[1]) as file:
        instances_by_verb = json.load(file)

    with open(sys.argv[2], "w") as file:
        json.dump({verb: instances[:len(instances) // 10] for verb, instances in instances_by_verb.items()}, file)

    with open(sys.argv[3], "w") as file:
        json.dump({verb: instances[len(instances) // 10:] for verb, instances in instances_by_verb.items()}, file)


if __name__ == '__main__':
    main()
