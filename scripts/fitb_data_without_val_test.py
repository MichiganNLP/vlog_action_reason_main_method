#!/usr/bin/env python
import json
import sys
from collections import defaultdict
from typing import Set, Tuple


def _add_instances_from_path(path: str, seen_instances: Set[Tuple[str, str, str, str]]) -> None:
    with open(path) as file:
        seen_instances.update((verb, instance["video"], instance["time_s"], instance["time_e"])
                              for verb, instances in json.load(file).items()
                              for instance in instances)


def main() -> None:
    with open(sys.argv[1]) as file:
        instances_by_verb = json.load(file)

    seen_instances = set()
    _add_instances_from_path(sys.argv[2], seen_instances)
    _add_instances_from_path(sys.argv[3], seen_instances)

    new_instances_by_verb = defaultdict(list)
    for verb, instances in instances_by_verb.items():
        for instance in instances:
            if (verb, instance["video"], instance["time_s"], instance["time_e"]) not in seen_instances:
                new_instances_by_verb[verb].append(instance)

    print(json.dumps(new_instances_by_verb))


if __name__ == '__main__':
    main()
