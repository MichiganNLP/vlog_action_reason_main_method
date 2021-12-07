#!/usr/bin/env python
import json
import sys
from collections import defaultdict

from ifitb.util.file_utils import cached_path


def main() -> None:
    with open(cached_path(sys.argv[1])) as file:
        instances_by_verb = json.load(file)

    with open(cached_path(sys.argv[2])) as file:
        seen_instances = {(verb, instance["video"], instance["time_s"], instance["time_e"])
                          for verb, instances in json.load(file).items()
                          for instance in instances}

    new_instances_by_verb = defaultdict(list)
    for verb, instances in instances_by_verb.items():
        for instance in instances:
            if (verb, instance["video"], instance["time_s"], instance["time_e"]) not in seen_instances:
                new_instances_by_verb[verb].append(instance)

    print(json.dumps(new_instances_by_verb))


if __name__ == '__main__':
    main()
