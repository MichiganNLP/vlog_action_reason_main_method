# Main method of "WhyAct: Identifying Action Reasons in Lifestyle Vlogs"

This is the main method of the paper "WhyAct: Identifying Action Reasons in Lifestyle Vlogs" by Ignat et al., 2021. See more info in the [main repo](https://github.com/michigannlp/vlog_action_reason).

## Setup

Using Conda:

```bash
conda env create
conda activate intention-fitb
```

Place the JSON files under `data/`.

## Train

```bash
./scripts/run.py --train
```

See the available options using `--help`.
