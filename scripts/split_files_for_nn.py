#!/usr/bin/env python

import os
from types import SimpleNamespace
from pathlib import Path
import re
import random

def soft_link(src, dest):
    os.symlink(os.path.abspath(src), dest)
    # tf.io.gfile.copy


def split(input_dir, output_dir, pattern, no_shuffle, **kwargs):

    pattern = "*.parquet" if pattern is None else pattern
    input_dir = Path(input_dir)
    datatypes = ['trainset', 'valset', 'testset']
    output_dir = Path(output_dir)

    outdirs = SimpleNamespace(**dict(zip(datatypes, [output_dir / x for x in datatypes])))
    [x.mkdir(parents=True, exist_ok=True) for x in outdirs.__dict__.values()]

    evtid_pattern = "event([0-9]*)-particles.parquet"
    regrex = re.compile(evtid_pattern)

    def find_evt_info(x):
        matched = regrex.match(x.name)
        if matched:
            return int(matched.group(1).strip())
        else:
            return -1

    all_files = list(input_dir.glob("*-particles.parquet"))
    all_evtids = [find_evt_info(x) for x in all_files]
    all_evtids = [x for x in all_evtids if x >= 0]

    if not no_shuffle:
        print("Shuffling input files")
        random.shuffle(all_evtids)

    n_files = len(all_evtids)
    print("Total {} Events".format(n_files))

    train_frac, val_frac = 0.8, 0.1

    n_train = int(train_frac * n_files)
    n_val = int(val_frac * n_files)
    n_test = n_files - n_train - n_val
    print("Training   {} events".format(n_train))
    print("Validation {} events".format(n_val))
    print("Testing    {} events".format(n_test))

    def make_copy(evtids, outdir):
        for evtid in evtids:
            for fname in ["particles", "truth"]:
                file_name = input_dir / f"event{evtid:06d}-{fname}.parquet"
                dest = outdir / file_name.name
                soft_link(file_name, dest)

    make_copy(all_evtids[:n_train], outdirs.trainset)
    make_copy(all_evtids[n_train:n_train + n_val], outdirs.valset)
    make_copy(all_evtids[n_train + n_val:], outdirs.testset)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="split files in a folder into train, val and test")
    add_arg = parser.add_argument
    add_arg("input_dir", help="input directory")
    add_arg("output_dir", help="output directory")
    add_arg("--pattern", help='input data pattern',)
    add_arg("--no-shuffle", help='shuffle input files', action='store_true')
    args = parser.parse_args()

    split(**args.__dict__)
