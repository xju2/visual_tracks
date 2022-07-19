#!/usr/bin/env python
import os

from multiprocessing import Pool
from functools import partial

from acctrack.io.utils import save_to_np
from acctrack import io


def process(evtid, reader, output_dir):
    data = reader(evtid)
    filename = os.path.join(output_dir, str(evtid))
    save_to_np(filename, data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="process events in the folder")
    add_arg = parser.add_argument
    add_arg("in_dir", help="input directory")
    add_arg("out_dir", help="output directory")
    add_arg("-w", "--workers", help='number of workers', default=1, type=int)
    add_arg("--spname", help='spacepoint name', default='spacepoints')
    add_arg("--reader", help='reader class', default=None, choices=list(io.__all__))
    args = parser.parse_args()

    if args.reader is None:
        print("Specify a data type via --reader")
        parser.print_help()
        exit(1)


    reader = getattr(io, args.reader)(args.in_dir, args.spname)

    if args.workers < 2:
        for evtid in reader.all_evtids:
            process(evtid, reader, args.out_dir)
    else:
        with Pool(args.workers) as p:
            p.map(
                partial(process, reader=reader, output_dir=args.out_dir),
                reader.all_evtids)