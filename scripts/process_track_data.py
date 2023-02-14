#!/usr/bin/env python
"""
This script uses the io.reader to read the raw data (spacepoints, particles, etc)
in the trackml, athena, or acts format and save them in numpy format.

The reader will construct the true edges between spacepoints, label spacepoints
by their truth information. 
"""
import os

from multiprocessing import Pool
from functools import partial

from acctrack.io.utils import save_to_np
from acctrack.io.utils import load_from_np
from acctrack.io.utils import dump_data
from acctrack import io
from acctrack.viewer import viewer


def process(evtid, reader, output_dir, check_only=False):
    filename = os.path.join(output_dir, str(evtid))
    if check_only:
        data = load_from_np( filename + ".npz")
        dump_data(data)
        viewer.view_graph(data[0], data[1], data[2], outname=filename+"_graph.png")
    else:
        data = reader(evtid)
        save_to_np(filename, data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="process events in the folder")
    add_arg = parser.add_argument
    add_arg("in_dir", help="input directory")
    add_arg("out_dir", help="output directory")
    add_arg("-w", "--workers", help='number of workers', default=1, type=int)
    add_arg("--spname", help='spacepoint name', default='spacepoints')
    add_arg("--reader", help='reader class', default=None, choices=list(io.__all__)[:-1])
    add_arg("-c", "--check", help='performance checks of processed data', action='store_true')

    args = parser.parse_args()

    if args.reader is None:
        print("Specify a data type via --reader")
        parser.print_help()
        exit(1)

    reader = getattr(io, args.reader)(args.in_dir, args.spname)
    if args.check:
        ## read the first event and check the data
        evtid = reader.all_evtids[0]
        process(evtid, reader, args.out_dir, check_only=True)
    else:
        if args.workers < 2:
            for evtid in reader.all_evtids:
                process(evtid, reader, args.out_dir)
        else:
            with Pool(args.workers) as p:
                p.map(
                    partial(process, reader=reader, output_dir=args.out_dir),
                    reader.all_evtids)