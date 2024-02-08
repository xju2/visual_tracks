# visual_tracks
Visualize tracks as graph


## Instructions

Available tasks `python run.py -h`.
```text
(vroot) xju@Epic: visual_tracks$ python run.py -h
run is powered by Hydra.

== Configuration groups ==
Compose your configuration from those groups (group=option)

extras: default
paths: default
reader: trackml
task: convert_gnn_tracks_for_fitting, plot_bad_tracks, process_raw_athena_root, process_raw_track_data, tasks, test


== Config ==
Override anything in the config (foo.bar=value)

task_name: ''
paths:
  root_dir: ${oc.env:PROJECT_ROOT}
  log_dir: ${paths.root_dir}/logs/
  output_dir: ${hydra:runtime.output_dir}
  work_dir: ${hydra:runtime.cwd}
extras:
  ignore_warnings: false
  enforce_tags: false
  print_config: true


Powered by Hydra (https://hydra.cc)
Use --hydra-help to view Hydra specific help
```