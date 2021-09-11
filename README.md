# ga-scheduling

To run the code found, first install required packages using this command:
```
pip3 install -r requirements.txt
```

Then create a dataset using the following command:
```
python3 create_data.py -i small_data.yaml
```

Other YAML files may be used if a different sized dataset is desired. Included in the code is the larger dataset called `medium_data.yaml`.

To run the genetic algorithm use the code:
```
python3 genetic_algorithm_main.py --input small_data.npz 
```

The results will be store in the `results` folder. There are many other options than can be given to `genetic_algorithm_main.py` To see these, type:

```
python3 genetic_algorithm_main.py --help
```

Alternatively the file `run_ga.py` can be edited directly with desired settings, then run with 
```
python3 run_ga.py
```

To see a persons timetable, use the following command. `--input` may also be included if a choice of results file is needed.

```
python3 timetable.py --person_time_table 42
```