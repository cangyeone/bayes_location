import json
import pandas as pd

def pretty_json_file(infile_name, outfile_name): 
    with open(infile_name, 'r') as infile:
        js = json.load(infile)
    with open(outfile_name, 'w') as outfile: 
        json.dump(js, outfile, indent=4)

pretty_json_file("run_fm3d/data/svi/out/Catalogue.json", "run_fm3d/data/svi/out/Catalogue.json.pretty.json")