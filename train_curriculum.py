'''Train with Curriculum Learning.'''

import os
import copy
import yaml
import pandas as pd

from src.arguments import parse_arguments
from src.pacing import PacingFunction

def main():
    # Get the arguments for the current execution.
    args = parse_arguments()
    print(args)

    # Read the training manifest tsv file.
    df = pd.read_csv(args.manifest, sep='\t')

    # Define the pacing function and apply it to the sorted manifest.
    pacing_function = PacingFunction().create(args)
    df_list = pacing_function(df)

    # Create the directory for output smaller datasets storage.
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    df_paths_list = []
    for index, sub_df in enumerate(df_list):
        path = os.path.join(args.out_dir, f'train_{index}.tsv')
        sub_df.to_csv(path, sep='\t', index=None)
        df_paths_list.append(path)

    if args.fairseq:
        train_fairseq(args, df_paths_list)

def train_fairseq(args, df_paths_list):
    '''Trains using Curriculum Learning with Fairseq framework.'''

    # First, let's read the base yaml file with the parameters for every curriculum training step.
    with open(args.fairseq_yaml, 'r') as file:
        base_yaml = yaml.load(file, Loader=yaml.FullLoader)

    # Now, let's create a separate yaml file for every curriculum training step.
    step_yamls = []
    for index, df_path in enumerate(df_paths_list):
        step_yaml = copy.deepcopy(base_yaml)
        if index != 0:
            step_yaml['train-subset'] = step_yamls[index - 1]['train-subset'] + ',' + os.path.basename(args.out_dir) + '/' + os.path.splitext(os.path.basename(df_path))[0]
        else:
            step_yaml['train-subset'] = os.path.basename(args.out_dir) + '/' + os.path.splitext(os.path.basename(df_path))[0]
        
        if index != len(df_paths_list) - 1:
            step_yaml['max-update'] = (index + 1) * args.step_length

        step_yaml['save-dir'] = os.path.join(step_yaml['save-dir'], os.path.basename(args.out_dir))
        step_yaml['tensorboard-logdir'] = step_yaml['save-dir']

        step_yamls.append(step_yaml)

        if args.save_curriculum_yaml:
            with open(os.path.splitext(df_path)[0] + '.yaml', 'w') as outfile:
                yaml.dump(step_yaml, outfile, default_flow_style=False)

    # Proceed to train.
    for step in step_yamls:
        command = 'fairseq-train '
        command += step['data'] + ' '
        for key, value in step.items():
            if key == 'data':
                continue
            else:
                if str(value) == "None":
                    command += '--' + str(key) + ' '
                else:
                    command += '--' + str(key) + ' ' + str(value) + ' '

        os.system(command)

if __name__ == '__main__':
    main()
