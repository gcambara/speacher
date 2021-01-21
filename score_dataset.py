'''Score each sample of an input dataset according to a scoring function.'''

import os
import pandas as pd

from src.arguments import parse_arguments
from src.scoring import ScoringFunction

def main():
    # Get the arguments for the current execution.
    args = parse_arguments()
    print(args)

    # Read the training manifest tsv file.
    df = pd.read_csv(args.manifest, sep='\t')

    # Define the scoring function and apply it to the manifest.
    scoring_function = ScoringFunction().create(args)
    df = scoring_function(df)

    # Create the directory for output manifest storage.
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    if args.save_sorted_manifest:
        df.to_csv(os.path.join(args.out_dir, 'sorted_manifest.tsv'), sep='\t', index=None)

if __name__ == '__main__':
    main()
