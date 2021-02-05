import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Curriculum learning design for speech recognition')
    parser.add_argument('--manifest', default='', help='path to the train dataset manifest tsv file')
    parser.add_argument('--out_dir', default='./', help='path to the output directory for new manifests')
    # Scoring arguments
    parser.add_argument('--scoring_function', default='n_frames', help='scoring function: n_frames | asr')
    parser.add_argument('--save_sorted_manifest', dest='save_sorted_manifest', action='store_true', help='save manifest sorted by the defined scoring function')
    parser.set_defaults(save_sorted_manifest=False)
    parser.add_argument('--asr_model_zoo', default='./models', help='path to directory containing downloaded pre-trained models')
    parser.add_argument('--asr_model', default='', help='path to the ASR model checkpoint to be used')
    parser.add_argument('--asr_download_model', default='', help='name of the pretrained ASR model to be downloaded: s2t_transformer_s')
    # Pacing arguments
    parser.add_argument('--pacing_function', default='fixed_exponential', help='pacing function: fixed_exponential')
    parser.add_argument('--step_length', type=int, default=1000, help='number of iterations at each step')
    parser.add_argument('--increase', type=float, default=1.9, help='exponential factor to increase the number of samples at each step')
    parser.add_argument('--starting_percent', type=float, default=0.3, help='percentage of the total training data to start with at first step')
    # Training arguments
    parser.add_argument('--fairseq', dest='fairseq', action='store_true', help='train with Fairseq framework')
    parser.set_defaults(fairseq=False)
    parser.add_argument('--fairseq_yaml', default='', help='path to base configuration YAML for Fairseq framework')
    parser.add_argument('--save_curriculum_yaml', dest='save_curriculum_yaml', action='store_true', help='save generated curriculum learning YAMLs')
    parser.set_defaults(save_curriculum_yaml=False)
    return parser.parse_args()