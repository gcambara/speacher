import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Curriculum learning design for speech recognition')
    parser.add_argument('--manifest', default='', help='path to the train dataset manifest tsv file')
    parser.add_argument('--out_dir', default='./', help='path to the output directory for new manifests')
    # Scoring arguments
    parser.add_argument('--scoring_function', default='n_frames', help='scoring function: n_frames | asr')
    parser.add_argument('--sort_manifest', dest='sort_manifest', action='store_true', help='save manifest sorted by the defined scoring function')
    parser.set_defaults(sort_manifest=False)
    parser.add_argument('--asr_metric', default='wer', help='ASR metric for scoring function: wer | ter')
    parser.add_argument('--asr_model_zoo', default='./models', help='path to directory containing downloaded pre-trained models')
    parser.add_argument('--asr_model', default='', help='path to the ASR model checkpoint to be used')
    parser.add_argument('--asr_download_model', default='', help='name of the pretrained ASR model to be downloaded: s2t_transformer_s')
    parser.add_argument('--asr_tokenizer', default='', help='name of the ASR HuggingFace tokenizer to be downloaded, if empty, it is set identically to --asr_download_model')
    parser.add_argument('--scoring_sorting', default='', help='sort by length in order to speed up inference: ascending | descending')

    # Pacing arguments
    parser.add_argument('--pacing_function', default='fixed_exponential', help='pacing function: fixed_exponential | binning')
    ## Exponential pacing function arguments
    parser.add_argument('--step_length', type=int, default=1000, help='number of iterations at each step')
    parser.add_argument('--increase', type=float, default=1.9, help='exponential factor to increase the number of samples at each step')
    parser.add_argument('--starting_percent', type=float, default=0.3, help='percentage of the total training data to start with at first step')
    ## Binning pacing function arguments
    parser.add_argument('--bin_variable', default='', help='name of the variable in the manifest to create bins from')
    parser.add_argument('--descending', dest='descending', action='store_true', help='Sort bin variable in descending order')
    parser.set_defaults(descending=False)
    parser.add_argument('--bin_method', default='qcut', help='method for binning: cut | qcut')
    parser.add_argument('--n_bins', type=int, default=3, help='number of bins to create')
    parser.add_argument('--bin_validation_metric', default='', help='check the mean and significance of a metric for every bin, like wer (leave empty if you do not want to do this)')

    # Training arguments
    parser.add_argument('--fairseq', dest='fairseq', action='store_true', help='train with fairseq framework')
    parser.set_defaults(fairseq=False)
    parser.add_argument('--fairseq_yaml', default='', help='path to base configuration YAML for fairseq framework')
    parser.add_argument('--flashlight', dest='flashlight', action='store_true', help='train with Flashlight framework')
    parser.set_defaults(flashlight=False)
    parser.add_argument('--flashlight_log', default='', help='path to the Flashlight log with WER and TER scores')
    parser.add_argument('--huggingface', dest='huggingface', action='store_true', help='score with HuggingFace framework')
    parser.set_defaults(huggingface=False)
    parser.add_argument('--save_curriculum_yaml', dest='save_curriculum_yaml', action='store_true', help='save generated curriculum learning YAMLs')
    parser.set_defaults(save_curriculum_yaml=False)
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataloading')
    parser.add_argument('--sampling_rate', type=int, default=16000, help='sampling rate to work with')
    return parser.parse_args()