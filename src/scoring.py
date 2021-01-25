"""Scoring functions for speech recognition curriculum learning."""
import os
import wget
import numpy as np
from fairseq import scoring

class ScoringFunction():
    """Base class for scoring functions"""
    subclasses = {}

    def __init__(self):
        super(ScoringFunction, self).__init__()

    @classmethod
    def register_function(cls, function_name):
        def decorator(subclass):
            cls.subclasses[function_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, args):
        if args.scoring_function not in cls.subclasses:
            raise ValueError('Bad scoring function name {}'.format(args.scoring_function))

        return cls.subclasses[args.scoring_function](args)

@ScoringFunction.register_function('n_frames')
class NFrames(ScoringFunction):
    def __init__(self, args):
        super(NFrames, self).__init__()

    def __call__(self, df):
        return df.sort_values(by=['n_frames'])

@ScoringFunction.register_function('asr')
class ASR(ScoringFunction):
    def __init__(self, args):
        super(ASR, self).__init__()
        self.model_path = self.get_model_path(args)
        self.use_fairseq = args.fairseq
        self.manifest = args.manifest
        self.transcripts_dir = os.path.join(args.out_dir, 'asr_transcripts')
        self.scorer = scoring.build_scorer('wer', None)

    def get_model_path(self, args):
        if args.asr_model == '':
            assert (args.asr_download_model != ''), 'Please, specify a valid path to an ASR checkpoint with --asr_model, or indicate the name of a pretrained model to be downloaded with --asr_download_model'
            model_dir = os.path.join(args.asr_model_zoo, args.asr_download_model)
            os.makedirs(model_dir, exist_ok=True)
            return self.download_model(args.asr_download_model, model_dir)
        else:
            return args.asr_model

    def download_model(self, model_name, model_dir):
        url = model_urls[model_name]
        filename = os.path.basename(url)
        model_path = os.path.join(model_dir, filename)

        if not os.path.exists(model_path):
            wget.download(url, model_dir)

        return os.path.abspath(model_path)

    def compute_wer(self, ref, hyp):
        try:
            import editdistance as ed
        except ImportError:
            raise ImportError("Please install editdistance to use WER scorer")

    def __call__(self, df):
        assert (self.use_fairseq), "Fairseq mode must be set! There are not other frameworks implemented for ASR scoring at this moment."

        data_dir, manifest_name = os.path.split(self.manifest)
        manifest_name = manifest_name.replace('.tsv', '')

        inference_command = f"fairseq-generate {data_dir} --config-yaml config.yaml --gen-subset {manifest_name} --task speech_to_text --path {self.model_path} --max-tokens 50000 --beam 5 --scoring wer --results-path {self.transcripts_dir}"
        os.system(inference_command)

        # Currently, fairseq only reports the total score for the entire dataset inference.
        # Therefore, we must take the result transcripts and recompute the score for every utterance.
        # First, parse the obtained results.
        transcripts_path = os.path.join(self.transcripts_dir, f"generate-{manifest_name}.txt")
        eval_dict = {}
        with open(transcripts_path, 'r') as f:
            curr_sample_id = ""
            for line in f.readlines():
                if line.startswith('T-'):
                    curr_sample_id = line.split('\t')[0].replace('T-', '')
                    ref = line.split('\t')[1].strip()
                elif line.startswith('D-'):
                    assert (curr_sample_id == line.split('\t')[0].replace('D-', '')), "Reference transcript is not followed by a detokenized hypothesis!"
                    hyp = line.split('\t')[2].strip()
                    eval_dict[curr_sample_id] = (ref, hyp)

        # Now, compute and append the score for every utterance at the dataframe.
        df['wer'] = np.nan
        for sample_id, (ref, hyp) in eval_dict.items():
            self.scorer.add_string(ref, hyp)
            wer = self.scorer.result_string().split('WER: ')[1]
            self.scorer.reset()

            assert (ref == df.loc[int(sample_id), 'tgt_text']), "The reference text indicated by the sample ID in the transcripts file does not match with the one stored in the dataset!"
            df.at[int(sample_id), 'wer'] = wer

        return df.sort_values(by=['wer'])

model_urls = {'s2t_transformer_s': 'https://dl.fbaipublicfiles.com/fairseq/s2t/librispeech_transformer_s.pt',
              's2t_transformer_m': 'https://dl.fbaipublicfiles.com/fairseq/s2t/librispeech_transformer_m.pt',
              's2t_transformer_l': 'https://dl.fbaipublicfiles.com/fairseq/s2t/librispeech_transformer_l.pt'}