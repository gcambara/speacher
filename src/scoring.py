"""Scoring functions for speech recognition curriculum learning."""
import os
import wget

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

    def __call__(self, df):
        return df.sort_values(by=['n_frames'])

model_urls = {'s2t_transformer_s': 'https://dl.fbaipublicfiles.com/fairseq/s2t/librispeech_transformer_s.pt',
              's2t_transformer_m': 'https://dl.fbaipublicfiles.com/fairseq/s2t/librispeech_transformer_m.pt',
              's2t_transformer_l': 'https://dl.fbaipublicfiles.com/fairseq/s2t/librispeech_transformer_l.pt'}