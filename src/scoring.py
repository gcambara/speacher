"""Scoring functions for speech recognition curriculum learning."""

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