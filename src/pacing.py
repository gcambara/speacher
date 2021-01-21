"""Pacing functions for speech recognition curriculum learning."""

class PacingFunction():
    """Base class for pacing functions"""
    subclasses = {}

    def __init__(self):
        super(PacingFunction, self).__init__()

    @classmethod
    def register_function(cls, function_name):
        def decorator(subclass):
            cls.subclasses[function_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, args):
        if args.pacing_function not in cls.subclasses:
            raise ValueError('Bad pacing function name {}'.format(args.pacing_function))

        return cls.subclasses[args.pacing_function](args)

@PacingFunction.register_function('fixed_exponential')
class FixedExponentialPacing(PacingFunction):
    def __init__(self, args):
        super(FixedExponentialPacing, self).__init__()
        self.starting_percent = args.starting_percent
        self.increase = args.increase

    def __call__(self, df):
        total_samples = len(df)
        current_percent = self.starting_percent

        start_offset = 0 
        end_offset = int(current_percent * total_samples)

        assert (end_offset < total_samples), "The starting percent yields to a sub dataset size equal or bigger than the current dataset! Please, reduce it."

        df_list = [df[start_offset:end_offset]]
        while end_offset != total_samples:
            current_percent *= self.increase
            start_offset = end_offset
            end_offset = int(min(current_percent, 1) * total_samples)

            df_list.append(df[start_offset:end_offset])

        return df_list