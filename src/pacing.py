"""Pacing functions for speech recognition curriculum learning."""
import numpy as np
import pandas as pd

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

@PacingFunction.register_function('binning')
class Binning(PacingFunction):
    def __init__(self, args):
        super(Binning, self).__init__()
        self.bin_variable = args.bin_variable
        self.descending = args.descending  
        self.bin_method = args.bin_method
        self.n_bins = args.n_bins
        self.bin_validation_metric = args.bin_validation_metric

    #def __call__(self, df, descending=False):
    def __call__(self, df):
        labels = np.arange(self.n_bins)
        if self.bin_method == 'cut':
            bins = pd.cut(df[self.bin_variable], bins=self.n_bins, labels=labels)
        elif self.bin_method == 'qcut':
            bins = pd.qcut(df[self.bin_variable], q=self.n_bins, labels=labels)
        df['bin_label'] = bins

        df_list = []
        total_n_samples = 0
        for bin_label, bin_df in df.groupby('bin_label'):
            total_n_samples += len(bin_df)
            df_list.append(bin_df)

            print("--------------------------")
            print(f"Bin label = {bin_label}")
            print(f"Bin samples = {len(bin_df)} samples")

            if self.bin_validation_metric != '':
                mean_metric = bin_df[self.bin_validation_metric].mean()
                std_metric = bin_df[self.bin_validation_metric].std()
                sem_metric = std_metric / np.sqrt(len(bin_df))
                print(f"Bin mean {self.bin_validation_metric} = {mean_metric} (+/-) {sem_metric}")
        print("--------------------------")

        assert total_n_samples == len(df), f"Error! The total number of samples in the split bins is {total_n_samples}, but the total number of samples in the whole manifest is {len(df)}."
        
        if self.descending == True: 
            df_list.reverse()
            
        return df_list