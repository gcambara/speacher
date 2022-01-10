'''PyTorch base audio dataset.'''
from pathlib import Path
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from typing import List, Optional, Tuple, Union

class AudioDataset(Dataset):
    """Create a base audio Dataset.

    Args:
        df: Dataframe file containing the audio paths and transcripts.
    """

    _sample_id_column = 'id'
    _audio_path_column = 'audio'
    _text_path_column = 'tgt_text'
    _n_frames_column = 'n_frames'
    _speaker_column = 'speaker'

    def __init__(self, df):
        self.df = df
        self.num_samples = len(self.df)
        print(f"{self.num_samples} samples read from the tsvs.\n")

    def __getitem__(self, n: int) -> Tuple[int, Tensor, int, str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(n, waveform, sample_rate, sample_id, tgt_text)``
        """

        audio_path = self.df.iloc[n][self._audio_path_column]
        waveform, sample_rate = torchaudio.load(audio_path)

        sample_id = self.df.iloc[n][self._sample_id_column]
        tgt_text = self.df.iloc[n][self._text_path_column]

        return (n, waveform, sample_rate, sample_id, tgt_text)

    def __len__(self) -> int:
        return len(self.df)

    def collater(self, samples: List[Tuple[int, Tensor, int, str, str]]) -> Tuple:
        indexes = []
        waveforms = []
        labels = []
        for (index, waveform, _, _, utterance) in samples:
            indexes.append(index)
            waveforms.append(waveform.squeeze())
            labels.append(utterance)

        waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True).unsqueeze(1)

        return indexes, waveforms, labels
