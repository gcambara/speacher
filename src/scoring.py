"""Scoring functions for speech recognition curriculum learning."""
import collections
import os
import wget
import numpy as np
import pandas as pd
import torch
import jiwer
from .dataset import AudioDataset
from speechbrain.pretrained import EncoderDecoderASR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer

import time

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
        self.use_fairseq = args.fairseq
        self.use_flashlight = args.flashlight
        self.use_huggingface = args.huggingface

        self.flashlight_log = args.flashlight_log
        self.manifest = args.manifest
        self.transcripts_dir = os.path.join(args.out_dir, 'asr_transcripts')
        self.asr_metric = args.asr_metric
        self.sort_manifest = args.sort_manifest

        self.sampling_rate = args.sampling_rate

        if self.use_fairseq:
            self.model_path = self.get_model_path(args)
        elif self.use_huggingface:
            self.model_url = args.asr_download_model
            self.tokenizer_url = args.asr_tokenizer
            self.batch_size = args.batch_size
            self.num_workers = args.num_workers
            self.scoring_sorting = args.scoring_sorting

            self.hf_username, self.hf_modelname = self.model_url.split('/')

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

    def call_fairseq(self, df):
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
            measures = jiwer.compute_measures(ref, hyp)
            wer = measures['wer']*100.0

            assert (ref == df.loc[int(sample_id), 'tgt_text']), "The reference text indicated by the sample ID in the transcripts file does not match with the one stored in the dataset!"
            df.at[int(sample_id), 'wer'] = wer

        return df

    def call_huggingface(self, df):
        assert self.model_url != '', "Error! A model URL is needed for HuggingFace scoring, but --asr_download_model is empty"
        if self.tokenizer_url == '':
            print(f"Setting empty --tokenizer_url field identically to --asr_download_model: {self.model_url}")
            self.tokenizer_url = self.model_url

        if self.scoring_sorting == 'ascending':
            df = df.sort_values(by=['n_frames']).reset_index(drop=True)
        elif self.scoring_sorting == 'descending':
            df = df.sort_values(by=['n_frames'], ascending=False).reset_index(drop=True)
        elif self.scoring_sorting == '':
            pass
        else:
            raise NotImplementedError

        print(f"Preparing dataloader for manifest {self.manifest}...")
        dataset = AudioDataset(df)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=dataset.collater, num_workers=self.num_workers, pin_memory=True)

        if self.hf_username == 'facebook':
            print(f"Downloading tokenizer: {self.tokenizer_url}")
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(self.tokenizer_url)

            print(f"Downloading model: {self.model_url}")
            model = Wav2Vec2ForCTC.from_pretrained(self.model_url)
        elif self.hf_username == 'speechbrain':
            if torch.cuda.is_available():
                run_opts = {"device": "cuda"}
            else:
                run_opts = {"device": "cpu"}
            print(f"Downloading model: {self.model_url}")
            model = EncoderDecoderASR.from_hparams(source=self.model_url, 
                                                   run_opts=run_opts,
                                                   savedir=os.path.join('pretrained_models', self.hf_modelname))
        else:
            raise NotImplementedError

        model.eval()

        print("Scoring dataset...")
        df['wer'] = np.nan

        for batch in tqdm(dataloader):
            indexes, waveforms, transcripts, wav_lens = batch

            if self.hf_username == 'facebook':
                output_logits = model(waveforms.squeeze()).logits
                predicted_ids = torch.argmax(output_logits, dim=-1)
                pred_transcripts = tokenizer.batch_decode(predicted_ids)
            elif self.hf_username == 'speechbrain':
                waveforms = waveforms.squeeze()
                #waveforms = model.audio_normalizer(waveforms, self.sampling_rate)
                pred_transcripts = model.transcribe_batch(waveforms, wav_lens)[0]

            for index, ref in enumerate(transcripts):
                sample_id = indexes[index]
                ref = transcripts[index]
                pred = pred_transcripts[index]
                measures = jiwer.compute_measures(ref, pred)
                wer = measures['wer']*100.0
                assert (ref == df.loc[int(sample_id), 'tgt_text']), "The reference text indicated by the sample ID in the transcripts file does not match with the one stored in the dataset!"
                df.at[int(sample_id), 'wer'] = wer

        return df

    def flashlight_log_to_tsv(self, log, df):
        # The dataframe is sorted so it is faster to append new information.
        df = df.sort_values(by='path')

        eval_dict = {}
        with open(log, 'r') as f:
            for line in f.readlines():
                if line.startswith('[sample: '):
                    info = line.replace('[sample: ', '').strip().split(',')
                    sample_id = info[0]
                    wer = info[1].lstrip().rstrip().replace('WER: ', '').replace('%', '')
                    ter = info[2].lstrip().rstrip().replace('TER: ', '').replace('%', '')
                    path = sample_id + '.wav'

                    eval_dict[path] = (wer, ter)

        eval_dict = collections.OrderedDict(sorted(eval_dict.items()))
        paths = []
        wers = []
        ters = []
        for key, value in eval_dict.items():
            paths.append(key)
            wers.append(value[0])
            ters.append(value[1])

        if paths != df["path"].tolist():
            print("The order of the log and the tsv does not match! Going the slow way.")
            df["wer"] = np.nan
            df["ter"] = np.nan
            for key, value in eval_dict.items():
                df.at[df.index[df['path'] == key].tolist()[0], 'wer'] = value[0]
                df.at[df.index[df['path'] == key].tolist()[0], 'ter'] = value[1]
        else:
            # Since the log and the tsv are sorted, it is as easy as append the column, instead of
            # searching for the matching sample IDs and/or paths.
            df["wer"] = wers
            df["ter"] = ters

        # Sort back the dataframe to the original order.
        return df.sort_index()

    def __call__(self, df):
        if self.use_fairseq:
            df = self.call_fairseq(df)
        elif self.use_flashlight:
            df = self.flashlight_log_to_tsv(self.flashlight_log, df)
        elif self.use_huggingface:
            df = self.call_huggingface(df)
        else:
            assert "Flashlight, HuggingFace or fairseq mode must be set! There are not other frameworks implemented for ASR scoring at this moment."

        if self.sort_manifest:
            return df.sort_values(by=[self.asr_metric])
        else:
            return df

model_urls = {'s2t_transformer_s': 'https://dl.fbaipublicfiles.com/fairseq/s2t/librispeech_transformer_s.pt',
              's2t_transformer_m': 'https://dl.fbaipublicfiles.com/fairseq/s2t/librispeech_transformer_m.pt',
              's2t_transformer_l': 'https://dl.fbaipublicfiles.com/fairseq/s2t/librispeech_transformer_l.pt'}