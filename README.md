# speacher
speacher (speech teacher) is a curriculum learning toolkit for speech recognition, which enables research and development of curriculums for training speech recognition models. This framework takes dataset manifests in tab-separated values (.tsv) files as input, and generates an output folder with sets sub-sampled according to difficulty criteria. A variety of difficulty scoring and pacing functions are implemented. Besides, it can be used to assess speech synthesis samples with different quality metrics.

This framework has been designed to work with other speech recognition frameworks, like [fairseq](https://github.com/pytorch/fairseq) and [Flashlight](https://github.com/facebookresearch/flashlight/), which we recommend to install to allow speacher's full functionality. Fairseq is very easy to install with pip, and Flashlight is adviced to be installed via Docker, as they provide with the Docker image at their repo.

## Install the requirements

Once you have installed [fairseq](https://github.com/pytorch/fairseq), simply run the following command:
````
pip install -r requirements.txt
````

## Scoring a dataset
A dataset can be scored and sorted in order of difficulty with different metrics, like by sample lengths, or an error metric like WER or TER. For evaluation of such metrics with a pretrained ASR, three frameworks can be used: fairseq, HuggingFace and Flashlight.

### Using HuggingFace
If you want to use an ASR model from HuggingFace repository, specify the --huggingface argument. Also, introduce the name of the model so it can be downloaded from the model zoo:
```
python score_dataset.py --manifest <path_to_data_manifest> --out_dir <path_to_output_manifest> --sort_manifest --scoring_function asr --asr_metric wer --asr_download_model facebook/wav2vec2-base-960h --huggingface --batch_size 4
```

Inference can be speed up by sorting by length in order to reduce padding, using "n_frames" column in the manifest TSV. Use --scoring_sorting function:
```
python score_dataset.py --manifest <path_to_data_manifest> --out_dir <path_to_output_manifest> --sort_manifest --scoring_function asr --asr_metric wer --asr_download_model facebook/wav2vec2-base-960h --huggingface --batch_size 4 --scoring_sorting ascending
```

Currently, models coming from two HuggingFace's users are supported: facebook and [speechbrain](https://github.com/speechbrain/speechbrain). Find here an example of using a speechbrain model:
```
python score_dataset.py --manifest <path_to_data_manifest> --out_dir <path_to_output_manifest> --sort_manifest --scoring_function asr --asr_metric wer --asr_download_model speechbrain/asr-wav2vec2-commonvoice-en --huggingface --batch_size 4 --scoring_sorting ascending
```

Other SpeechBrain models can be found in the HuggingFace [repo](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=downloads&search=speechbrain).

### Using fairseq
Use --fairseq option, to select fairseq toolkit. With it, you will need to set the path to a pretrained model, and also a data manifest TSV with the paths to every audio sample. The fairseq framework will be called to assess such samples:
```
python score_dataset.py --manifest <path_to_data_manifest> --out_dir <path_to_output_manifest> --sort_manifest --scoring_function asr --asr_metric wer --asr_model <path_to_pretrained_checkpoint> --fairseq
```

You can also leave the --asr_model argument empty, and specify the name of a downloadable model. This way, the script will download a pretrained model from fairseq's repo, and use it for evaluation. For instance, for downloading and using "s2t_transformer_s" from the model zoo:

```
python score_dataset.py --manifest <path_to_data_manifest> --out_dir <path_to_output_manifest> --sort_manifest --scoring_function asr --asr_metric wer --asr_download_model s2t_transformer_s --fairseq
```

### Using Flashlight
On the other hand, calling Flashlight for usage is not currently supported. However, you can still pass in a log from a Flashlight testing execution, along with the data manifest TSV, and speacher will return the TSV sorted by the WER/TER in the Flashlight log. Simply type the following command:
```
python score_dataset.py --manifest <path_to_data_manifest> --out_dir <path_to_output_manifest> --scoring_function asr --asr_metric wer --flashlight --flashlight_log <path_to_test_log> --sort_manifest
```

## Training with a curriculum

Once you have a manifest sorted by scored order of difficulty, you can run a fairseq training with curriculum learning, using the following command:
```
python train_curriculum.py --manifest <path_to_sorted_manifest> --out_dir <path_to_saved_checkpoints> --starting_percent 0.04 --fairseq --fairseq_yaml <path_to_base_config_yaml> --save_curriculum_yaml
```
