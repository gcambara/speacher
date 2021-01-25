# speacher
speacher (speech teacher) is a curriculum learning toolkit for speech recognition, which enables research and development of curriculums for training speech recognition models. This framework takes dataset manifests in tab-separated values (.tsv) files as input, and generates an output folder with sets sub-sampled according to difficulty criteria. A variety of difficulty scoring and pacing functions are implemented. Besides, it can be used to assess speech synthesis samples with different quality metrics.

This framework has been designed to work with other speech recognition frameworks, like [fairseq](https://github.com/pytorch/fairseq) and [Flashlight](https://github.com/facebookresearch/flashlight/), which we recommend to install to allow speacher's full functionality. Fairseq is very easy to install with pip, and Flashlight is adviced to be installed via Docker, as they provide with the Docker image at their repo.

