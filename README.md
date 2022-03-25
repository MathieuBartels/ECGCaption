# Learning to Automatically Generate Accurate ECG Captions

cd into the EcgCaptionGenerator

```
$ cd EcgCaptionGenerator
```

Use poetry to install all requirements.

In EcgCaptionGenerator/utils/pycocoevalcap requirers the installation of https://github.com/salaniz/pycocoevalcap so also a need to download https://stanfordnlp.github.io/CoreNLP/index.html

As the data used in the research is private data owned by the patients, a public dataset is not available. To make the code still work we created a FakeDataset, that simulates the data we get ( waveforms and topics and words )

run python main_PKEL.py, main_PKT.py, main_PKTransformer.py, main_Ecgnet_LSTM.py to train a model.

in gen_results.ipynb, the trained models are used to generate descriptions for the test dataset, which are scored using the evaluation metrics.
