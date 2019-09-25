# ASRbasic
A lightweight ASR system to compute phoneme errors on a corpus such as Timit.\
The intention of this software was to analyse (automatic) speech recognition for degraded input signals or degraded features, i.e. focusing on the confusion of phonemes at frame level. For this reason, the features are somewhat different from the more typical Mel-scale features. Adaption to other input features is straightforward.

Nonetheless, a basic (i.e. slow) Viterbi code is provided to calculate phoneme error rates. It achieves 19% PER on the Timit corpus, which is comparable to similar architectures, but slightly higher than  state-of-the-art architectures (including LSTMs, wider and deeper, or more features) or solutions that went through more training.

The main part are two GRU networks, one causal (srA1, 2 layers, 64 units), one bidirectional (srA2, 2 GRU layers, 128 units, some Dense layers). The input of srA2 are the class probabilities of srA1. The output of srA2 are the class probabilities of the previous, current, and next phoneme.

To get started with the code, look at main() in Timit_train_and_eval.py\
Prerequesites: Tensorflow, Keras, librosa, github.com/js2251/myfft

TimitData: Extract features, prepare labels, etc.\
srA1: Causal GRU NN. Input: acoustic features. Output: 40 class probabilities (39 phonemes + the glottal stop) per frame\
srA2: Bidirectional GRU NN. Input: Output of srA1. Output: 3*40 class probabilities for previous, current and next phoneme of the frame.\
TimitLanguage: Viterbi. Input: Output of srA2. Not parallel to be compatible with all OS, iPython, etc.
