# Rhasspy ASR DeepSpeech

Rhasspy ASR library that uses Mozilla's DeepSpeech 0.6.

## Requirements

* Python 3.7
* [Mozilla DeepSpeech 0.6.1](https://github.com/mozilla/DeepSpeech/releases/tag/v0.6.1)
* `generate_trie` in `$PATH` from [native client](https://github.com/mozilla/DeepSpeech/releases/download/v0.6.1/native_client.amd64.cpu.linux.tar.xz)
* `build_binary` in `$PATH` from [KenLM](https://github.com/kpu/kenlm)
    * [Pre-built binaries](https://github.com/synesthesiam/prebuilt-apps)
    
## Installation

```bash
$ git clone https://github.com/rhasspy/rhasspy-asr-deepspeech
$ cd rhasspy-asr-deepspeech
$ ./configure
$ make
$ make install
```

## Deployment

```bash
$ make dist
```

See `dist/` directory for `.tar.gz` file.

## Running

```bash
$ bin/rhasspy-asr-deepspeech <ARGS>
```
