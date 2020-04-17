"""Automated speech recognition in Rhasspy using Mozilla's DeepSpeech."""
import io
import logging
import time
import typing
import wave
from pathlib import Path

import deepspeech
import numpy as np
from rhasspyasr import Transcriber, Transcription, TranscriptionToken

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class DeepSpeechTranscriber(Transcriber):
    """Speech to text with deepspeech library."""

    def __init__(
        self,
        model_path: typing.Optional[Path] = None,
        language_model_path: typing.Optional[Path] = None,
        trie_path: typing.Optional[Path] = None,
        model: typing.Optional[deepspeech.Model] = None,
        beam_width: int = 500,
        lm_alpha: float = 0.75,
        lm_beta: float = 1.85,
    ):
        self.model = model
        self.model_path = model_path
        self.language_model_path = language_model_path
        self.trie_path = trie_path

        self.beam_width = beam_width
        self.lm_alpha = lm_alpha
        self.lm_beta = lm_beta

    def transcribe_wav(self, wav_bytes: bytes) -> typing.Optional[Transcription]:
        """Speech to text from WAV data."""
        self.maybe_load_model()
        assert self.model, "Model was not loaded"

        start_time = time.perf_counter()

        # Convert to raw numpy buffer
        with io.BytesIO(wav_bytes) as wav_io:
            with wave.open(wav_io) as wav_file:
                audio_bytes = wav_file.readframes(wav_file.getnframes())
                audio_buffer = np.frombuffer(audio_bytes, np.int16)

        metadata = self.model.sttWithMetadata(audio_buffer)
        end_time = time.perf_counter()

        if metadata:
            # Actual transcription
            text = ""

            # Individual tokens
            tokens: typing.List[TranscriptionToken] = []
            word = ""
            word_start_time = 0
            for index, item in enumerate(metadata.items):
                text += item.character

                if item.character != " ":
                    # Add to current word
                    word += item.character

                if item.character == " " or (index == (len(metadata.items) - 1)):
                    # Combine into single tokens
                    tokens.append(
                        TranscriptionToken(
                            token=word,
                            likelihood=1,
                            start_time=word_start_time,
                            end_time=item.start_time,
                        )
                    )

                    # Word break
                    word = ""
                    word_start_time = 0
                elif len(word) > 1:
                    word_start_time = item.start_time

            return Transcription(
                text=text,
                likelihood=metadata.confidence,
                transcribe_seconds=(end_time - start_time),
                wav_seconds=get_wav_duration(wav_bytes),
                tokens=tokens,
            )

        # Failure
        return None

    # -------------------------------------------------------------------------

    def transcribe_stream(
        self,
        audio_stream: typing.Iterable[bytes],
        sample_rate: int,
        sample_width: int,
        channels: int,
    ) -> typing.Optional[Transcription]:
        """Speech to text from an audio stream."""
        # No online streaming support.
        # Re-package as a WAV.
        with io.BytesIO() as wav_buffer:
            wav_file: wave.Wave_write = wave.open(wav_buffer, "wb")
            with wav_file:
                wav_file.setframerate(sample_rate)
                wav_file.setsampwidth(sample_width)
                wav_file.setnchannels(channels)

                for frame in audio_stream:
                    wav_file.writeframes(frame)

            return self.transcribe_wav(wav_buffer.getvalue())

    def stop(self):
        """Stop the transcriber."""

    def __repr__(self) -> str:
        return "DeepSpeechTranscriber(" f"model={self.model}" ")"

    def maybe_load_model(self):
        """Load DeepSpeech model if not already loaded."""
        if self.model:
            return

        assert self.model_path, "No model path"

        _LOGGER.debug(
            "Loading model from %s (beam width=%s)", self.model_path, self.beam_width
        )
        self.model = deepspeech.Model(str(self.model_path), self.beam_width)

        if (
            self.language_model_path
            and self.language_model_path.is_file()
            and self.trie_path
            and self.trie_path.is_file()
        ):
            _LOGGER.debug(
                "Enabling language model (lm=%s, trie=%s, lm_alpha=%s, lm_beta=%s)",
                self.language_model_path,
                self.trie_path,
                self.lm_alpha,
                self.lm_beta,
            )

            self.model.enableDecoderWithLM(
                str(self.language_model_path),
                str(self.trie_path),
                self.lm_alpha,
                self.lm_beta,
            )


# -----------------------------------------------------------------------------


def get_wav_duration(wav_bytes: bytes) -> float:
    """Return the real-time duration of a WAV file"""
    with io.BytesIO(wav_bytes) as wav_buffer:
        wav_file: wave.Wave_read = wave.open(wav_buffer, "rb")
        with wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / float(rate)
