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
        scorer_path: typing.Optional[Path] = None,
        model: typing.Optional[deepspeech.Model] = None,
        beam_width: typing.Optional[int] = None,
        lm_alpha: typing.Optional[float] = None,
        lm_beta: typing.Optional[float] = None,
    ):
        self.model = model
        self.model_path = model_path
        self.scorer_path = scorer_path

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

        wav_seconds = get_wav_duration(wav_bytes)
        transcribe_seconds = end_time - start_time

        return DeepSpeechTranscriber.metadata_to_transcription(
            metadata, wav_seconds, transcribe_seconds
        )

    # -------------------------------------------------------------------------

    def transcribe_stream(
        self,
        audio_stream: typing.Iterable[bytes],
        sample_rate: int,
        sample_width: int,
        channels: int,
    ) -> typing.Optional[Transcription]:
        """Speech to text from an audio stream."""
        self.maybe_load_model()
        assert self.model, "Model was not loaded"

        stream = self.model.createStream()

        start_time = time.perf_counter()
        num_frames = 0
        for chunk in audio_stream:
            if chunk:
                stream.feedAudioContent(np.frombuffer(chunk, dtype=np.int16))
                num_frames += len(chunk) // sample_width

        metadata = stream.finishStreamWithMetadata()
        end_time = time.perf_counter()

        wav_seconds = num_frames / sample_rate
        transcribe_seconds = end_time - start_time

        return DeepSpeechTranscriber.metadata_to_transcription(
            metadata, wav_seconds, transcribe_seconds
        )

    # -------------------------------------------------------------------------

    @staticmethod
    def metadata_to_transcription(
        metadata: typing.Optional[deepspeech.Metadata],
        wav_seconds: float,
        transcribe_seconds: float,
    ) -> typing.Optional[Transcription]:
        """Convert DeepSpeech metadata to Rhasspy Transcription"""

        if metadata:
            # Actual transcription
            text = ""

            # Individual tokens
            tokens: typing.List[TranscriptionToken] = []
            confidence = 1
            if metadata.transcripts:
                transcript = next(iter(metadata.transcripts))
                confidence = transcript.confidence
                for token in transcript.tokens:
                    text += token.text
                    if tokens:
                        # Previous token ends where current one starts
                        tokens[-1].end_time = token.start_time

                    tokens.append(
                        TranscriptionToken(
                            token=token.text,
                            likelihood=1,
                            start_time=token.start_time,
                            end_time=token.start_time,
                        )
                    )

            if tokens:
                # Set final token end time
                tokens[-1].end_time = wav_seconds

            return Transcription(
                text=text,
                likelihood=confidence,
                transcribe_seconds=transcribe_seconds,
                wav_seconds=wav_seconds,
                tokens=tokens,
            )

        # Failure
        return None

    def stop(self):
        """Stop the transcriber."""

    def __repr__(self) -> str:
        return "DeepSpeechTranscriber(" f"model={self.model}" ")"

    def maybe_load_model(self):
        """Load DeepSpeech model if not already loaded."""
        if self.model:
            return

        assert self.model_path, "No model path"

        _LOGGER.debug("Loading model from %s", self.model_path)
        self.model = deepspeech.Model(str(self.model_path))

        if self.scorer_path and self.scorer_path.is_file():
            _LOGGER.debug("Enabling scorer: %s)", self.scorer_path)

            self.model.enableExternalScorer(str(self.scorer_path))

        if self.beam_width is not None:
            _LOGGER.debug("Setting beam width to %s", self.beam_width)
            self.model.setBeamWidth(self.beam_width)

        if (self.lm_alpha is not None) and (self.lm_beta is not None):
            _LOGGER.debug(
                "Setting lm_alpha=%s, lm_beta=%s", self.lm_alpha, self.lm_beta
            )
            self.model.setScorerAlphaBeta(self.lm_alpha, self.lm_beta)


# -----------------------------------------------------------------------------


def get_wav_duration(wav_bytes: bytes) -> float:
    """Return the real-time duration of a WAV file"""
    with io.BytesIO(wav_bytes) as wav_buffer:
        wav_file: wave.Wave_read = wave.open(wav_buffer, "rb")
        with wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / float(rate)
