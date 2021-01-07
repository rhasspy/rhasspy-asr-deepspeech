"""Methods for training DeepSpeech language model."""
import logging
import shutil
import subprocess
import tempfile
import typing
from pathlib import Path

import networkx as nx
import rhasspynlu

_LOGGER = logging.getLogger("rhasspyasr_deepspeech")


def train(
    graph: nx.DiGraph,
    language_model: typing.Union[str, Path],
    scorer_path: typing.Union[str, Path],
    alphabet_path: typing.Union[str, Path],
    language_model_fst: typing.Optional[typing.Union[str, Path]] = None,
    base_language_model_fst: typing.Optional[typing.Union[str, Path]] = None,
    base_language_model_weight: typing.Optional[float] = None,
    mixed_language_model_fst: typing.Optional[typing.Union[str, Path]] = None,
    balance_counts: bool = True,
):
    """Re-generates language model and scorer from intent graph"""
    # Language model mixing
    base_fst_weight = None
    if (
        (base_language_model_fst is not None)
        and (base_language_model_weight is not None)
        and (base_language_model_weight > 0)
    ):
        base_fst_weight = (base_language_model_fst, base_language_model_weight)

    # Begin training
    with tempfile.NamedTemporaryFile(mode="w+") as arpa_file:
        with tempfile.NamedTemporaryFile(mode="w+") as vocab_file:
            # 1. Create language model
            _LOGGER.debug("Converting to ARPA language model")
            rhasspynlu.arpa_lm.graph_to_arpa(
                graph,
                arpa_file.name,
                model_path=language_model_fst,
                base_fst_weight=base_fst_weight,
                merge_path=mixed_language_model_fst,
                vocab_path=vocab_file.name,
            )

            arpa_file.seek(0)
            vocab_file.seek(0)

            with tempfile.NamedTemporaryFile(mode="wb+") as lm_file:
                # 2. Convert to binary language model
                arpa_to_binary(arpa_file.name, lm_file.name)

                lm_file.seek(0)

                with tempfile.NamedTemporaryFile(mode="wb+") as scorer_file:
                    # 3. Generate scorer
                    make_scorer(
                        alphabet_path, lm_file.name, vocab_file.name, scorer_file.name
                    )

                    # Copy over actual files
                    lm_file.seek(0)
                    shutil.copy(lm_file.name, language_model)
                    _LOGGER.debug("Wrote binary language model to %s", language_model)

                    scorer_file.seek(0)
                    shutil.copy(scorer_file.name, scorer_path)
                    _LOGGER.debug("Wrote scorer to %s", scorer_path)


def arpa_to_binary(
    arpa_path: typing.Union[str, Path], binary_lm_path: typing.Union[str, Path]
):
    """Convert ARPA language model to binary format using kenlm."""
    # NOTE: Using -i because other LM tools mistakenly produce positive log
    # probabilities. This option sets those to 0.
    binary_command = [
        "build_binary",
        "-s",
        "-i",
        "-a",
        "255",
        "-q",
        "8",
        "-v",
        "trie",
        str(arpa_path),
        str(binary_lm_path),
    ]
    _LOGGER.debug(binary_command)
    subprocess.check_call(binary_command)


def make_scorer(
    alphabet_path: typing.Union[str, Path],
    binary_lm_path: typing.Union[str, Path],
    vocab_path: typing.Union[str, Path],
    scorer_path: typing.Union[str, Path],
    default_alpha: float = 0.931289039105002,
    default_beta: float = 1.1834137581510284,
):
    """Generate scorer using Mozilla native-client tool."""
    scorer_command = [
        "generate_scorer_package",
        "--alphabet",
        str(alphabet_path),
        "--lm",
        str(binary_lm_path),
        "--vocab",
        str(vocab_path),
        "--package",
        str(scorer_path),
        "--default_alpha",
        str(default_alpha),
        "--default_beta",
        str(default_beta),
    ]

    _LOGGER.debug(scorer_command)
    subprocess.check_call(scorer_command)
