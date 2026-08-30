"""Identity-blind character profiling experiments."""

from .corpus import CommentCorpus
from .pipeline import ExperimentRunner, ProfileBuilder
from .schema import CharacterSpec, Comment, Cue, PersonModel

__all__ = [
    "CharacterSpec",
    "Comment",
    "CommentCorpus",
    "Cue",
    "ExperimentRunner",
    "PersonModel",
    "ProfileBuilder",
]

__version__ = "0.1.0"

