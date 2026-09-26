"""Evaluation capability contract."""

from abc import ABC, abstractmethod
from typing import Sequence

from dotebench.domain import EvaluationRequest, EvaluationResult

__all__ = ["EvaluationRequest", "EvaluationResult", "EvaluationProtocol"]


class EvaluationProtocol(ABC):
    @abstractmethod
    def identity(self) -> dict:
        """Describe this prepared evaluator for the runner's execution snapshot."""

    @abstractmethod
    def prepare(self) -> None: ...

    @abstractmethod
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult: ...

    @abstractmethod
    def aggregate(self, results: Sequence[EvaluationResult]) -> dict: ...

    @abstractmethod
    def close(self) -> None: ...
