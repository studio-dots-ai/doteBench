"""AuK Base + PE using the shared compiler executor and native transport."""

from dotebench.compilers.sequential import SequentialCompiler
from dotebench.models.native import NativeCandidate

from .cache_reader import InstructionCache, require_instruction
from .compilers.one_take import OneTakeCompiler


class AuKBasePE(NativeCandidate):
    candidate_name = "auk_base_pe"

    def __init__(self, *, instruction_cache_dir, compiler="one_take", **kwargs):
        if compiler not in {"one_take", "sequential"}:
            raise ValueError("compiler must be one_take or sequential")
        one_take = OneTakeCompiler()
        super().__init__(
            compiler=(
                one_take if compiler == "one_take" else SequentialCompiler(one_take)
            ),
            **kwargs,
        )
        self.instruction_cache_dir = instruction_cache_dir
        self.instruction_cache = None

    def prepare(self):
        self.instruction_cache = InstructionCache(self.instruction_cache_dir)
        super().prepare()

    def resolve_instruction(self, item):
        record, provenance = self.instruction_cache.lookup(item)
        provenance = {
            **provenance,
            "identity": self.instruction_cache.identity,
            "source_text": item["source_text"],
        }
        self.last_call_metadata = {
            "pe_cache": provenance,
            "pe_status": record["status"],
        }
        return require_instruction(record), provenance

    def is_candidate_failure(self, detail):
        return (
            isinstance(detail, dict) and detail.get("error_code") == "candidate_error"
        ) or super().is_candidate_failure(detail)
