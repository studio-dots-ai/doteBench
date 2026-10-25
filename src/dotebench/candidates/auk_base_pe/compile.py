"""Bind an official PE instruction to the executor's current audio."""

import hashlib

from dotebench.models.requests import NativeRequest


def compile_request(request, prepared, provenance):
    return NativeRequest(
        endpoint="/edit/pe",
        fields=tuple(
            {
                "prepared": prepared,
                "source_text": provenance["source_text"],
                "language": request.language,
                "metadata": {
                    "request_id": request.id,
                    "pe_cache": provenance,
                    "source_audio_sha256": hashlib.sha256(
                        request.source_audio.data
                    ).hexdigest(),
                },
            }.items()
        ),
        audio_field="audio",
        audio_encoding="file",
        response_encoding="audio_base64",
        selected_operation_index=None,
        scope="whole",
    )
