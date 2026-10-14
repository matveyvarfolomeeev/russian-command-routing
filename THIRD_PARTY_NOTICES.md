# Third-party models and runtimes

The repository license applies to original project code, reports and authored fictional data. It does not replace licenses on downloaded models, native implementations, Python packages or Qt. Weights and upstream code checkouts are not bundled.

Pinned identifiers are recorded in [results/source-models.json](results/source-models.json). Consult each source's notices before downloading or redistributing its artifacts.

| Dependency | Pinned source |
| --- | --- |
| E5-small | [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3) |
| Laya multilingual | [convaiinnovations/laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual/tree/1720e3e3357cfe1e281542e223f8273b0890ca34) |
| Kev-0.8B | [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b/tree/bf75a6a8848ea6960ff2ed108d9ed44c2941174f) |
| Kev runtime | [jaredpalmer/kev](https://github.com/jaredpalmer/kev/tree/fe64b1274ea7f80d4095866df90666abb03e9cf6) |
| Kev base | [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68) |
| Nano-Jev | [sdmlai/nano-jev](https://huggingface.co/sdmlai/nano-jev/tree/b9f855e8253f177d162b50807c99d80507ec1e61), [native code](https://github.com/shubham10divakar/nano-jev/tree/25b4655ac129c7151eb39d2ed1d15227f93f1ba6) |
| RuModernBERT | [ARGA100/ru-reranker-modernbert-small](https://huggingface.co/ARGA100/ru-reranker-modernbert-small/tree/8d4ea05d7c793bc812879ca18e7e310ac4cb228f) |
| RuBERT-tiny2 | [cointegrated/rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2/tree/e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae) |

Upstream model-card measurements are distinct from this project's results. The trained Kev condition uses unmerged adapters and FP32 inference; upstream serving defaults should not be substituted silently.

Whisper Large V3 transcription was accessed through Groq. Existing transcripts can be evaluated offline; recreating audio/transcriptions requires the stated TTS/ASR services and a key.

The demo depends on PySide6/Qt and Windows APIs. Python versions are pinned in [requirements-dev.lock](requirements-dev.lock); GPU/native-model dependencies are documented with each historical training protocol.
