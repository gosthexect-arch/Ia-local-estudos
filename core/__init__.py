"""Núcleo do app de chat local com VLM (llama.cpp).

Os módulos `config`, `hardware`, `gguf_info` e `offload` usam apenas a
biblioteca padrão: o `start.bat` os executa antes de instalar as dependências
para escolher a build correta do llama-cpp-python (CUDA / Vulkan / CPU).
"""

__version__ = "1.0.0"
