# Fase 1: Preparação do Ambiente Python e Instalação de Dependências Avançadas

Esta fase detalha a configuração do seu ambiente Python e a instalação de todas as bibliotecas necessárias para suportar a arquitetura de dois modelos (Phi-3.5 para pesquisa e Qwen3 para automação), incluindo funcionalidades avançadas de manipulação de documentos e automação de navegador.

## 1. Configuração do Ambiente Virtual Python

É crucial utilizar um ambiente virtual Python para isolar as dependências do seu projeto, evitando conflitos com outras instalações Python no seu sistema.

1.  **Crie um ambiente virtual** (se ainda não o fez):
    ```bash
    python -m venv ia_local_env
    ```
2.  **Ative o ambiente virtual**:
    *   **No Windows (PowerShell ou CMD)**:
        ```bash
        .\ia_local_env\Scripts\activate
        ```
    *   **No Linux/macOS (Terminal)**:
        ```bash
        source ia_local_env/bin/activate
        ```

## 2. Instalação de Bibliotecas Python Essenciais

Com o ambiente virtual ativado, instale todas as bibliotecas necessárias. Isso inclui as ferramentas para automação de navegador, integração com Ollama, manipulação básica de arquivos, criação de PDFs, e as bibliotecas avançadas para leitura e análise de PDFs.

```bash
pip install browser-use langchain-ollama playwright PyPDF2 fpdf2 pandas pymupdf pypdf pdfplumber ocrmypdf unstructured
playwright install
```

### Detalhamento das Bibliotecas:

*   `browser-use`: A biblioteca principal para permitir que seu agente de IA interaja com navegadores web.
*   `langchain-ollama`: Permite a integração do framework LangChain (utilizado pelo `browser-use` para orquestração) com o Ollama.
*   `playwright`: O motor de automação de navegador que o `browser-use` utiliza. O comando `playwright install` baixa os drivers dos navegadores (Chromium, Firefox, WebKit).
*   `PyPDF2`: Para leitura e manipulação básica de arquivos PDF.
*   `fpdf2`: Para a criação de arquivos PDF a partir de conteúdo textual.
*   `pandas`: Essencial para manipulação e análise de dados estruturados, útil para extrair informações de tabelas em PDFs ou páginas web.
*   `pymupdf` (fitz): Uma biblioteca poderosa para extração rápida e precisa de texto, imagens e tabelas de PDFs. Altamente recomendada para análise aprofundada de PDFs.
*   `pypdf`: Complementar ao PyPDF2, oferece funcionalidades robustas para manipulação de PDFs.
*   `pdfplumber`: Especialmente útil para extrair dados de PDFs onde o layout é importante, como tabelas e texto com formatação específica.
*   `ocrmypdf`: Uma ferramenta de linha de comando que adiciona uma camada de texto pesquisável a PDFs escaneados (baseados em imagem). **Requer a instalação de Tesseract OCR no sistema operacional.**
*   `unstructured`: Para particionamento semântico de diversos formatos de arquivo (PDFs, documentos Office, HTML), facilitando a extração de seções específicas ou dados estruturados para o LLM.

### Instalação do Tesseract OCR (para `ocrmypdf`)

Para que o `ocrmypdf` funcione, você precisará instalar o Tesseract OCR no seu sistema operacional. As instruções variam:

*   **No Windows**: Baixe o instalador do Tesseract OCR em [https://tesseract-ocr.github.io/tessdoc/Downloads.html](https://tesseract-ocr.github.io/tessdoc/Downloads.html) e siga as instruções. Certifique-se de adicionar o Tesseract ao PATH do sistema.
*   **No Linux (Ubuntu/Debian)**:
    ```bash
    sudo apt update
    sudo apt install tesseract-ocr
    sudo apt install tesseract-ocr-por # Para suporte a português
    ```
*   **No macOS (Homebrew)**:
    ```bash
    brew install tesseract
    brew install tesseract-langfiles # Para idiomas adicionais
    ```

Após a conclusão desta fase, seu ambiente Python estará pronto com todas as ferramentas necessárias para construir seu agente de IA avançado.
