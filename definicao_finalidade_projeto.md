# Documento de Definição e Finalidade do Projeto: IA Local de Automação e Pesquisa

## 1. Visão Geral

O projeto consiste no desenvolvimento e implementação de uma infraestrutura de Inteligência Artificial (IA) local, projetada para operar de forma autônoma e eficiente em um ambiente de hardware específico. A finalidade central é criar um assistente virtual poderoso que combina capacidades de **pesquisa aprofundada** com **automação de tarefas práticas**, garantindo total privacidade e gratuidade através do uso de modelos de código aberto.

## 2. Objetivos Principais

*   **Automação de Navegador**: Capacitar a IA para interagir com a web, realizar pesquisas, extrair dados e operar outros serviços online através de automação determinística.
*   **Manipulação de Documentos**: Permitir que a IA leia, analise e crie arquivos em diversos formatos, com foco especial em PDFs (incluindo documentos escaneados via OCR).
*   **Pesquisa Especializada e Aprendizado**: Utilizar modelos de alta capacidade para sintetizar informações complexas, auxiliando no aprendizado de temas técnicos como engenharia e tecnologia.
*   **Memória Persistente**: Garantir que a IA mantenha o contexto de interações passadas, permitindo um aprendizado contínuo sobre as preferências e necessidades do usuário.
*   **Privacidade e Independência**: Executar todo o processamento localmente, sem dependência de APIs pagas ou envio de dados para a nuvem.

## 3. Estratégia Técnica: Arquitetura de Dois Modelos

Para otimizar o desempenho no hardware disponível (GTX 1060 6GB VRAM / 16GB RAM), o projeto adota uma estratégia de modelos especializados:

| Modelo | Papel | Modo de Operação | Finalidade |
|---|---|---|---|
| **Phi-3.5-medium (14B)** | **O Pesquisador** | CPU / RAM | Análise profunda, síntese de documentos extensos e raciocínio lógico complexo. |
| **Qwen3 4B Instruct** | **O Automador** | GPU / VRAM | Execução rápida de comandos, automação de navegador e tarefas de resposta imediata. |

## 4. Funcionalidades Implementadas

*   **Agente Orquestrador**: Um script central que decide qual modelo utilizar com base na complexidade da tarefa.
*   **Ferramentas Personalizadas**:
    *   `create_pdf_file`: Geração de relatórios e documentos em PDF.
    *   `read_pdf_content`: Extração de texto de documentos PDF locais.
    *   `calculate`: Execução de operações matemáticas precisas.
    *   `navigate_and_get_title`: Automação de navegação web básica e avançada.
*   **Sistema de Memória**: Gerenciamento de histórico em formato JSON para persistência de contexto.

## 5. Público-Alvo e Casos de Uso

Este projeto é destinado ao uso pessoal para:
*   **Estudantes e Profissionais**: Que buscam auxílio no aprendizado técnico e pesquisa acadêmica.
*   **Desenvolvedores**: Que desejam automatizar fluxos de trabalho repetitivos no navegador.
*   **Entusiastas de IA**: Que priorizam a soberania de dados e o uso de tecnologias open-source.

## 6. Conclusão

A finalidade deste projeto não é apenas criar uma ferramenta de automação, mas sim um ecossistema de IA local inteligente, resiliente e adaptável. Ao separar a inteligência de pesquisa da velocidade de automação, o sistema maximiza o potencial do hardware do usuário, entregando uma experiência de assistência virtual de nível profissional.

---
**Data de Definição**: 15 de Junho de 2026
**Autor**: Manus AI (em colaboração com o Usuário)
