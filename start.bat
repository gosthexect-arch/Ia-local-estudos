@echo off
setlocal EnableExtensions EnableDelayedExpansion
title IA Local - Chat VLM
cd /d "%~dp0"

rem ======================================================================
rem  IA Local - unico ponto de entrada do app.
rem   1) Pede a chave do Tavily (somente na primeira instalacao) -> .env
rem   2) Encontra/instala o Python e cria o ambiente virtual (.venv)
rem   3) Escaneia o hardware (GPU/CPU/RAM) e escolhe a build do llama.cpp
rem   4) Instala o llama-cpp-python acelerado + requirements.txt
rem   5) Calcula threads e camadas de GPU ideais e abre a interface
rem  (Arquivo propositalmente sem acentos: o cmd.exe le .bat em codepage OEM.)
rem ======================================================================

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "VENV=%~dp0.venv"
set "PY=%VENV%\Scripts\python.exe"
set "LLAMA_SPEC=llama-cpp-python>=0.3.26,<0.4"
set "WHL_BASE=https://abetlen.github.io/llama-cpp-python/whl"
set "TMPF=%TEMP%\ia_local_%RANDOM%.txt"

echo.
echo  ==============================================================
echo     IA Local - Chat com VLM (llama.cpp) + Tavily + Terminal
echo  ==============================================================
echo.

rem ---------------------------------------------------------------- 1
call :ensure_tavily_key || goto :fail

rem ---------------------------------------------------------------- 2
if not exist "%PY%" (
    call :find_python
    if not defined SYS_PY call :install_python
    if not defined SYS_PY goto :fail_python
    echo [1/5] Criando ambiente virtual em .venv com !SYS_PY! ...
    !SYS_PY! -m venv "%VENV%"
    if errorlevel 1 goto :fail
    "%PY%" -m pip install --upgrade pip wheel
    if errorlevel 1 goto :fail
) else (
    echo [1/5] Ambiente virtual encontrado.
)

rem ---------------------------------------------------------------- 3
echo [2/5] Escaneando hardware para escolher a build do llama.cpp...
set "BACKENDS=cpu"
"%PY%" -m core.hardware --backend-order > "%TMPF%" 2>nul
if exist "%TMPF%" set /p BACKENDS=<"%TMPF%"
echo       Ordem de tentativa: !BACKENDS!

rem ---------------------------------------------------------------- 4
set "INSTALLED="
set "PREV_CANDS="
if exist "%VENV%\llama_backend.txt" set /p INSTALLED=<"%VENV%\llama_backend.txt"
if exist "%VENV%\llama_candidates.txt" set /p PREV_CANDS=<"%VENV%\llama_candidates.txt"
if defined INSTALLED if "!PREV_CANDS!"=="!BACKENDS!" (
    "%PY%" -m core.hardware --verify-llama !INSTALLED! >nul 2>&1
    if not errorlevel 1 (
        echo [3/5] llama-cpp-python [!INSTALLED!] ja instalado.
        goto :llama_ok
    )
)
echo [3/5] Instalando llama-cpp-python acelerado para este computador...
set "LLAMA_OK="
for %%B in (!BACKENDS!) do (
    if not defined LLAMA_OK call :install_llama %%B
)
if not defined LLAMA_OK goto :fail_llama
> "%VENV%\llama_candidates.txt" echo !BACKENDS!
:llama_ok
set "BEST="
set "CUR="
for /f "tokens=1" %%B in ("!BACKENDS!") do set "BEST=%%B"
if exist "%VENV%\llama_backend.txt" set /p CUR=<"%VENV%\llama_backend.txt"
if not "!CUR!"=="!BEST!" (
    echo       Aviso: usando a build "!CUR!" porque a preferida "!BEST!" falhou.
    echo       Para tentar de novo, apague o arquivo .venv\llama_backend.txt
)

echo [4/5] Verificando dependencias do requirements.txt...
set "REQ_HASH="
for /f "delims=" %%H in ('certutil -hashfile requirements.txt SHA256 ^| findstr /v ":"') do (
    if not defined REQ_HASH set "REQ_HASH=%%H"
)
set "OLD_HASH="
if exist "%VENV%\requirements.sha256" set /p OLD_HASH=<"%VENV%\requirements.sha256"
if not defined REQ_HASH set "REQ_HASH=sempre"
if "!REQ_HASH!"=="!OLD_HASH!" (
    echo       Dependencias ja instaladas.
) else (
    "%PY%" -m pip install -r requirements.txt --prefer-binary
    if errorlevel 1 goto :fail
    > "%VENV%\requirements.sha256" echo !REQ_HASH!
)

rem ---------------------------------------------------------------- 5
echo [5/5] Calculando threads e camadas de GPU ideais...
"%PY%" -m core.hardware
if errorlevel 1 echo       (aviso: scan de hardware incompleto; o app recalcula ao carregar o modelo)
del "%TMPF%" >nul 2>&1
echo.
echo  Abrindo a interface no navegador... (feche esta janela para encerrar o app)
echo.
"%PY%" app.py
if errorlevel 1 goto :fail
goto :eof


rem ======================================================================
rem  Sub-rotinas
rem ======================================================================

:ensure_tavily_key
if exist ".env" (
    findstr /r /c:"^TAVILY_API_KEY=..*" ".env" >nul 2>&1
    if not errorlevel 1 exit /b 0
)
echo  Primeira instalacao: a IA precisa de uma chave da API do Tavily para pesquisar na web.
echo  Crie uma chave gratuita em https://app.tavily.com  (formato: tvly-...)
echo.
:ask_key
set "TAVILY_KEY="
set /p "TAVILY_KEY=  Cole sua chave do Tavily e pressione Enter: "
if not defined TAVILY_KEY (
    echo  A chave e obrigatoria.
    goto :ask_key
)
set "TAVILY_KEY=!TAVILY_KEY: =!"
if not defined TAVILY_KEY goto :ask_key
if /i not "!TAVILY_KEY:~0,5!"=="tvly-" echo  Aviso: chaves do Tavily normalmente comecam com "tvly-". Salvando mesmo assim.
if exist ".env" >> ".env" echo(
>> ".env" echo TAVILY_API_KEY=!TAVILY_KEY!
echo  Chave salva em .env (nao sera pedida novamente).
echo.
exit /b 0


:find_python
set "SYS_PY="
for %%V in (3.12 3.11 3.13 3.10) do (
    if not defined SYS_PY (
        py -%%V -c "import sys" >nul 2>&1
        if not errorlevel 1 set "SYS_PY=py -%%V"
    )
)
if defined SYS_PY exit /b 0
for %%P in (python python3) do (
    if not defined SYS_PY (
        %%P -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1
        if not errorlevel 1 set "SYS_PY=%%P"
    )
)
if defined SYS_PY exit /b 0
for %%D in ("%LOCALAPPDATA%\Programs\Python\Python312" "%LOCALAPPDATA%\Programs\Python\Python311" "%LOCALAPPDATA%\Programs\Python\Python313" "%ProgramFiles%\Python312" "%ProgramFiles%\Python311") do (
    if not defined SYS_PY if exist "%%~D\python.exe" set "SYS_PY="%%~D\python.exe""
)
exit /b 0


:install_python
where winget >nul 2>&1
if errorlevel 1 exit /b 1
echo  Python 3.10-3.13 nao encontrado. Instalando o Python 3.12 via winget...
winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
call :find_python
exit /b 0


:install_llama
set "BK=%~1"
echo.
echo       -^> Tentando a build "%BK%" ...
"%PY%" -m pip uninstall -y llama-cpp-python >nul 2>&1
"%PY%" -m pip install "%LLAMA_SPEC%" --only-binary=llama-cpp-python --prefer-binary --extra-index-url "%WHL_BASE%/%BK%"
if errorlevel 1 (
    echo       [x] Sem wheel "%BK%" compativel com este Python/sistema.
    exit /b 1
)
set "CUDA_PKGS="
"%PY%" -m core.hardware --cuda-runtime %BK% > "%TMPF%" 2>nul
if exist "%TMPF%" set /p CUDA_PKGS=<"%TMPF%"
if defined CUDA_PKGS (
    echo       Instalando o runtime CUDA via pip [CUDA Toolkit nao encontrado]: !CUDA_PKGS!
    "%PY%" -m pip install !CUDA_PKGS!
    if errorlevel 1 exit /b 1
)
"%PY%" -m core.hardware --verify-llama %BK%
if errorlevel 1 exit /b 1
> "%VENV%\llama_backend.txt" echo %BK%
set "LLAMA_OK=1"
exit /b 0


:fail_python
echo.
echo  [ERRO] Python 3.10 a 3.13 nao encontrado e a instalacao automatica falhou.
echo         Instale o Python 3.12 em https://www.python.org/downloads/
echo         (marque "Add python.exe to PATH") e execute o start.bat novamente.
goto :fail_pause

:fail_llama
echo.
echo  [ERRO] Nao foi possivel instalar o llama-cpp-python para este computador.
echo         Verifique a conexao com a internet e execute o start.bat novamente.
goto :fail_pause

:fail
echo.
echo  [ERRO] A inicializacao falhou. Veja as mensagens acima.
:fail_pause
del "%TMPF%" >nul 2>&1
echo.
pause
exit /b 1
