PY ?= python3
export PYTHONPATH := src:tests

.PHONY: test doctor doctor-json collect backfill status replay config help

help:
	@echo "make test        - roda a suite completa"
	@echo "make doctor      - diagnostico da Fase 1"
	@echo "make doctor-json - diagnostico em JSON"
	@echo "make collect     - coleta continua (Ctrl-C para parar)"
	@echo "make backfill    - recupera velas historicas por REST"
	@echo "make status      - saude do armazenamento e continuidade"
	@echo "make replay      - reentrega o historico gravado"
	@echo "make config      - mostra a configuracao efetiva"

test:
	$(PY) -m unittest discover -s tests -v

doctor:
	$(PY) -m cripto_monitor doctor

doctor-json:
	$(PY) -m cripto_monitor doctor --json

collect:
	$(PY) -m cripto_monitor collect

backfill:
	$(PY) -m cripto_monitor backfill

status:
	$(PY) -m cripto_monitor status

replay:
	$(PY) -m cripto_monitor replay --resumo

config:
	$(PY) -m cripto_monitor config
