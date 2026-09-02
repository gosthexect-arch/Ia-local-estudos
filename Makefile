PY ?= python3
export PYTHONPATH := src:tests

.PHONY: test doctor doctor-json config help

help:
	@echo "make test        - roda a suite (biblioteca padrao, sem instalar nada)"
	@echo "make doctor      - diagnostico completo da Fase 1"
	@echo "make doctor-json - diagnostico em JSON"
	@echo "make config      - mostra a configuracao efetiva"

test:
	$(PY) -m unittest discover -s tests -v

doctor:
	$(PY) -m cripto_monitor doctor

doctor-json:
	$(PY) -m cripto_monitor doctor --json

config:
	$(PY) -m cripto_monitor config
