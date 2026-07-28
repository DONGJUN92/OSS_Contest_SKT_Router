# LPB-Router — POSIX 환경용. Windows에서는 python reproduce.py 사용.
.PHONY: install test reproduce quick

install:
	pip install -r requirements.txt

test:
	python tests/test_cost_mirror.py

reproduce:
	python reproduce.py

quick:
	python reproduce.py --quick
