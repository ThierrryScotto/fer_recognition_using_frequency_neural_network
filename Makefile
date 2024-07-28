.PHONY: install

install: install-cmake
	@mkdir .venv | true
	@pipenv --version | pip install pipenv
	@pipenv install

install-cmake: 
	@sudo apt install cmake -y