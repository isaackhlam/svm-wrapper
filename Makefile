install:
	git config core.hooksPath .githooks
	git submodule update --init --recursive --remote --force
	pip install -r requirements.txt
	pnpm install

lint/quick:
	git ls-files --exclude-standard -s '*.py' | awk "{print $$4}" | flake8
	git ls-files --exclude-standard -s '*.ts' | awk "{print $$4}" | pnpm eslint

format/quick:
	git ls-files --exclude-standard -s '*.py' | awk '{print $$4}' | xargs -I {} black {}
	git ls-files --exclude-standard -s '*.py' | awk '{print $$4}' | xargs -I {} isort {}
	git ls-files --exclude-standard -s '*.yml' '*.yaml' | awk '{print $$4}' | xargs -t -I {} yq -i -S -Y . {}

lint:
	for i in $$(ls -d py-*/); do flake8 $$i; bandit -r $$i; done
	for i in $$(ls -d ts-*/); do pnpm eslint $$i; pnpm depcheck $$i; done

format:
	for i in $$(ls -d py-*/); do python -m black $$i; isort $$i; done
