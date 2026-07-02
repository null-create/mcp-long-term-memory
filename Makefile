# Local development helpers
init:
	$(info Initializing development environment...)
	@python -m venv .venv
	@source ./venv/bin/activate
	@pip install --upgrade pip
	@pip install -r requirements.txt

clean:
	$(info Cleaning project...)
	rm -rf .venv

run:
	$(info Starting development environment...)
	@docker-compose -f docker-compose.yml up -d --build

stop:
	$(info Stopping development environment...)
	@docker-compose -f docker-compose.yml down 
	