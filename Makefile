# Local development helpers
init:
	$(info Initializing development environment...)
	@python -m venv .venv
	@source ./venv/bin/activate
	@pip install --upgrade pip
	@pip install -r requirements.txt

# Run the MCP server locally (outside Docker). PYTHONPATH must include
# ./tools because tools/long_term_memory.py does bare ``from embeddings
# import ...`` (matching how the Dockerfile sets ``PYTHONPATH=/app:/app/tools``).
dev:
	$(info Starting MCP server locally...)
	@PYTHONPATH=.:./tools python server.py

clean:
	$(info Cleaning project...)
	rm -rf .venv

build:
	$(info Starting development environment...)
	@docker-compose -f docker-compose.yml up -d --build

rebuild:
	$(info Rebuilding development environment...)
	@docker-compose -f docker-compose.yml down 
	@docker-compose -f docker-compose.yml up -d --build

start:
	$(info Starting development environment...)
	@docker-compose -f docker-compose.yml up -d

stop:
	$(info Stopping development environment...)
	@docker-compose -f docker-compose.yml down 

restart:
	$(info Restarting development environment...)
	@docker-compose -f docker-compose.yml down 
	@docker-compose -f docker-compose.yml up -d