CONTAINER = xpipeline

all: build run

build:
	docker build . -t xwcl/$(CONTAINER)

force-build:
	docker build --no-cache . -t xwcl/$(CONTAINER)

push:
	docker push xwcl/$(CONTAINER)

run: build
	docker run -v $(PWD)/srv:/srv -it xwcl/$(CONTAINER) bash -l

.PHONY: all build force-build push run
