# The binary to build (just the basename).
MODULE := autoballs

# Where to push the docker image.
# need to update!
REGISTRY := rdg31

IMAGE := $(REGISTRY)/$(MODULE)

# DISPLAY
DISPLAY := $(shell echo $(DISPLAY))

DATA_DIR := "/home/ryan/Google Drive/TFM Cambridge/2021/Frogs"

# This version-strategy uses git tags to set the version string
TAG := $(shell git describe --tags --always --dirty)

BLUE='\033[0;34m'
NC='\033[0m' # No Color


build-dev:
	@echo "\n${BLUE}Building Development image with labels:\n"
	@echo "name: $(MODULE)"
	@echo "version: $(TAG)${NC}\n"
	@sed                                 \
	    -e 's|{NAME}|$(MODULE)|g'        \
	    -e 's|{VERSION}|$(TAG)|g'        \
	    docker/dev.Dockerfile | docker build -t $(IMAGE):$(TAG) -f- .


# Example: make shell CMD="-c 'date > datefile'"
shell: build-dev
	@echo "\n${BLUE}Launching a shell in the containerized build environment...${NC}\n"
		@xhost local:root
		@docker run                                                 \
			-e DISPLAY=$(DISPLAY) \
			-v "/tmp/.X11-unix:/tmp/.X11-unix" \
			-ti                                                     \
			--rm                                                    \
			--entrypoint /bin/bash                                  \
			-u 0					                                \
			--mount type=bind,source=$(shell pwd)/,target=/autoballs\
			--mount type=bind,source=$(DATA_DIR)/,target=/autoballs/data\
			$(IMAGE):$(TAG)										    \
			

# Example: make shell CMD="-c 'date > datefile'"
shell-gpu: build-dev
	@echo "\n${BLUE}Launching a shell in the containerized build environment...${NC}\n"
		@xhost local:root
		@docker run                                                 \
			-e CURRENT_TAG=$(TAG) \
			-e DISPLAY=$(DISPLAY) \
			-v "/tmp/.X11-unix:/tmp/.X11-unix" \
			-it                                                     \
			--rm                                                    \
			--entrypoint /bin/bash                                  \
			-u 0					                                \
			--mount type=bind,source=$(shell pwd)/,target=/autoballs\
			--mount type=bind,source=$(DATA_DIR)/,target=/autoballs/data\
			--gpus all												\
			--ipc=host												\
			$(IMAGE):$(TAG)										    \
			$(CMD)


shell-gpu-nx: build-dev
	@echo "\n${BLUE}Launching a shell in the containerized build environment...${NC}\n"
		@docker run                                                 \
			-e CURRENT_TAG=$(TAG) \
			-it                                                     \
			--rm                                                    \
			--entrypoint /bin/bash                                  \
			-u 0					                                \
			--mount type=bind,source=$(shell pwd)/,target=/autoballs\
			--mount type=bind,source=$(DATA_DIR)/,target=/autoballs/data\
			--gpus all												\
			--ipc=host												\
			$(IMAGE):$(TAG)										    \
			$(CMD)
