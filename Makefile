#################
# General setup #
#################

GIT_BRANCH=main

# The following variables are assumed to already exist as environment variables locally, or can be edited below.
#ACCOUNT=$(GITHUB_ACCESS_TOKEN)
ACCOUNT=""

PORT=8890

#######
# TPU #
#######

ACCELERATOR_TYPE=""
BASE_CMD=gcloud alpha compute tpus tpu-vm
NAME=""
PROJECT=""
ZONE=""


WORKER=all
FILE=""
RUNTIME_VERSION=tpu-vm-base

.PHONY: create_vm
create_vm:
	$(BASE_CMD) create $(NAME) --zone $(ZONE) \
		--project $(PROJECT) \
		--accelerator-type $(ACCELERATOR_TYPE) \
		--version $(RUNTIME_VERSION)

.PHONY: prepare_vm
prepare_vm:
	$(BASE_CMD) ssh --zone $(ZONE) $(NAME) \
		--project $(PROJECT) \
		--worker=$(WORKER) \
		--command="git clone -b ${GIT_BRANCH} https://github.com/instadeepai/tpu-workshop.git"

.PHONY: create
create: create_vm prepare_vm

.PHONY: start
start:
	$(BASE_CMD) start $(NAME) --zone=$(ZONE) --project $(PROJECT)

.PHONY: connect
connect:
	$(BASE_CMD) ssh $(NAME) --zone $(ZONE) --project $(PROJECT)

.PHONY: listen
listen:
	$(BASE_CMD) ssh $(NAME) --zone $(ZONE) --project $(PROJECT) \
	-- -NfL $(PORT):localhost:$(PORT)

.PHONY: list
list:
	$(BASE_CMD) list --zone=$(ZONE) --project $(PROJECT)

.PHONY: stop
stop:
	$(BASE_CMD) stop $(NAME) --zone=$(ZONE) --project $(PROJECT)

.PHONY: delete
delete:
	$(BASE_CMD) delete $(NAME) --zone $(ZONE) --project $(PROJECT)

##########
# Docker #
##########

SHELL := /bin/bash

# variables
WORK_DIR = $(PWD)
USER_ID = $$(id -u)
GROUP_ID = $$(id -g)

DOCKER_BUILD_ARGS = \
	--build-arg USER_ID=$(USER_ID) \
	--build-arg GROUP_ID=$(GROUP_ID)

DOCKER_RUN_FLAGS = --rm --privileged -p ${PORT}:${PORT} --network host
DOCKER_IMAGE_NAME = tpu_workshop
DOCKER_CONTAINER_NAME = tpu_workshop_container

.PHONY: docker_build_tpu
docker_build_tpu:
	sudo docker build -t $(DOCKER_IMAGE_NAME) $(DOCKER_BUILD_ARGS) -f docker/tpu.Dockerfile . \
	--build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID)

.PHONY: docker_run
docker_run:
	sudo docker run $(DOCKER_RUN_FLAGS) --name $(DOCKER_CONTAINER_NAME) \
	-v $(WORK_DIR):/app $(DOCKER_IMAGE_NAME) $(command)

.PHONY: docker_enter
docker_enter:
	sudo docker run $(DOCKER_RUN_FLAGS) -v $(WORK_DIR):/app -it $(DOCKER_IMAGE_NAME)

.PHONY: docker_kill
docker_kill:
	sudo docker kill $(DOCKER_CONTAINER_NAME)

.PHONY: docker_notebook
docker_notebook:
	#echo "Make sure you have properly exposed your VM before, with the gcloud ssh command followed by -- -N -f -L $(PORT):localhost:$(PORT)"
	sudo docker run $(DOCKER_RUN_FLAGS) -p ${PORT}:${PORT} -v $(WORK_DIR):/app -t $(DOCKER_IMAGE_NAME) \
		jupyter lab --port=$(PORT) --no-browser --ip 0.0.0.0 --allow-root
#	echo "Go to http://localhost:${PORT} and enter token above."
