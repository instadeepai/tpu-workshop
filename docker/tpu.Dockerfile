FROM mambaorg/micromamba:0.22.0 as conda

# Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1

COPY docker/environment_tpu.yaml /tmp/environment_tpu.yaml


RUN micromamba create -y --file /tmp/environment_tpu.yaml \
    && micromamba clean --all --yes \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete


FROM debian:bullseye-slim as test-image

COPY --from=conda /opt/conda/envs/. /opt/conda/envs/
ENV PATH=/opt/conda/envs/google-tpu-workshop/bin/:$PATH APP_FOLDER=/app
ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY . $APP_FOLDER
WORKDIR $APP_FOLDER

ARG USER_ID=1000
ARG GROUP_ID=1000
ENV USER=eng
ENV GROUP=eng

RUN pip install --user -e ./gptj-demo


FROM test-image as run-image
# The run-image (default) is the same as the dev-image with the some files directly
# copied inside
RUN apt update
RUN apt install -y curl unzip aria2 wget
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

RUN mkdir /usr/share/grpc/
RUN curl -Lo roots.pem https://pki.google.com/roots.pem
RUN mv roots.pem /usr/share/grpc/roots.pem
