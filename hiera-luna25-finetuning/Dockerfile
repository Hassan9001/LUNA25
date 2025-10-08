# Portions of this code are adapted from luna25-baseline-public
# Source: https://github.com/DIAGNijmegen/luna25-baseline-public/blob/main/Dockerfile
# License: Apache License 2.0 (see https://github.com/DIAGNijmegen/luna25-baseline-public/blob/main/LICENSE)

FROM --platform=linux/amd64 pytorch/pytorch AS example-algorithm-amd64

ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user experiment_config.py dataloader.py processor.py inference.py /opt/app/
COPY --chown=user:user models /opt/app/models
COPY --chown=user:user results /opt/app/resources

ENTRYPOINT ["python", "inference.py"]
