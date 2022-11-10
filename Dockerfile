FROM python:3.10-alpine AS builder
COPY . /src
WORKDIR /src
RUN pip install poetry
RUN poetry build

FROM python:3.10 AS dist
RUN --mount=from=builder,source=/src/dist,target=/src/dist \
    pip install --find-links=/src/dist dlkoopman

COPY README.md /
