# pciSeq in a box. The image holds python, pciSeq and everything pciSeq needs, so
# a run does not depend on what is installed on the machine.
#
# Build, from the top of the repo. PCISEQ_REF is the tag, branch or commit of
# pciSeq to install:
#
#     docker build --build-arg PCISEQ_REF=v0.1.2 -t pciseq:0.1.2 docker/
#
# Run, with the folder that holds the config file and the data mounted at /data.
# Paths inside the config file have to start with /data:
#
#     docker run --rm -u $(id -u):$(id -g) -v /path/to/my_run:/data pciseq:0.1.2 run /data/config.yaml

FROM python:3.10-slim

ARG PCISEQ_REF=v0.1.2

# git is there for pip, which clones the repo, and for setup.py, which reads the
# commit and the tag from it so a run can say which code made it
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# pciSeq itself comes from GitHub, nothing is copied from the checkout. The
# versions of its dependencies come from constraints.txt, so two builds give the
# same image. See the top of that file for how it was made.
COPY constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
        "pciSeq_3d[mcp] @ git+https://github.com/acycliq/pciSeq_3d.git@${PCISEQ_REF}" \
    && rm /tmp/constraints.txt

# The image is run as whoever starts it (-u above, and always under Apptainer),
# and that user cannot write next to the installed code. numba wants to save its
# compiled functions there and matplotlib its font list, so point both at /tmp.
# pciSeq writes its log under TMPDIR.
ENV NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    MPLBACKEND=Agg \
    TMPDIR=/tmp \
    HOME=/tmp

WORKDIR /data
ENTRYPOINT ["pciseq"]
CMD ["--help"]
