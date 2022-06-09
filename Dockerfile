FROM ubuntu@sha256:bace9fb0d5923a675c894d5c815da75ffe35e24970166a48a4460a48ae6e0d19
RUN apt-get -q update \
 && apt-get install -yq --no-install-recommends \
    wget \
    ca-certificates \
    locales \
    fonts-liberation \
    git \
    build-essential \
    curl \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
# irrelevant for Singularity, good practice for Docker to make an unprivileged user:
RUN useradd -ms /bin/bash containeruser
WORKDIR /tmp
RUN bash -c "curl -L -O \"https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh\""
ENV CONDA_DIR=/mamba
ENV PATH=$CONDA_DIR/bin:$PATH
RUN bash -c "bash Mambaforge-$(uname)-$(uname -m).sh -b -p ${CONDA_DIR}"
# note that blas and numba can both leverage openmp
RUN mamba install --quiet --yes -c conda-forge \
    'blas=*=mkl' \
    numpy \
    scipy \
    matplotlib \
    astropy \
    scikit-image \
    graphviz \
    python-graphviz \
    dask \
    distributed \
    python-dateutil \
    orjson \
    fsspec \
    numba \
    jupyter-server-proxy \
    py-spy \
    memory_profiler \
    sphinx \
    pytest \
    mkl-service \
    tbb \
    && conda clean --all -f -y
# inspired by Open Science Grid Dockerfiles
RUN for MNTPOINT in \
        /cvmfs \
        /ceph \
        /hadoop \
        /hdfs \
        /lizard \
        /mnt/hadoop \
        /mnt/hdfs \
        /xenon \
        /spt \
        /stash2 \
    ; do \
        mkdir -p $MNTPOINT ; \
    done
RUN pip install stashcp
RUN pip install 'ray[default]'
RUN pip install git+https://github.com/fmfn/BayesianOptimization.git@35535c6312f365ead729de3d889d7b1fae1a8e0b
RUN pip install git+https://github.com/xwcl/irods_fsspec.git#egg=irods_fsspec
RUN pip install git+https://github.com/xwcl/xconf.git#egg=xconf
RUN mkdir -p /opt/xpipeline
ADD . /opt/xpipeline/
RUN pip install -e /opt/xpipeline
WORKDIR /opt/xpipeline
# Not only does this ensure we don't build a broken image,
# but it also compiles those Numba functions we can ahead of time
RUN pytest -x
RUN mkdir -p /srv
# not used when run in Singularity
WORKDIR /srv
ENV DEBIAN_FRONTEND interactive
USER containeruser
# smoke test
RUN python -c "import xpipeline"
RUN xp diagnostic
