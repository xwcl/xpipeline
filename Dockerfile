FROM ubuntu@sha256:86ac87f73641c920fb42cc9612d4fb57b5626b56ea2a19b894d0673fd5b4f2e9
RUN apt-get -q update \
 && apt-get install -yq --no-install-recommends \
    wget \
    ca-certificates \
    locales \
    fonts-liberation \
    git \
    build-essential \
 && apt-get clean && rm -rf /var/lib/apt/lists/*
# irrelevant for Singularity, good practice for Docker to make an unprivileged user:
RUN useradd -ms /bin/bash containeruser
WORKDIR /tmp
ADD https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh miniconda3.sh
ENV CONDA_DIR=/opt/miniconda3
ENV PATH=$CONDA_DIR/bin:$PATH
RUN bash miniconda3.sh -b -p ${CONDA_DIR}
# note that blas and numba can both leverage openmp
RUN conda install --quiet --yes -c conda-forge \
    "blas=*=mkl" \
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
    pytest && conda clean --all -f -y
RUN conda install pytorch cudatoolkit=11.1 -c pytorch -c nvidia && conda clean --all -f -y
RUN pip install git+https://github.com/xwcl/irods_fsspec.git#egg=irods_fsspec
RUN pip install git+https://github.com/xwcl/xconf.git#egg=xconf
RUN mkdir -p /opt/xpipeline
ADD . /opt/xpipeline/
RUN pip install -e /opt/xpipeline
RUN mkdir -p /srv
# not used when run in Singularity
WORKDIR /srv
ENV MKL_DEBUG_CPU_TYPE 5
ENV DEBIAN_FRONTEND interactive
USER containeruser
# smoke test
RUN python -c "import xpipeline"
RUN xp diagnostic
