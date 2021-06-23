FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN  apt-get update \
  && apt-get install -y wget \
    p7zip-full \
    unace \
    zip \
    unzip \
    xz-utils \
    sharutils \
    uudeview \
    mpack \
    arj \
    cabextract \
    file-roller \
  && rm -rf /var/lib/apt/lists/*


RUN mkdir Fiji &&\
    cd Fiji &&\
    wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1KxbVQIsav-jZeFsXx0D8XmJZlYu-8k4o' -O Fiji.app.zip \
RUN mkdir autoballs


# # Install OpenJDK-8
# RUN apt-get update && \
#     apt-get install -y openjdk-8-jdk curl && \
#     apt-get install -y ant && \
#     apt-get clean;

# # Fix certificate issues
# RUN apt-get update && \
#     apt-get install ca-certificates-java && \
#     apt-get clean && \
#     update-ca-certificates -f;

# # Setup JAVA_HOME -- useful for docker commandline
# ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
# RUN export JAVA_HOME

ARG jdk=8
ARG maven=3.5

RUN conda install -q -y\
        openjdk=${jdk} \
        maven=${maven} &&\
    conda clean --all

WORKDIR /autoballs
ENV HOME /autoballs




ENTRYPOINT ["/bin/bash/"]
USER root


LABEL name={NAME}
LABEL version={VERSION}
