#!/bin/sh

if [ ! -d "Fiji" ]; then
  mkdir Fiji
fi

cd Fiji                            && \

if [ ! -d "Fiji.app" ]; then
    wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=14X0PI0YEIUESLvCMzsRiD1vLTuo9bTyr' -O Fiji.app.zip  && \
    unzip Fiji.app.zip                                  && \
    rm Fiji.app.zip
fi


cd ..
