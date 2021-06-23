#!/bin/sh

if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
  if [ ! -d "data" ]; then
    mkdir data
  fi

  cd data                            && \

  if [ ! -d "example" ]; then
      wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id={ADD ZIP ID}' -O example.zip  && \
      unzip example.zip                                  && \
      rm example.zip
  fi
  cd ..
elif [[ "$OSTYPE" == "win32" ]]; then
	echo 'Not implemented for windows, visit https://docs.google.com/uc?export=download&id=14X0PI0YEIUESLvCMzsRiD1vLTuo9bTyr'
fi


