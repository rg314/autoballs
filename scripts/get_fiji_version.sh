#!/bin/sh

if [[ "$OSTYPE" == "linux-gnu"* ]]; then

  if [ ! -d "Fiji" ]; then
    mkdir Fiji
  fi

  cd Fiji                            && \

  if [ ! -d "Fiji.app" ]; then
      wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1KxbVQIsav-jZeFsXx0D8XmJZlYu-8k4o' -O Fiji.app.zip  && \
      unzip Fiji.app.zip                                  && \
      rm Fiji.app.zip
  fi

  cd ..
fi

# elif [ "$OSTYPE" == "darwin"* ]; then

#   if [ ! -d "Fiji" ]; then
#     mkdir Fiji
#   fi

#   cd Fiji                            && \

#   if [ ! -d "Fiji.app" ]; then
#       wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1RXXXwjBRTtiRyw5F1mhEB44U9_UyCyVA' -O Fiji.app.zip  && \
#       unzip Fiji.app.zip                                  && \
#       rm Fiji.app.zip
#   fi

#   cd ..

# elif [ "$OSTYPE" == "win32" ]; then
# 	echo 'Not implemented for windows, visit https://docs.google.com/uc?export=download&id=14X0PI0YEIUESLvCMzsRiD1vLTuo9bTyr'
# fi


