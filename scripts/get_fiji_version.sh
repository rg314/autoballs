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

<<<<<<< HEAD
# elif [ "$OSTYPE" == "darwin"* ]; then
=======
elif [[ "$OSTYPE" == "darwin"* ]]; then
>>>>>>> 6f1968aef3a683bd112a0f9a3847675a5f87ef1a

#   if [ ! -d "Fiji" ]; then
#     mkdir Fiji
#   fi

#   cd Fiji                            && \

<<<<<<< HEAD
#   if [ ! -d "Fiji.app" ]; then
#       wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1RXXXwjBRTtiRyw5F1mhEB44U9_UyCyVA' -O Fiji.app.zip  && \
#       unzip Fiji.app.zip                                  && \
#       rm Fiji.app.zip
#   fi
=======
  if [[ ! -d "Fiji.app" ]]; then
      wget --no-check-certificate -r 'https://docs.google.com/uc?export=download&id=1RXXXwjBRTtiRyw5F1mhEB44U9_UyCyVA' -O Fiji.app.zip  && \
      unzip Fiji.app.zip                                  && \
      rm Fiji.app.zip
  fi
>>>>>>> 6f1968aef3a683bd112a0f9a3847675a5f87ef1a

#   cd ..

<<<<<<< HEAD
# elif [ "$OSTYPE" == "win32" ]; then
# 	echo 'Not implemented for windows, visit https://docs.google.com/uc?export=download&id=14X0PI0YEIUESLvCMzsRiD1vLTuo9bTyr'
# fi
=======
elif [[ "$OSTYPE" == "win32" ]]; then
	echo 'Not implemented for windows, visit https://docs.google.com/uc?export=download&id=14X0PI0YEIUESLvCMzsRiD1vLTuo9bTyr'
fi
>>>>>>> 6f1968aef3a683bd112a0f9a3847675a5f87ef1a


