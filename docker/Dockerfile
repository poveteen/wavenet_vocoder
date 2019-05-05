# Environment for Wavenet Vocoder
FROM smartear/wavenet_vocoder:common

# Setting up Open SSH Server for VSCode Remote-SSH
RUN apt-get update
RUN apt-get install openssh-server -y
RUN /etc/init.d/ssh start

# Attach Entry point
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create User ubuntu
RUN useradd -ms /bin/bash ubuntu

ENTRYPOINT ["/entrypoint.sh"]
