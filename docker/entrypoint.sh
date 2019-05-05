#!/bin/sh

# Start the ssh server
/etc/init.d/ssh start

su - ubuntu
cd /home/ubuntu

# Execute the CMD
exec "$@"
