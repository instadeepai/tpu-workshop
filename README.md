# tpu-workshop
Materials for the TPU Workshop

##### Connecting to TPU VM

Once you have populated the Makefile with configuration details, you can connect to the VM with the following commands:

```
# Locally from tpu-workshop directory
make create
make listen
make connect

# From VM
cd tpu-workshop/
make docker_build_tpu
make docker_notebook

# Connect to nb (note port can change depending on Makefile setting).
# Access token available in VM terminal.
Go to http://localhost:8889

# Side note: to find and kill listening ports (in case it get's out of sync).
sudo lsof -i :8889
kill -9 <PID>
```