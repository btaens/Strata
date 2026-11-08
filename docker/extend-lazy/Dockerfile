# Extends the prebuilt Strata image with launch parameters. Nothing is compiled: the Terminal's serve/server.py and
# strata_cli/launch/cmd.py are Python source inside the image and are patched in place.
#
# Only needed for the published 0.2.7 images. From 0.2.11 on the server reads STRATA_LAZY_LOAD, STRATA_IDLE_UNLOAD_S
# and STRATA_DISABLE_WEBUI itself (docs/DOCKER.md), so an image built from this source tree - or any image published
# from 0.2.11 on - needs no patch layer; the same -e flags work as they are. --no-browser is not needed either:
# nothing opens a browser without `--open`, and `strata launch` never passes it.
#
#   STRATA_LAZY_LOAD=1       start the API with the model unloaded; the first request (or POST /load) loads it
#   STRATA_IDLE_UNLOAD_S=N   unload the model after N seconds without requests (0 or unset: never)
#   STRATA_DISABLE_WEBUI=1   no web page: /, /api-monitor, /web/* and /fonts/* answer 404; the API stays up
#   --no-browser             accepted by `strata launch` and serve.server; overrides --open (no browser opens by default)
#
# Build and run (from this folder):
#   docker build -t strata:0.2.7-cuda13-lazy .
#   docker run --gpus all -e STRATA_LAZY_LOAD=1 -e STRATA_IDLE_UNLOAD_S=1800 -v strata-data:/data -p 8080:8080 strata:0.2.7-cuda13-lazy
#
# The build fails if the base image's files are not the ones the patch expects (another Strata version).

ARG BASE_IMAGE=ghcr.io/niko1221/strata:0.2.7-cuda13
FROM ${BASE_IMAGE}

COPY patch_server.py /tmp/patch_server.py
RUN /opt/strata/terminal/python/bin/python3 /tmp/patch_server.py /opt/strata/terminal \
 && rm /tmp/patch_server.py
