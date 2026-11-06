# Extends the prebuilt Strata image with launch parameters. Nothing is compiled: the Terminal's serve/server.py and
# strata_cli/launch/cmd.py are Python source inside the image and are patched in place.
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
