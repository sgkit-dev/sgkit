Contributions towards documentations and examples are always welcome!

Documentation is built using [Sphinx.](https://www.sphinx-doc.org/en/master/). If you've ever seen any docs over at [readthedocs](https://readthedocs.org/) then you've likely seen some examples of sphinx out in the wild.

## Build the Documentation

These instructions require that you have docker installed. The best way to do that is to follow the installation instructions at [Get Docker](https://docs.docker.com/get-docker/). The upside to this is that you don't need to clobber any existing conda environments in order to build your docs.

You don't need to have any particular understanding of docker to run these commands. We are treating the docker image as a shell.

```
docker build -t sphinx-sgkit -f docs/Dockerfile .
cd docs
docker run --rm -i -v "$(pwd):/docs" sphinx-sgkit make clean html
```

## Serve the Documentation

Now that we've run built the docs let's view them in their native html state.

```
# You can use any port you'd like instead of 8080
docker run -p 8080:80 -v "$(pwd)/_build/html:/usr/share/nginx/html:ro"  nginx
```

Now open up localhost:8080 in your browser and you'll see the docs just as they appear on the docs website.

