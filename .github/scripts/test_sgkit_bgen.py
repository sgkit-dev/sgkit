import urllib.request

from sgkit.io.bgen import read_bgen

if __name__ == "__main__":
    urllib.request.urlretrieve(
        "https://github.com/pystatgen/sgkit/raw/main/sgkit/tests/io/bgen/data/example.bgen",
        "example.bgen",
    )
    ds = read_bgen("example.bgen")
    print(ds)
