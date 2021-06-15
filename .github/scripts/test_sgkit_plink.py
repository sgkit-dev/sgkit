import urllib.request

from sgkit.io.plink import read_plink

if __name__ == "__main__":
    for ext in (".bed", ".bim", ".fam"):
        urllib.request.urlretrieve(
            f"https://github.com/pystatgen/sgkit/raw/main/sgkit/tests/io/plink/data/plink_sim_10s_100v_10pmiss{ext}",
            f"plink_sim_10s_100v_10pmiss{ext}",
        )
    ds = read_plink(path="plink_sim_10s_100v_10pmiss")
    print(ds)
