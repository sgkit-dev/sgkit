import numpy as np
import numpy.testing as npt
import pandas as pd

from sgkit import genee
from sgkit.model import create_genotype_call_dataset
from sgkit.utils import encode_array


def test_genee(datadir):
    # Simulated test data was created using https://github.com/ramachandran-lab/genee
    #
    # Edit Simulation.R to create a smaller dataset:
    #
    # -ngene=100; min_gsize=5; max_gsize=20; nsnp_intergenic=400
    # +ngene=5; min_gsize=5; max_gsize=10; nsnp_intergenic=20
    #
    # Then run
    # R --vanilla < Simulation.R
    #
    # Followed by
    #
    #   library("genee")
    #   load("Simulated_LD.RData")
    #   load("Simulated_Summary_Statistics.RData")
    #   load("gene_list.RData")
    #   # use alpha = -1 for OLS
    #   result = genee(mydata, ld, alpha = -1, gene_list = gene_list)
    #   write.csv(mydata, "/path/to/sgkit/sgkit/tests/test_genee/mydata.csv")
    #   write.csv(ld, "/path/to/sgkit/sgkit/tests/test_genee/ld.csv")
    #   write.csv(result, "/path/to/sgkit/sgkit/tests/test_genee/result.csv")
    #   write.csv(t(sapply(gene_list, unlist)), "/path/to/sgkit/sgkit/tests/test_genee/gene_list.csv")

    mydata = pd.read_csv(datadir / "mydata.csv", index_col=0)
    ld = pd.read_csv(datadir / "ld.csv", index_col=0)

    # This was extracted from gene_list.csv
    gene_list = "1:7,8:14,15:19,20:25,26:35"
    gene_list = [[int(s) for s in ss.split(":")] for ss in gene_list.split(",")]
    gene_start, gene_stop = list(zip(*gene_list))
    gene_start = np.array(gene_start) - 1  # make 0-based
    gene_stop = np.array(gene_stop)

    ds = to_sgkit(mydata)

    # turn ld into an array
    ld = ld.to_numpy()

    # genes are windows in this simple example
    ds["window_contig"] = (["windows"], np.full(len(gene_start), 0))
    ds["window_start"] = (["windows"], gene_start)
    ds["window_stop"] = (["windows"], gene_stop)

    df = genee(ds, ld).compute()

    expected = pd.read_csv(datadir / "result.csv", index_col=0)
    expected = expected.reset_index()

    npt.assert_allclose(df["test_q"], expected["test_q"])
    npt.assert_allclose(df["q_var"], expected["q_var"], rtol=0.01)
    npt.assert_allclose(
        df[df["pval"] > 1e-6]["pval"],
        expected[expected["pval"] > 1e-6]["pval"],
        rtol=0.04,
    )


def to_sgkit(mydata):
    """Convert summary stats produced by genee R package to sgkit dataset"""
    variant_contig, variant_contig_names = encode_array(mydata.V1.to_numpy())
    variant_contig = variant_contig.astype("int16")
    variant_contig_names = [str(contig) for contig in variant_contig_names]
    variant_position = mydata.V3.to_numpy()
    variant_id = mydata.V2.to_numpy()
    variant_allele = np.array([["A"]] * len(variant_contig), dtype="S1")  # not used
    sample_id = ["SAMPLE1"]
    ds = create_genotype_call_dataset(
        variant_contig_names=variant_contig_names,
        variant_contig=variant_contig,
        variant_position=variant_position,
        variant_allele=variant_allele,
        sample_id=sample_id,
        variant_id=variant_id,
    )
    ds["beta"] = (["variants"], mydata.V4.to_numpy())
    return ds
