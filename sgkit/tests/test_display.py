from textwrap import dedent

import numpy as np
import pytest
import xarray as xr

from sgkit import display_genotypes, display_pedigree
from sgkit.display import genotype_as_bytes
from sgkit.testing import simulate_genotype_call_dataset


def test_display_genotypes():
    ds = simulate_genotype_call_dataset(n_variant=3, n_sample=3, seed=0)
    disp = display_genotypes(ds)
    expected = """\
        samples    S0   S1   S2
        variants               
        0         0/0  1/0  1/0
        1         0/1  1/0  0/1
        2         0/0  1/0  1/1"""  # noqa: W291
    assert str(disp) == dedent(expected)

    expected_html = """<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>samples</th>
      <th>S0</th>
      <th>S1</th>
      <th>S2</th>
    </tr>
    <tr>
      <th>variants</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0/0</td>
      <td>1/0</td>
      <td>1/0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0/1</td>
      <td>1/0</td>
      <td>0/1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0/0</td>
      <td>1/0</td>
      <td>1/1</td>
    </tr>
  </tbody>
</table>""".strip()
    assert expected_html in disp._repr_html_()


def test_display_genotypes__variant_ids():
    ds = simulate_genotype_call_dataset(n_variant=3, n_sample=3, seed=0)
    # set some variant IDs
    ds["variant_id"] = (["variants"], np.array(["V0", "V1", "V2"]))
    ds["variant_id_mask"] = (["variants"], np.array([False, False, False]))
    disp = display_genotypes(ds)
    expected = """\
        samples    S0   S1   S2
        variants               
        V0        0/0  1/0  1/0
        V1        0/1  1/0  0/1
        V2        0/0  1/0  1/1"""  # noqa: W291
    assert str(disp) == dedent(expected)


def test_display_genotypes__missing_variant_ids():
    ds = simulate_genotype_call_dataset(n_variant=3, n_sample=3, seed=0)
    # set some variant IDs
    ds["variant_id"] = (["variants"], np.array(["."] * 3))
    ds["variant_id_mask"] = (["variants"], np.array([False] * 3))
    disp = display_genotypes(ds)
    expected = """\
        samples    S0   S1   S2
        variants               
        0         0/0  1/0  1/0
        1         0/1  1/0  0/1
        2         0/0  1/0  1/1"""  # noqa: W291
    assert str(disp) == dedent(expected)


def test_display_genotypes__duplicate_variant_ids():
    ds = simulate_genotype_call_dataset(n_variant=3, n_sample=3, seed=0)
    # set some variant IDs
    ds["variant_id"] = (["variants"], np.array(["V0", "V1", "V1"]))
    ds["variant_id_mask"] = (["variants"], np.array([False, False, False]))
    disp = display_genotypes(ds)
    expected = """\
        samples    S0   S1   S2
        variants               
        0         0/0  1/0  1/0
        1         0/1  1/0  0/1
        2         0/0  1/0  1/1"""  # noqa: W291
    assert str(disp) == dedent(expected)


def test_display_genotypes__truncated_rows():
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=10, seed=0)
    disp = display_genotypes(ds, max_variants=4, max_samples=10)
    expected = """\
        samples    S0   S1   S2   S3   S4   S5   S6   S7   S8   S9
        variants                                                  
        0         0/0  1/0  1/0  0/1  1/0  0/1  0/0  1/0  1/1  0/0
        1         1/0  0/1  1/0  1/1  1/1  1/0  1/0  0/0  1/0  1/1
        ...       ...  ...  ...  ...  ...  ...  ...  ...  ...  ...
        8         0/1  0/0  1/0  0/1  0/1  1/0  1/0  0/1  1/0  1/0
        9         1/1  0/1  1/0  0/1  1/0  1/1  0/1  1/0  1/1  1/0

        [10 rows x 10 columns]"""  # noqa: W291
    assert str(disp) == dedent(expected)


def test_display_genotypes__truncated_columns():
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=10, seed=0)
    disp = display_genotypes(ds, max_variants=10, max_samples=4)
    expected = """\
        samples    S0   S1  ...   S8   S9
        variants            ...          
        0         0/0  1/0  ...  1/1  0/0
        1         1/0  0/1  ...  1/0  1/1
        2         1/1  1/1  ...  0/0  1/0
        3         0/1  0/0  ...  1/0  0/0
        4         0/1  0/0  ...  0/0  1/1
        5         1/1  1/0  ...  0/0  1/0
        6         1/1  0/0  ...  1/0  0/1
        7         1/0  0/1  ...  0/1  0/0
        8         0/1  0/0  ...  1/0  1/0
        9         1/1  0/1  ...  1/1  1/0

        [10 rows x 10 columns]"""  # noqa: W291
    assert str(disp) == dedent(expected)


@pytest.mark.parametrize("chunked", [False, True])
def test_display_genotypes__truncated_rows_and_columns(chunked):
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=10, seed=0)
    if chunked:
        ds = ds.chunk(dict(variants=5, samples=5))
    disp = display_genotypes(ds, max_variants=4, max_samples=4)
    expected = """\
        samples    S0   S1  ...   S8   S9
        variants            ...          
        0         0/0  1/0  ...  1/1  0/0
        1         1/0  0/1  ...  1/0  1/1
        ...       ...  ...  ...  ...  ...
        8         0/1  0/0  ...  1/0  1/0
        9         1/1  0/1  ...  1/1  1/0

        [10 rows x 10 columns]"""  # noqa: W291
    assert str(disp) == dedent(expected)

    expected_html = """<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>samples</th>
      <th>S0</th>
      <th>S1</th>
      <th>...</th>
      <th>S8</th>
      <th>S9</th>
    </tr>
    <tr>
      <th>variants</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0/0</td>
      <td>1/0</td>
      <td>...</td>
      <td>1/1</td>
      <td>0/0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/0</td>
      <td>0/1</td>
      <td>...</td>
      <td>1/0</td>
      <td>1/1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0/1</td>
      <td>0/0</td>
      <td>...</td>
      <td>1/0</td>
      <td>1/0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1/1</td>
      <td>0/1</td>
      <td>...</td>
      <td>1/1</td>
      <td>1/0</td>
    </tr>
  </tbody>
</table>
<p>10 rows x 10 columns</p>""".strip()
    assert expected_html in disp._repr_html_()


@pytest.mark.parametrize("chunked", [False, True])
def test_display_genotypes__mixed_ploidy_and_phase(chunked):
    ds = simulate_genotype_call_dataset(
        n_variant=10, n_sample=10, n_ploidy=4, phased=False, missing_pct=0.1, seed=0
    )
    # convert some samples to diploid
    ds.call_genotype.data[:, 0:3, 2:] = -2
    # make some samples phased
    ds.call_genotype_phased.data[:, 2:8] = True
    if chunked:
        ds = ds.chunk(dict(variants=5, samples=5))
    disp = display_genotypes(ds, max_variants=6, max_samples=6)
    expected = """\
        samples    S0   S1   S2  ...       S7       S8       S9
        variants                 ...                           
        0         0/0  1/0  1|0  ...  1|1|1|0  ./0/0/0  1/./1/1
        1         1/1  0/0  0|.  ...  0|1|0|1  0/1/1/0  ./0/0/0
        2         0/1  1/0  1|0  ...  1|.|1|0  1/1/1/0  ./0/1/0
        ...       ...  ...  ...  ...      ...      ...      ...
        7         1/0  1/1  0|0  ...  0|1|0|1  1/0/1/0  0/0/1/1
        8         0/1  1/0  0|1  ...  .|0|1|0  ./1/1/0  0/1/0/.
        9         ./0  1/1  0|1  ...  1|0|0|1  1/./0/1  0/1/0/1

        [10 rows x 10 columns]"""  # noqa: W291
    assert str(disp) == dedent(expected)
    expected_html = """<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>samples</th>
      <th>S0</th>
      <th>S1</th>
      <th>S2</th>
      <th>...</th>
      <th>S7</th>
      <th>S8</th>
      <th>S9</th>
    </tr>
    <tr>
      <th>variants</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0/0</td>
      <td>1/0</td>
      <td>1|0</td>
      <td>...</td>
      <td>1|1|1|0</td>
      <td>./0/0/0</td>
      <td>1/./1/1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1/1</td>
      <td>0/0</td>
      <td>0|.</td>
      <td>...</td>
      <td>0|1|0|1</td>
      <td>0/1/1/0</td>
      <td>./0/0/0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0/1</td>
      <td>1/0</td>
      <td>1|0</td>
      <td>...</td>
      <td>1|.|1|0</td>
      <td>1/1/1/0</td>
      <td>./0/1/0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1/0</td>
      <td>1/1</td>
      <td>0|0</td>
      <td>...</td>
      <td>0|1|0|1</td>
      <td>1/0/1/0</td>
      <td>0/0/1/1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0/1</td>
      <td>1/0</td>
      <td>0|1</td>
      <td>...</td>
      <td>.|0|1|0</td>
      <td>./1/1/0</td>
      <td>0/1/0/.</td>
    </tr>
    <tr>
      <th>9</th>
      <td>./0</td>
      <td>1/1</td>
      <td>0|1</td>
      <td>...</td>
      <td>1|0|0|1</td>
      <td>1/./0/1</td>
      <td>0/1/0/1</td>
    </tr>
  </tbody>
</table>
<p>10 rows x 10 columns</p></div>""".strip()
    assert expected_html in disp._repr_html_()


def test_display_genotypes__large():
    ds = simulate_genotype_call_dataset(n_variant=100_000, n_sample=1000, seed=0)
    disp = display_genotypes(ds, max_variants=4, max_samples=4)
    expected = """\
        samples    S0   S1  ... S998 S999
        variants            ...          
        0         0/0  1/0  ...  0/1  1/1
        1         1/1  1/1  ...  0/1  1/1
        ...       ...  ...  ...  ...  ...
        99998     0/1  1/1  ...  1/0  0/1
        99999     1/0  1/0  ...  1/0  1/0

        [100000 rows x 1000 columns]"""  # noqa: W291
    assert str(disp) == dedent(expected)


def test_display_genotypes__large_alleles():
    np.random.seed(0)
    ds = xr.Dataset()
    ds["sample_id"] = "samples", list("ABCDE")
    ds["variant_position"] = "variants", np.arange(10)
    ds["allele"] = "alleles", np.arange(1050)
    ds["call_genotype"] = ["variants", "samples", "ploidy"], np.random.randint(
        950, 1050, size=(10, 5, 2)
    )
    disp = display_genotypes(ds, max_variants=6, max_samples=4)
    expected = """\
      samples           A          B  ...         D          E
      variants                        ...                     
      0           994/997  1014/1017  ...  1033/971   986/1037
      1         1020/1038   1038/962  ...  989/1037   996/1038
      2          1031/987   975/1027  ...  970/1030  1019/1029
      ...             ...        ...  ...       ...        ...
      7           985/961   996/1032  ...  964/1049   1003/962
      8          992/1034  1025/1018  ...   997/953  1026/1002
      9          1028/965   970/1049  ...  1029/963   1035/998

      [10 rows x 5 columns]"""  # noqa: W291
    assert str(disp) == dedent(expected)


@pytest.mark.parametrize(
    "genotype, phased, max_allele_chars, expect",
    [
        ([0, 1], False, 2, b"0/1"),
        ([1, 2, 3, 1], True, 2, b"1|2|3|1"),
        ([0, -1], False, 2, b"0/."),
        ([0, -2], False, 2, b"0"),
        ([0, -2, 1], True, 2, b"0|1"),
        ([-1, -2, 1], False, 2, b"./1"),
        ([22, -1, -2, 7, -2], False, 2, b"22/./7"),
        ([0, 333], False, 2, b"0/33"),  # truncation
        ([0, 333], False, 3, b"0/333"),
        (
            [[0, 1, 2, -1], [0, 2, -2, -2]],
            np.array([False, True]),
            2,
            [b"0/1/2/.", b"0|2"],
        ),
    ],
)
def test_genotype_as_bytes(genotype, phased, max_allele_chars, expect):
    genotype = np.array(genotype)
    np.testing.assert_array_equal(
        expect,
        genotype_as_bytes(genotype, phased, max_allele_chars),
    )


def pedigree_Hamilton_Kerr():
    ds = xr.Dataset()
    ds["sample_id"] = "samples", ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
    ds["parent_id"] = ["samples", "parents"], [
        [".", "."],
        [".", "."],
        [".", "S2"],
        ["S1", "."],
        ["S1", "S3"],
        ["S1", "S3"],
        ["S6", "S2"],
        ["S6", "S2"],
    ]
    ds["stat_Hamilton_Kerr_tau"] = ["samples", "parents"], [
        [1, 1],
        [2, 2],
        [0, 2],
        [2, 0],
        [1, 1],
        [2, 2],
        [2, 2],
        [2, 2],
    ]
    ds["stat_Hamilton_Kerr_lambda"] = ["samples", "parents"], [
        [0.0, 0.0],
        [0.167, 0.167],
        [0.0, 0.167],
        [0.041, 0.0],
        [0.0, 0.0],
        [0.918, 0.041],
        [0.167, 0.167],
        [0.167, 0.167],
    ]
    return ds


def test_display_pedigree__no_coords():
    ds = pedigree_Hamilton_Kerr()
    graph = display_pedigree(ds)
    expect = """    digraph {
    \t0
    \t1
    \t2
    \t3
    \t4
    \t5
    \t6
    \t7
    \t1 -> 2
    \t0 -> 3
    \t0 -> 4
    \t2 -> 4
    \t0 -> 5
    \t2 -> 5
    \t5 -> 6
    \t1 -> 6
    \t5 -> 7
    \t1 -> 7
    }
    """
    assert str(graph) == dedent(expect)


def test_display_pedigree__samples_coords():
    ds = pedigree_Hamilton_Kerr()
    ds = ds.assign_coords(samples=ds.sample_id)
    graph = display_pedigree(ds)
    expect = """    digraph {
    \t0 [label=S1]
    \t1 [label=S2]
    \t2 [label=S3]
    \t3 [label=S4]
    \t4 [label=S5]
    \t5 [label=S6]
    \t6 [label=S7]
    \t7 [label=S8]
    \t1 -> 2
    \t0 -> 3
    \t0 -> 4
    \t2 -> 4
    \t0 -> 5
    \t2 -> 5
    \t5 -> 6
    \t1 -> 6
    \t5 -> 7
    \t1 -> 7
    }
    """
    assert str(graph) == dedent(expect)


def test_display_pedigree__samples_coords_reorder():
    ds = pedigree_Hamilton_Kerr()
    ds = ds.sel(samples=[7, 3, 5, 0, 4, 1, 2, 6])
    ds = ds.assign_coords(samples=ds.sample_id)
    graph = display_pedigree(ds)
    expect = """    digraph {
    \t0 [label=S8]
    \t1 [label=S4]
    \t2 [label=S6]
    \t3 [label=S1]
    \t4 [label=S5]
    \t5 [label=S2]
    \t6 [label=S3]
    \t7 [label=S7]
    \t2 -> 0
    \t5 -> 0
    \t3 -> 1
    \t3 -> 2
    \t6 -> 2
    \t3 -> 4
    \t6 -> 4
    \t5 -> 6
    \t2 -> 7
    \t5 -> 7
    }
    """
    assert str(graph) == dedent(expect)


def test_display_pedigree__samples_labels():
    ds = pedigree_Hamilton_Kerr()
    graph = display_pedigree(ds, node_attrs=dict(label=ds.sample_id))
    expect = """    digraph {
    \t0 [label=S1]
    \t1 [label=S2]
    \t2 [label=S3]
    \t3 [label=S4]
    \t4 [label=S5]
    \t5 [label=S6]
    \t6 [label=S7]
    \t7 [label=S8]
    \t1 -> 2
    \t0 -> 3
    \t0 -> 4
    \t2 -> 4
    \t0 -> 5
    \t2 -> 5
    \t5 -> 6
    \t1 -> 6
    \t5 -> 7
    \t1 -> 7
    }
    """
    assert str(graph) == dedent(expect)


def test_display_pedigree__broadcast():
    ds = pedigree_Hamilton_Kerr()
    inbreeding = np.array([0.0, 0.077, 0.231, 0.041, 0.0, 0.197, 0.196, 0.196])
    label = (ds.sample_id.str + "\n").str + inbreeding.astype("U")
    edges = xr.where(
        ds.stat_Hamilton_Kerr_tau == 2,
        "black:black",
        "black",
    )
    graph = display_pedigree(
        ds,
        graph_attrs=dict(splines="false", outputorder="edgesfirst"),
        node_attrs=dict(
            style="filled", fillcolor="black", fontcolor="white", label=label
        ),
        edge_attrs=dict(arrowhead="crow", color=edges),
    )
    expect = """    digraph {
    \toutputorder=edgesfirst splines=false
    \t0 [label="S1\n    0.0" fillcolor=black fontcolor=white style=filled]
    \t1 [label="S2\n    0.077" fillcolor=black fontcolor=white style=filled]
    \t2 [label="S3\n    0.231" fillcolor=black fontcolor=white style=filled]
    \t3 [label="S4\n    0.041" fillcolor=black fontcolor=white style=filled]
    \t4 [label="S5\n    0.0" fillcolor=black fontcolor=white style=filled]
    \t5 [label="S6\n    0.197" fillcolor=black fontcolor=white style=filled]
    \t6 [label="S7\n    0.196" fillcolor=black fontcolor=white style=filled]
    \t7 [label="S8\n    0.196" fillcolor=black fontcolor=white style=filled]
    \t1 -> 2 [arrowhead=crow color="black:black"]
    \t0 -> 3 [arrowhead=crow color="black:black"]
    \t0 -> 4 [arrowhead=crow color=black]
    \t2 -> 4 [arrowhead=crow color=black]
    \t0 -> 5 [arrowhead=crow color="black:black"]
    \t2 -> 5 [arrowhead=crow color="black:black"]
    \t5 -> 6 [arrowhead=crow color="black:black"]
    \t1 -> 6 [arrowhead=crow color="black:black"]
    \t5 -> 7 [arrowhead=crow color="black:black"]
    \t1 -> 7 [arrowhead=crow color="black:black"]
    }
    """
    assert str(graph) == dedent(expect)
