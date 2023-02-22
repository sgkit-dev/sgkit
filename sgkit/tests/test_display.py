from textwrap import dedent

import numpy as np
import pytest

from sgkit import display_genotypes
from sgkit.display import genotype_as_bytes, truncate
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


def test_display_genotypes__truncated_rows_and_columns():
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=10, seed=0)
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


def test_truncate_fails_with_only_one_dimension():
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=10, seed=0)
    with pytest.raises(
        ValueError, match="Truncation is only supported for two dimensions"
    ):
        truncate(ds, {"variants": 10})


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
