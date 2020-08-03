from sgkit import display_genotypes
from sgkit.testing import simulate_genotype_call_dataset


def test_display_genotypes():
    ds = simulate_genotype_call_dataset(n_variant=3, n_sample=3, seed=0)
    disp = display_genotypes(ds)
    assert (
        str(disp)
        == """
samples    S0   S1   S2
variants               
0         0/0  1/0  1/0
1         0/1  1/0  0/1
2         0/0  1/0  1/1
""".strip()  # noqa: W291
    )

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


def test_display_genotypes__truncated():
    ds = simulate_genotype_call_dataset(n_variant=10, n_sample=10, seed=0)
    disp = display_genotypes(ds, max_variants=4, max_samples=4)
    assert (
        str(disp)
        == """
samples    S0   S1  ...   S8   S9
variants            ...          
0         0/0  1/0  ...  1/1  0/0
1         1/0  0/1  ...  1/0  1/1
...       ...  ...  ...  ...  ...
8         0/1  0/0  ...  1/0  1/0
9         1/1  0/1  ...  1/1  1/0

[10 rows x 10 columns]
""".strip()  # noqa: W291
    )

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
    assert (
        str(disp)
        == """
samples    S0   S1  ... S998 S999
variants            ...          
0         0/0  1/0  ...  0/1  1/1
1         1/1  1/1  ...  0/1  1/1
...       ...  ...  ...  ...  ...
99998     0/1  1/1  ...  1/0  0/1
99999     1/0  1/0  ...  1/0  1/0

[100000 rows x 1000 columns]
""".strip()  # noqa: W291
    )


def test_display_genotypes__sample_order():
    # Test that samples are not reordered lexicograpically (S10 is after S9)
    ds = simulate_genotype_call_dataset(n_variant=4, n_sample=12, seed=0)
    disp = display_genotypes(ds)
    assert (
        str(disp)
        == """samples    S0   S1   S2   S3   S4  ...   S7   S8   S9  S10  S11
variants                           ...                         
0         0/0  1/0  1/0  0/1  1/0  ...  1/0  1/1  0/0  1/0  0/1
1         1/0  1/1  1/1  1/0  1/0  ...  1/1  1/1  1/1  0/0  0/0
2         0/1  1/0  1/0  1/0  0/0  ...  0/0  0/0  0/0  0/1  0/1
3         0/1  1/0  1/0  0/0  0/1  ...  0/1  1/0  1/1  0/0  1/0

[4 rows x 12 columns]
""".strip()  # noqa: W291
    )
