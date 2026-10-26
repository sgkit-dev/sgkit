import typing

import dask.array as da
import numpy as np
import pytest
from scipy.spatial.distance import correlation, euclidean, pdist, squareform

from sgkit.distance.api import DeviceTypes, MetricTypes, pairwise_distance
from sgkit.typing import ArrayLike


def detect_gpu() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except (ImportError, RuntimeError):
        return False


def skip_gpu_tests_if_no_gpu(device: DeviceTypes) -> None:
    if device == "gpu" and not detect_gpu():
        pytest.skip("CuPy or a GPU not found")


def get_vectors(
    array_type: str = "da",
    dtype: str = "i8",
    size: typing.Tuple[int, int] = (100, 100),
    chunk: typing.Tuple[int, int] = (20, 10),
) -> ArrayLike:
    if array_type == "da":
        rs = da.random.RandomState(0)
        x = rs.randint(0, 3, size=size).astype(dtype).rechunk(chunk)
    else:
        rs = np.random.RandomState(0)  # type: ignore
        x = rs.rand(size[0], size[1]).astype(dtype)  # type: ignore
    return x


def create_distance_matrix(
    x: ArrayLike, metric_func: typing.Callable[[ArrayLike, ArrayLike], np.float64]
) -> ArrayLike:
    """
    Parameters
    ----------
    x
        [array-like, shape: (M, N)]
        An array like two dimensional matrix. The rows are the
        vectors used for comparison, i.e. for pairwise distance.
    metric_func
        metric function for the distance metric.

    Returns
    -------
    A two dimensional distance matrix.

    """
    m = x.shape[0]
    distance_matrix = np.zeros((m, m), dtype=np.float64)
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            k = np.stack([x[i], x[j]])
            k = k[:, k.min(axis=0) >= 0]
            vi, vj = k[0], k[1]
            try:
                distance_matrix[i][j] = metric_func(vi, vj)
            except RuntimeWarning:
                # unable to calculate distance metric which
                # which means array contains only one element or
                # not possible to calculate distance metric
                distance_matrix[i][j] = np.nan
    return distance_matrix


@pytest.mark.parametrize(
    "size, chunk, device, metric",
    [
        pytest.param(
            shape,
            chunks,
            device,
            metric,
            marks=pytest.mark.gpu if device == "gpu" else "",  # type: ignore[arg-type]
        )
        for shape in [(30, 30), (15, 30), (30, 15)]
        for chunks in [(10, 10), (5, 10), (10, 5)]
        for device in ["cpu", "gpu"]
        for metric in ["euclidean", "correlation"]
    ],
)
def test_pairwise(
    size: typing.Tuple[int, int],
    chunk: typing.Tuple[int, int],
    device: DeviceTypes,
    metric: MetricTypes,
) -> None:
    skip_gpu_tests_if_no_gpu(device)
    x = get_vectors(size=size, chunk=chunk)
    distance_matrix = pairwise_distance(x, metric=metric, device=device)
    distance_array = pdist(x, metric=metric)
    expected_matrix = squareform(distance_array)
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


@pytest.mark.parametrize(
    "size, chunk, split_every, metric",
    [
        ((100, 100), (25, 10), 5, "euclidean"),
        ((100, 100), (20, 25), 3, "euclidean"),
        ((100, 100), (25, 10), 5, "correlation"),
        ((100, 100), (20, 25), 3, "correlation"),
    ],
)
def test_pairwise_split_every(
    size: typing.Tuple[int, int],
    chunk: typing.Tuple[int, int],
    split_every: int,
    metric: MetricTypes,
) -> None:
    x = get_vectors(size=size, chunk=chunk)
    distance_matrix = pairwise_distance(x, metric=metric, split_every=split_every)
    expected_matrix = squareform(pdist(x, metric=metric))
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


def test_distance_ndarray() -> None:
    x = get_vectors(array_type="np")
    distance_matrix = pairwise_distance(x, metric="euclidean")
    expected_matrix = squareform(pdist(x))
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


@pytest.mark.parametrize(
    "metric, metric_func, dtype, device",
    [
        ("euclidean", euclidean, "f8", "cpu"),
        ("euclidean", euclidean, "i8", "cpu"),
        pytest.param("euclidean", euclidean, "f8", "gpu", marks=pytest.mark.gpu),
        pytest.param("euclidean", euclidean, "i8", "gpu", marks=pytest.mark.gpu),
        ("correlation", correlation, "f8", "cpu"),
        ("correlation", correlation, "i8", "cpu"),
        pytest.param("correlation", correlation, "f8", "gpu", marks=pytest.mark.gpu),
        pytest.param("correlation", correlation, "i8", "gpu", marks=pytest.mark.gpu),
    ],
)
def test_missing_values(
    metric: MetricTypes,
    metric_func: typing.Callable[[ArrayLike, ArrayLike], np.float64],
    dtype: str,
    device: DeviceTypes,
) -> None:
    skip_gpu_tests_if_no_gpu(device)
    x = get_vectors(array_type="np", dtype=dtype)

    ri_times = np.random.randint(5, 20)
    m, n = x.shape
    for i in range(ri_times):
        if dtype == "f8":
            x[np.random.randint(0, m)][np.random.randint(0, m)] = np.nan
        x[np.random.randint(0, m)][np.random.randint(0, m)] = np.random.randint(
            -100, -1
        )

    distance_matrix = pairwise_distance(x, metric=metric, device=device)
    expected_matrix = create_distance_matrix(x, metric_func)
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


@pytest.mark.parametrize(
    "metric, metric_func",
    [("euclidean", euclidean), ("correlation", correlation)],
)
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_genotype_calls(
    metric: MetricTypes,
    metric_func: typing.Callable[[ArrayLike, ArrayLike], np.float64],
    device: DeviceTypes,
) -> None:
    skip_gpu_tests_if_no_gpu(device)
    # int8 calls with -1 for missing values, as in call_genotype
    rs = np.random.RandomState(0)
    x = rs.randint(0, 3, size=(30, 300)).astype(np.int8)
    x[rs.rand(*x.shape) < 0.1] = -1
    x[0] = -1  # no observed values
    x[1] = 1  # constant
    x[2, 2:] = -1  # two observed values

    distance_matrix = pairwise_distance(
        da.from_array(x, chunks=(10, 150)), metric=metric, device=device
    )
    expected_matrix = create_distance_matrix(x, metric_func)
    np.testing.assert_almost_equal(distance_matrix, expected_matrix)


@pytest.mark.parametrize("metric", ["euclidean", "correlation"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_identical_genotype_calls(metric: MetricTypes, device: DeviceTypes) -> None:
    skip_gpu_tests_if_no_gpu(device)
    # the sums of squares and products of integers are exact, so copies of a
    # row, in the same chunk or in another one, are exactly zero apart
    rs = np.random.RandomState(0)
    x = rs.randint(0, 3, size=(4, 100_000)).astype(np.int8)
    x[rs.rand(*x.shape) < 0.05] = -1
    x[1] = x[0]
    x[3] = x[0]

    distance_matrix = pairwise_distance(
        da.from_array(x, chunks=(2, 25_000)), metric=metric, device=device
    ).compute()
    assert distance_matrix[0, 1] == distance_matrix[0, 3] == 0
    assert distance_matrix[1, 3] == 0


@pytest.mark.parametrize(
    "metric, metric_func, atol",
    [("euclidean", euclidean, 1e-3), ("correlation", correlation, 1e-6)],
)
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_nearly_identical_float32_rows(
    metric: MetricTypes,
    metric_func: typing.Callable[[ArrayLike, ArrayLike], np.float64],
    atol: float,
    device: DeviceTypes,
) -> None:
    skip_gpu_tests_if_no_gpu(device)
    # float32 dosages, where the second row is a copy of the first and the third
    # differs from it in five entries. When the sums of squares and products
    # were subtracted in float32, these distances were lost.
    rs = np.random.RandomState(0)
    x = rs.uniform(0, 2, size=(4, 100_000)).round(3).astype(np.float32)
    x[rs.rand(*x.shape) < 0.05] = np.nan
    x[1] = x[0]
    x[2] = x[0]
    x[2, :5] += 0.05

    distance_matrix = pairwise_distance(
        da.from_array(x, chunks=(2, 25_000)), metric=metric, device=device
    ).compute()
    expected_matrix = create_distance_matrix(x.astype(np.float64), metric_func)
    np.testing.assert_allclose(distance_matrix, expected_matrix, rtol=1e-6, atol=atol)
    assert distance_matrix.min() >= 0


@pytest.mark.parametrize("metric", ["euclidean", "correlation"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_float16(metric: MetricTypes, device: DeviceTypes) -> None:
    skip_gpu_tests_if_no_gpu(device)
    # the sums of squares and products are larger than the float16 maximum
    x = np.array([[0, 2] * 10_000, [2, 0] * 10_000, [0, 2] * 10_000], dtype="f2")

    distance_matrix = pairwise_distance(x, metric=metric, device=device)
    assert distance_matrix.dtype == np.float16
    expected_matrix = squareform(pdist(x.astype(np.float64), metric=metric))
    np.testing.assert_allclose(distance_matrix, expected_matrix, rtol=1e-3)


@pytest.mark.parametrize(
    "metric, expected_matrix",
    [
        ("euclidean", np.sqrt([[0, 2, 2], [2, 0, 0], [2, 0, 0]])),
        ("correlation", [[0, 2, 2], [2, 0, 0], [2, 0, 0]]),
    ],
)
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_infinite_values(
    metric: MetricTypes, expected_matrix: ArrayLike, device: DeviceTypes
) -> None:
    skip_gpu_tests_if_no_gpu(device)
    # infinite values are missing, like negative and nan values, so the first
    # column is only observed in the last row. In the matrix products, the
    # infinite value made every distance of the first row nan.
    x = np.array([[np.inf, 0, 1], [-1, 1, 0], [2, 1, 0]])

    distance_matrix = pairwise_distance(x, metric=metric, device=device)
    np.testing.assert_allclose(distance_matrix, expected_matrix)


@pytest.mark.gpu
@pytest.mark.parametrize("metric", ["euclidean", "correlation"])
def test_cupy_backed_array(metric: MetricTypes) -> None:
    cupy = pytest.importorskip("cupy")
    skip_gpu_tests_if_no_gpu("gpu")
    x = get_vectors()
    distance_matrix = pairwise_distance(x.map_blocks(cupy.asarray), metric=metric)
    assert isinstance(distance_matrix._meta, cupy.ndarray)
    expected_matrix = squareform(pdist(x, metric=metric))
    np.testing.assert_almost_equal(
        cupy.asnumpy(distance_matrix.compute()), expected_matrix
    )


@pytest.mark.parametrize(
    "metric, dtype, expected, device",
    [
        ("euclidean", "i1", "float64", "cpu"),
        ("euclidean", "i8", "float64", "cpu"),
        ("euclidean", "f4", "float32", "cpu"),
        ("euclidean", "f8", "float64", "cpu"),
        pytest.param("euclidean", "i8", "float64", "gpu", marks=pytest.mark.gpu),
        pytest.param("euclidean", "f4", "float32", "gpu", marks=pytest.mark.gpu),
        pytest.param("euclidean", "f8", "float64", "gpu", marks=pytest.mark.gpu),
        ("correlation", "i1", "float64", "cpu"),
        ("correlation", "i8", "float64", "cpu"),
        ("correlation", "f4", "float32", "cpu"),
        ("correlation", "f8", "float64", "cpu"),
        pytest.param("correlation", "i8", "float64", "gpu", marks=pytest.mark.gpu),
        pytest.param("correlation", "f4", "float32", "gpu", marks=pytest.mark.gpu),
        pytest.param("correlation", "f8", "float64", "gpu", marks=pytest.mark.gpu),
    ],
)
def test_data_types(
    metric: MetricTypes, dtype: str, expected: str, device: DeviceTypes
) -> None:
    skip_gpu_tests_if_no_gpu(device)
    x = get_vectors(dtype=dtype)
    distance_matrix = pairwise_distance(x, metric=metric, device=device)
    assert distance_matrix.dtype.name == expected
    assert distance_matrix.compute().dtype.name == expected


def test_undefined_metric() -> None:
    x = get_vectors(array_type="np")
    with pytest.raises(NotImplementedError):
        pairwise_distance(x, metric="not-implemented-metric")  # type: ignore[arg-type]


def test_invalid_device() -> None:
    x = get_vectors(array_type="np")
    with pytest.raises(ValueError):
        pairwise_distance(x, device="invalid-device")  # type: ignore[arg-type]


def test_wrong_dimension_array() -> None:
    with pytest.raises(ValueError):
        pairwise_distance(da.arange(6).reshape(1, 2, 3))

    with pytest.raises(ValueError):
        pairwise_distance(da.arange(10))
