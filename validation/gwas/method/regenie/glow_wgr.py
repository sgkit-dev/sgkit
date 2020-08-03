import os
import sys

# Use same conda environment for spark workers as driver env
os.environ["PYSPARK_PYTHON"] = sys.executable

import io
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import fire
import glow
import numpy as np
import pandas as pd
import pkg_resources
import pyspark.sql.functions as F
import yaml
from glow import expand_struct, genotype_states, linear_regression_gwas, mean_substitute
from glow.wgr.functions import (
    block_variants_and_samples,
    check_argument_types,
    get_sample_ids,
)
from glow.wgr.linear_model import RidgeReducer, RidgeRegression
from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession

logging.config.fileConfig("logging.ini")
logger = logging.getLogger(__name__)

HR = "-" * 50

glow_version = pkg_resources.get_distribution("glow.py").version


def _info(df):
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()


def _schema(df):
    return df._jdf.schema().treeString()


def spark_session():
    spark = (
        SparkSession.builder.config(
            "spark.jars.packages", "io.projectglow:glow_2.11:0.5.0"
        )
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    glow.register(spark)
    return spark


def _flatten_reduced_blocks(df):
    return (
        df.select("*", F.posexplode("values"))
        .withColumnRenamed("pos", "sample_value_index")
        .withColumnRenamed("col", "sample_value")
        .drop("values")
    )


def reshape_for_gwas(spark, label_df):
    # https://github.com/projectglow/glow/blob/04257f65ad64b45b2ad4a9417292e0ead6f94212/python/glow/wgr/functions.py
    assert check_argument_types()

    if label_df.index.nlevels == 1:  # Indexed by sample id
        transposed_df = label_df.T
        column_names = ["label", "values"]
    elif label_df.index.nlevels == 2:  # Indexed by sample id and contig name
        # stacking sorts the new column index, so we remember the original sample
        # ordering in case it's not sorted
        ordered_cols = pd.unique(label_df.index.get_level_values(0))
        transposed_df = label_df.T.stack()[ordered_cols]
        column_names = ["label", "contigName", "values"]
    else:
        raise ValueError(
            "label_df must be indexed by sample id or by (sample id, contig name)"
        )

    transposed_df["values_array"] = transposed_df.to_numpy().tolist()
    return spark.createDataFrame(
        transposed_df[["values_array"]].reset_index(), column_names
    )


def infer_chromosomes(blockdf: DataFrame) -> List[str]:
    # From: https://github.com/projectglow/glow/blob/master/python/glow/wgr/linear_model/functions.py#L328
    # Regex captures the chromosome name in the header
    # level 1 header: chr_3_block_8_alpha_0_label_sim100
    # level 2 header: chr_3_alpha_0_label_sim100
    chromosomes = [
        r.chromosome
        for r in blockdf.select(
            F.regexp_extract("header", r"^chr_(.+?)_(alpha|block)", 1).alias(
                "chromosome"
            )
        )
        .distinct()
        .collect()
    ]
    print(f"Inferred chromosomes: {chromosomes}")
    return chromosomes


def transform_loco(
    self,
    blockdf: DataFrame,
    labeldf: pd.DataFrame,
    sample_blocks: Dict[str, List[str]],
    modeldf: DataFrame,
    cvdf: DataFrame,
    covdf: pd.DataFrame = pd.DataFrame({}),
    chromosomes: List[str] = [],
) -> pd.DataFrame:
    # From https://github.com/projectglow/glow/blob/master/python/glow/wgr/linear_model/ridge_model.py#L320
    loco_chromosomes = chromosomes if chromosomes else infer_chromosomes(blockdf)
    loco_chromosomes.sort()

    all_y_hat_df = pd.DataFrame({})
    for chromosome in loco_chromosomes:
        loco_model_df = modeldf.filter(
            ~F.col("header").rlike(f"^chr_{chromosome}_(alpha|block)")
        )
        loco_y_hat_df = self.transform(
            blockdf, labeldf, sample_blocks, loco_model_df, cvdf, covdf
        )
        loco_y_hat_df["contigName"] = chromosome
        all_y_hat_df = all_y_hat_df.append(loco_y_hat_df)
    return all_y_hat_df.set_index("contigName", append=True)


def run(
    plink_path: str,
    traits_path: str,
    covariates_path: str,
    variants_per_block: int,
    sample_block_count: int,
    output_dir: str,
    plink_fam_sep: str = "\t",
    plink_bim_sep: str = "\t",
    alphas: Optional[list] = None,
    contigs: List[str] = None,
):
    """Run Glow WGR"""
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=False)

    if alphas is None:
        alphas = np.array([])
    else:
        alphas = np.array(alphas).astype(float)

    spark = spark_session()
    logger.info(
        f"Loading PLINK dataset at {plink_path} (fam sep = {plink_fam_sep}, bim sep = {plink_bim_sep}, alphas = {alphas})"
    )
    df = (
        spark.read.format("plink")
        .option("bimDelimiter", plink_bim_sep)
        .option("famDelimiter", plink_fam_sep)
        .option("includeSampleIds", True)
        .option("mergeFidIid", False)
        .load(plink_path)
    )

    variant_df = df.withColumn(
        "values", mean_substitute(genotype_states(F.col("genotypes")))
    ).filter(F.size(F.array_distinct("values")) > 1)
    if contigs is not None:
        variant_df = variant_df.filter(F.col("contigName").isin(contigs))

    sample_ids = get_sample_ids(variant_df)
    logger.info(f"Found {len(sample_ids)} samples, first 10: {sample_ids[:10]}")

    ###########
    # Stage 1 #
    ###########

    logger.info(HR)
    logger.info("Calculating variant/sample block info")
    block_df, sample_blocks = block_variants_and_samples(
        variant_df,
        sample_ids,
        variants_per_block=variants_per_block,
        sample_block_count=sample_block_count,
    )

    label_df = pd.read_csv(traits_path, index_col="sample_id")
    label_df = (label_df - label_df.mean()) / label_df.std(ddof=0)
    logger.info(HR)
    logger.info("Trait info:")
    logger.info(_info(label_df))

    cov_df = pd.read_csv(covariates_path, index_col="sample_id")
    cov_df = (cov_df - cov_df.mean()) / cov_df.std(ddof=0)
    logger.info(HR)
    logger.info("Covariate info:")
    logger.info(_info(cov_df))

    stack = RidgeReducer(alphas=alphas)
    reduced_block_df = stack.fit_transform(block_df, label_df, sample_blocks, cov_df)
    logger.info(HR)
    logger.info("Stage 1: Reduced block schema:")
    logger.info(_schema(reduced_block_df))

    path = output_path / "reduced_blocks.parquet"
    reduced_block_df.write.parquet(str(path), mode="overwrite")
    logger.info(f"Stage 1: Reduced blocks written to {path}")

    # Flatten to scalars for more convenient access w/o Spark
    flat_reduced_block_df = spark.read.parquet(str(path))
    path = output_path / "reduced_blocks_flat.csv.gz"
    flat_reduced_block_df = _flatten_reduced_blocks(flat_reduced_block_df)
    flat_reduced_block_df = flat_reduced_block_df.toPandas()
    flat_reduced_block_df.to_csv(path, index=False)
    # flat_reduced_block_df.write.parquet(str(path), mode='overwrite')
    logger.info(f"Stage 1: Flattened reduced blocks written to {path}")

    ###########
    # Stage 2 #
    ###########

    # Monkey-patch this in until there's a glow release beyond 0.5.0
    if glow_version != "0.5.0":
        raise NotImplementedError(
            f"Must remove adjustements for glow != 0.5.0 (found {glow_version})"
        )
    # Remove after glow update
    RidgeRegression.transform_loco = transform_loco
    estimator = RidgeRegression(alphas=alphas)
    model_df, cv_df = estimator.fit(reduced_block_df, label_df, sample_blocks, cov_df)
    logger.info(HR)
    logger.info("Stage 2: Model schema:")
    logger.info(_schema(model_df))
    logger.info("Stage 2: CV schema:")
    logger.info(_schema(cv_df))

    y_hat_df = estimator.transform(
        reduced_block_df, label_df, sample_blocks, model_df, cv_df, cov_df
    )

    logger.info(HR)
    logger.info("Stage 2: Prediction info:")
    logger.info(_info(y_hat_df))
    logger.info(y_hat_df.head(5))

    path = output_path / "predictions.csv"
    y_hat_df.reset_index().to_csv(path, index=False)
    logger.info(f"Stage 2: Predictions written to {path}")

    # y_hat_df_loco = estimator.transform_loco(reduced_block_df, label_df, sample_blocks, model_df, cv_df, cov_df)

    # path = output_path / 'predictions_loco.csv'
    # y_hat_df_loco.reset_index().to_csv(path, index=False)
    # logger.info(f'Stage 2: LOCO Predictions written to {path}')

    ###########
    # Stage 3 #
    ###########

    # Do this to correct for the error in Glow at https://github.com/projectglow/glow/issues/257
    if glow_version != "0.5.0":
        raise NotImplementedError(
            f"Must remove adjustements for glow != 0.5.0 (found {glow_version})"
        )
    cov_arr = cov_df.to_numpy()
    cov_arr = cov_arr.T.ravel(order="C").reshape(cov_arr.shape)

    # Convert the pandas dataframe into a Spark DataFrame
    adjusted_phenotypes = reshape_for_gwas(spark, label_df - y_hat_df)

    # Run GWAS w/o LOCO (this could be for a much larger set of variants)
    wgr_gwas = (
        variant_df.withColumnRenamed("values", "callValues")
        .crossJoin(adjusted_phenotypes.withColumnRenamed("values", "phenotypeValues"))
        .select(
            "start",
            "names",
            "label",
            expand_struct(
                linear_regression_gwas(
                    F.col("callValues"), F.col("phenotypeValues"), F.lit(cov_arr)
                )
            ),
        )
    )

    logger.info(HR)
    logger.info("Stage 3: GWAS (no LOCO) schema:")
    logger.info(_schema(wgr_gwas))

    # Convert to pandas
    wgr_gwas = wgr_gwas.toPandas()
    logger.info(HR)
    logger.info("Stage 3: GWAS (no LOCO) info:")
    logger.info(_info(wgr_gwas))
    logger.info(wgr_gwas.head(5))

    path = output_path / "gwas.csv"
    wgr_gwas.to_csv(path, index=False)
    logger.info(f"Stage 3: GWAS (no LOCO) results written to {path}")
    logger.info(HR)
    logger.info("Done")

    # TODO: Enable this once WGR is fully released
    # See: https://github.com/projectglow/glow/issues/256)

    # Run GWAS w/ LOCO
    # adjusted_phenotypes = reshape_for_gwas(spark, label_df - y_hat_df_loco)
    # wgr_gwas = (
    #     variant_df
    #     .withColumnRenamed('values', 'callValues')
    #     .join(
    #         adjusted_phenotypes
    #         .withColumnRenamed('values', 'phenotypeValues'),
    #         ['contigName']
    #     )
    #     .select(
    #         'contigName',
    #         'start',
    #         'names',
    #         'label',
    #         expand_struct(linear_regression_gwas(
    #             F.col('callValues'),
    #             F.col('phenotypeValues'),
    #             F.lit(cov_arr)
    #         ))
    #     )
    # )

    # # Convert to pandas
    # wgr_gwas = wgr_gwas.toPandas()
    # logger.info(HR)
    # logger.info('Stage 3: GWAS (with LOCO) info:')
    # logger.info(_info(wgr_gwas))
    # logger.info(wgr_gwas.head(5))

    # path = output_path / 'gwas_loco.csv'
    # wgr_gwas.to_csv(path, index=False)
    # logger.info(f'Stage 3: GWAS (with LOCO) results written to {path}')
    # logger.info(HR)
    # logger.info('Done')


def run_from_config(
    dataset: str,
    paramset: str,
    dataset_dir: str = "data/dataset",
    output_dir: str = "data/result",
):
    with open("config.yml") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    ds_config = config["datasets"][dataset]
    ps_config = config["paramsets"][paramset]
    logger.info(f"Loaded config for dataset {dataset}: {ds_config}")
    logger.info(f"Loaded config for paramset {paramset}: {ps_config}")
    dataset_dir = Path(dataset_dir) / dataset
    sample_block_count = ds_config["n_samples"] // ps_config["sample_block_size"]
    run(
        plink_path=str(dataset_dir / "genotypes.bed"),
        traits_path=str(dataset_dir / "traits.csv"),
        covariates_path=str(dataset_dir / "covariates.csv"),
        variants_per_block=ps_config["variant_block_size"],
        sample_block_count=sample_block_count,
        output_dir=str(Path(output_dir) / f"{dataset}-{paramset}"),
        alphas=ps_config["alphas"],
    )


fire.Fire()
