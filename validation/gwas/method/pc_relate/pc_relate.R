#!/usr/bin/env Rscript

library(gdsfmt)
library(SNPRelate)
library(GWASTools)
library(GENESIS)
library(tictoc)

args <- commandArgs(trailingOnly=TRUE)

if (length(args) < 1) {
  stop("usage: <gds_file>", call.=FALSE)
}

gds_filepath = args[1]

tic("KING kinship")
genofile <- snpgdsOpen(gds_filepath)
king_mat <- snpgdsIBDKING(genofile, num.thread=8)
king_mat_2 = kingToMatrix(king_mat)
toc(log = TRUE)
snpgdsClose(genofile)

reader <- GdsGenotypeReader(gds_filepath, "scan,snp")
geno_data <- GenotypeData(reader)

tic("PC-AIR")
pcair_result <- pcair(geno_data,
                      kinobj = king_mat_2,
                      divobj = king_mat_2)
toc(log = TRUE)
summary(pcair_result)

write.csv(pcair_result$vectors[,1:2], file = "pcs.csv")

geno_data <- GenotypeBlockIterator(geno_data)
tic("PC-Relate")
pcrelate_result <- pcrelate(geno_data,
                            pcs = pcair_result$vectors[,1:2])
toc(log = TRUE)


write.csv(pcrelate_result$kinSelf, "kinself.csv")
write.csv(pcrelate_result$kinBtwn, "kinbtwe.csv")
write.csv(pcair_result$unrels, "unrels.csv")
write.csv(pcair_result$rels, "rels.csv")
summary(pcrelate_result)
