#!/usr/bin/env Rscript

library(gdsfmt)
library(SNPRelate)

args <- commandArgs(trailingOnly=TRUE)

if (length(args) < 2) {
  stop("usage: <file_name> <out>", call.=FALSE)
}

snpgdsBED2GDS(bed.fn=paste(args[1], ".bed", sep = ""),
              bim.fn=paste(args[1], ".bim", sep = ""),
              fam.fn=paste(args[1], ".fam", sep = ""),
              out.gdsfn=args[2])
