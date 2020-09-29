# Ignore VCF files during pytest collection, so it doesn't fail if cyvcf2 isn't installed.
collect_ignore_glob = ["sgkit/io/vcf/*.py"]
