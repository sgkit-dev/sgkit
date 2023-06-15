This runs test to validate our implementation gets the same
exact results as the reference R implementation for production
HapMap data.

This code is scheduled as part of the Github Actions CI.

To run manually, you need to first download the test data
from `https://storage.googleapis.com/sgkit-data/validation/hapmap_JPT_CHB_r23a_filtered.zip`,
the file size is about 32MB.

```bash
wget https://storage.googleapis.com/sgkit-data/validation/hapmap_JPT_CHB_r23a_filtered.zip -P /tmp/
./run.sh /tmp/hapmap_JPT_CHB_r23a_filtered.zip
```

`run.sh` will:
 * convert plink data to GDS
 * run reference [R PC-Relate implementation](pc_relate.R)  
 * run [our PC-Relate and compare results](validate_pc_relate.py)

The only requirement is that you have Docker and Bash installed.
