#!/usr/bin/env bash
# Install Spark into existing master only. Requires Python 3.12 and Java 11 in master.
# Matches the package versions used for the submitted experiment.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p packages
cd packages
if [ ! -f pyspark-3.5.7.tar.gz ]; then
  curl --fail --location 'https://files.pythonhosted.org/packages/cb/65/2764b1840aa6ea5f7e668a36702c3128ac54d0b861d3d57669b2453469a6/pyspark-3.5.7.tar.gz' -o pyspark-3.5.7.tar.gz
fi
python3 - <<'PY'
import json, urllib.request, os
for package, version, target, match in [
    ('numpy', '1.26.4', 'numpy.whl', lambda n: 'cp312-cp312-manylinux_2_17_x86_64' in n),
    ('setuptools', '75.8.2', 'setuptools.whl', lambda n: n.endswith('py3-none-any.whl')),
]:
    if not os.path.exists(target):
        with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json') as response:
            meta = json.load(response)
        artifact = next(item for item in meta['urls'] if match(item['filename']))
        urllib.request.urlretrieve(artifact['url'], target)
PY
if ! docker exec master test -x /usr/local/spark/bin/spark-submit; then
  tar -xzf pyspark-3.5.7.tar.gz
  mkdir -p pyspark-3.5.7/deps/python/lib
  mv pyspark-3.5.7/pyspark pyspark-3.5.7/deps/python/
  cp pyspark-3.5.7/lib/py4j-0.10.9.7-src.zip pyspark-3.5.7/deps/python/lib/
  python3 -m zipfile -c pyspark-3.5.7/deps/python/lib/pyspark.zip pyspark-3.5.7/deps/python/pyspark
  docker cp pyspark-3.5.7/deps master:/usr/local/spark
  docker exec master bash -lc 'chmod +x /usr/local/spark/bin/*'
fi
for artifact in numpy setuptools; do
  docker cp "$artifact.whl" "master:/tmp/$artifact.whl"
  docker exec master python3 -m zipfile -e "/tmp/$artifact.whl" /usr/local/lib/python3.12/dist-packages
done
docker exec master /usr/local/spark/bin/spark-submit --version
docker exec master python3 -c 'import numpy, distutils; print("NumPy", numpy.__version__)'
