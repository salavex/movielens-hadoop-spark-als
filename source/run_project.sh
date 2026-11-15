#!/usr/bin/env bash
# Run on the existing Ubuntu/Docker Hadoop environment. No HDFS formatting.
set -euo pipefail
SOURCE_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SOURCE_DIR/.."
mkdir -p data results
if [ ! -f data/ml-latest-small/ratings.csv ]; then
  if [ ! -f data/ml-latest-small.zip ]; then
    curl --fail --location https://files.grouplens.org/datasets/movielens/ml-latest-small.zip -o data/ml-latest-small.zip
  fi
  python3 -m zipfile -e data/ml-latest-small.zip data
fi
docker inspect master >/dev/null
docker exec master test -x /usr/local/spark/bin/spark-submit
docker exec master hdfs dfs -test -d /
docker exec master mkdir -p /project/results
docker cp "$SOURCE_DIR/movie_recommender.py" master:/project/movie_recommender.py
docker cp data/ml-latest-small master:/project/data
docker exec master hdfs dfs -mkdir -p /course-movie/raw
docker exec master hdfs dfs -put -f /project/data/ratings.csv /project/data/movies.csv /course-movie/raw/
docker exec master bash -lc '
  export SPARK_HOME=/usr/local/spark
  export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop
  export PYSPARK_PYTHON=/usr/bin/python3
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  /usr/local/spark/bin/spark-submit --master "local[2]" --driver-memory 2g \
    /project/movie_recommender.py --output /project/results > /project/run.log 2>&1
'
docker exec master hdfs dfs -get -f /course-movie/output/als_model /project/results/
docker cp master:/project/results/. results/
docker cp master:/project/run.log results/run.log
echo 'Completed. Read results/results.json and results/recommendations.csv.'
