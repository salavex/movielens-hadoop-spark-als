"""MovieLens offline recommendation with HDFS and Spark MLlib.
Run: spark-submit --master local[2] movie_recommender.py --output /project/results
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='hdfs://master:9000/course-movie/raw')
    p.add_argument('--hdfs-output', default='hdfs://master:9000/course-movie/output')
    p.add_argument('--output', default='/project/results')
    args = p.parse_args()
    os.makedirs(args.output, exist_ok=True)
    spark = (SparkSession.builder.appName('MovieLens-ALS-Course-Project')
             .config('spark.sql.shuffle.partitions', '4')
             .config('spark.default.parallelism', '2').getOrCreate())
    spark.sparkContext.setLogLevel('WARN')
    started = time.perf_counter()
    rating_schema = T.StructType([
        T.StructField('userId', T.IntegerType()), T.StructField('movieId', T.IntegerType()),
        T.StructField('rating', T.DoubleType()), T.StructField('timestamp', T.LongType())])
    movie_schema = 'movieId INT, title STRING, genres STRING'
    raw = spark.read.option('header', True).schema(rating_schema).csv(args.input + '/ratings.csv').cache()
    movies = spark.read.option('header', True).schema(movie_schema).csv(args.input + '/movies.csv').cache()
    raw_count = raw.count()
    valid = raw.dropna(subset=['userId', 'movieId', 'rating', 'timestamp']).filter(
        (F.col('userId') > 0) & (F.col('movieId') > 0) & F.col('rating').between(0.5, 5.0))
    valid_count = valid.count()
    # GroupLens data has one rating per user/movie. Retain latest timestamp if duplicates exist.
    from pyspark.sql.window import Window
    window = Window.partitionBy('userId', 'movieId').orderBy(F.desc('timestamp'), F.desc('rating'))
    clean = (valid.withColumn('_row', F.row_number().over(window)).filter('_row = 1').drop('_row')
             .join(movies.select('movieId').distinct(), 'movieId', 'left_semi').cache())
    clean_count = clean.count()
    clean.write.mode('overwrite').parquet(args.hdfs_output + '/clean_ratings')

    def rows(df):
        return [r.asDict(recursive=True) for r in df.collect()]

    stats = {
        'raw_ratings': raw_count, 'valid_before_dedup': valid_count, 'clean_ratings': clean_count,
        'invalid_rows': raw_count-valid_count, 'removed_total': raw_count-clean_count,
        'users': clean.select('userId').distinct().count(), 'catalog_movies': movies.count(),
        'rated_movies': clean.select('movieId').distinct().count(),
        'mean_rating': clean.agg(F.avg('rating')).first()[0],
        'min_timestamp': clean.agg(F.min('timestamp')).first()[0],
        'max_timestamp': clean.agg(F.max('timestamp')).first()[0],
    }
    distribution = rows(clean.groupBy('rating').count().orderBy('rating'))
    popular = rows(clean.groupBy('movieId').agg(F.count('*').alias('rating_count'),
        F.avg('rating').alias('mean_rating')).join(movies, 'movieId')
        .orderBy(F.desc('rating_count'), F.asc('movieId')).limit(10))
    print('DATA_QUALITY ' + json.dumps(stats), flush=True)
    # Stable key hashing avoids randomSplit recomputation differences across execution plans.
    assigned = clean.withColumn('_split', F.pmod(F.xxhash64('userId', 'movieId', F.lit(42)), F.lit(100))).cache()
    train = assigned.filter('_split < 80').drop('_split').cache()
    validation = assigned.filter('_split >= 80 AND _split < 90').drop('_split').cache()
    test = assigned.filter('_split >= 90').drop('_split').cache()
    split_counts = {'train': train.count(), 'validation': validation.count(), 'test': test.count()}
    assert sum(split_counts.values()) == clean_count, 'Split lost rows'
    assert train.join(validation, ['userId', 'movieId']).count() == 0
    assert train.join(test, ['userId', 'movieId']).count() == 0
    assert validation.join(test, ['userId', 'movieId']).count() == 0
    print('SPLIT_COUNTS ' + json.dumps(split_counts), flush=True)

    def evaluate(predictions, total):
        usable = predictions.filter(F.col('prediction').isNotNull() & ~F.isnan('prediction')).cache()
        n = usable.count()
        if n == 0:
            raise RuntimeError('No usable predictions')
        result = {'rmse': RegressionEvaluator(labelCol='rating', metricName='rmse').evaluate(usable),
                  'mae': RegressionEvaluator(labelCol='rating', metricName='mae').evaluate(usable),
                  'predicted_rows': n, 'total_rows': total, 'coverage': n / total}
        usable.unpersist()
        return result

    experiments, fitted = [], []
    for rank, reg in [(10, 0.1), (20, 0.15)]:
        als = ALS(userCol='userId', itemCol='movieId', ratingCol='rating',
                  rank=rank, regParam=reg, maxIter=10, seed=42,
                  implicitPrefs=False, nonnegative=False, coldStartStrategy='drop',
                  numUserBlocks=2, numItemBlocks=2)
        tick = time.perf_counter()
        model = als.fit(train)
        fit_seconds = time.perf_counter()-tick
        metrics = evaluate(model.transform(validation), split_counts['validation'])
        metrics.update(rank=rank, regParam=reg, maxIter=10, fit_seconds=fit_seconds)
        experiments.append(metrics)
        fitted.append(model)
        print('VALIDATION_RESULT ' + json.dumps(metrics), flush=True)
    best_index = min(range(len(experiments)), key=lambda i: experiments[i]['rmse'])
    best = fitted[best_index]
    test_predictions = best.transform(test).cache()
    test_metrics = evaluate(test_predictions, split_counts['test'])
    # Compare on exactly the same cold-start-eligible test observations.
    eligible = test_predictions.filter(F.col('prediction').isNotNull() & ~F.isnan('prediction'))
    train_mean = train.agg(F.avg('rating')).first()[0]
    baseline = evaluate(eligible.drop('prediction').withColumn('prediction', F.lit(train_mean)), split_counts['test'])
    test_metrics.update(model='ALS', rank=experiments[best_index]['rank'], regParam=experiments[best_index]['regParam'])
    baseline.update(model='Training mean', training_mean=train_mean)
    print('TEST_RESULT ' + json.dumps(test_metrics), flush=True)
    print('BASELINE_RESULT ' + json.dumps(baseline), flush=True)
    best.write().overwrite().save(args.hdfs_output + '/als_model')

    # A small popularity threshold limits unstable recommendations for rarely rated movies.
    candidates = train.groupBy('movieId').count().filter('count >= 20').select('movieId')
    selected_users = [1, 414]
    users_df = spark.createDataFrame([(u,) for u in selected_users], 'userId INT')
    all_seen = clean.select('userId', 'movieId')
    pairs = users_df.crossJoin(candidates).join(all_seen, ['userId', 'movieId'], 'left_anti')
    predicted = best.transform(pairs).filter(~F.isnan('prediction')).join(movies, 'movieId')
    ranked = predicted.withColumn('position', F.row_number().over(
        Window.partitionBy('userId').orderBy(F.desc('prediction'), F.asc('movieId'))))
    recommendations_df = ranked.filter('position <= 10').select('userId', 'position', 'movieId', 'title', 'genres', 'prediction')
    recommendations = rows(recommendations_df.orderBy('userId', 'position'))
    # Runtime contracts check outcomes rather than mirroring the algorithm implementation.
    assert recommendations_df.join(all_seen, ['userId', 'movieId'], 'inner').count() == 0, 'Seen movie leaked'
    assert len(recommendations) == 20, 'Expected two complete Top 10 lists'
    assert sum(split_counts.values()) == clean_count, 'Split lost rows'
    recommendations_df.write.mode('overwrite').parquet(args.hdfs_output + '/recommendations')
    histories = {}
    for uid in selected_users:
        histories[str(uid)] = rows(clean.filter(F.col('userId') == uid).join(movies, 'movieId')
            .orderBy(F.desc('rating'), F.desc('timestamp')).select('title', 'genres', 'rating').limit(5))
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'spark_version': spark.version,
        'master': spark.sparkContext.master, 'application_id': spark.sparkContext.applicationId,
        'dataset_url': 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
        'seed': 42, 'split_ratio': [0.8,0.1,0.1], 'split_method': 'pmod(xxhash64(userId,movieId,42),100)', 'split_counts': split_counts, 'data': stats,
        'rating_distribution': distribution, 'popular_movies': popular, 'validation_experiments': experiments,
        'selected_index': best_index, 'test_metrics': test_metrics, 'baseline_metrics': baseline,
        'recommendations': recommendations, 'histories': histories,
        'candidate_min_train_ratings': 20, 'candidate_movies': candidates.count(),
        'checks': {'no_seen_recommendations': True, 'two_complete_top10': True, 'split_row_count': True, 'split_no_overlap': True},
        'elapsed_seconds': time.perf_counter()-started}
    with open(args.output + '/results.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(args.output + '/recommendations.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['userId', 'position', 'movieId', 'title', 'genres', 'prediction'])
        writer.writeheader(); writer.writerows(recommendations)
    print('RECOMMENDATIONS')
    recommendations_df.orderBy('userId','position').show(20, truncate=55)
    print('CHECKS_PASSED: unseen movies, 2 x Top10, split row count', flush=True)
    print('OUTPUT: ' + args.output + '/results.json', flush=True)
    spark.stop()


if __name__ == '__main__':
    main()
