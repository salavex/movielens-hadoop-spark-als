# 基于 Hadoop 与 Spark MLlib 的电影推荐系统

课程大作业，成员：熊浚钧，学号：524031910119。单人项目。

## 功能与架构

MovieLens CSV → HDFS → Spark 数据清洗、统计 → MLlib ALS → 离线评估与用户 Top 10 推荐。

实际环境为 Ubuntu Server 24.04.5 虚拟机（2 vCPU、8 GB），Docker 容器 master、worker1、worker2，Hadoop 3.3.6，Spark 3.5.7，Java 11.0.32.1，Python 3.12.3，NumPy 1.26.4，setuptools 75.8.2。

HDFS 有一个 NameNode、两个 DataNode。Spark 以 `local[2]` 在 master 内执行，并经 HDFS 读写数据。本项目未将 Spark 任务提交到 YARN，未开展分布式 Spark 性能实验。已有 HBase、Hive 与 MySQL 不参与本项目。

## 数据与实验

- 官方数据：https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
- 数据说明：https://grouplens.org/datasets/movielens/
- 实际 ZIP SHA-256：696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436
- 原始数据通过上方官方链接下载，包含官方 README 和使用条款。仓库不重复托管数据集。
- 100,836 条评分，610 位用户，9,742 部目录电影，其中 9,724 部有评分。评分范围 0.5–5。
- 清洗检查缺失值、ID、评分范围、重复用户电影对、电影关联。本数据未发现需删除的记录，清洗后仍为 100,836 条。
- 按 `pmod(xxhash64(userId, movieId, 42), 100)` 固定划分：0–79 为训练集，80–89 为验证集，90–99 为测试集。实际数量分别为 80,415、10,202、10,219。数量守恒和三集合两两无交叉均已运行校验。
- ALS 显式评分模型，maxIter=10，seed=42。比较 rank=10/regParam=0.10 和 rank=20/regParam=0.15，仅依据验证集 RMSE 选取后者，再评估独立测试集。没有用测试集选择参数。

### 正式运行结果

应用 ID：`local-1794648027524`，结果生成时间：2026-11-14 09:21:14 UTC（北京时间17:21:14）。

| 参数 | 验证 RMSE | 验证 MAE | 训练耗时（秒） |
|---|---:|---:|---:|
| rank=10, reg=0.10 | 0.881151 | 0.677381 | 5.42 |
| rank=20, reg=0.15 | 0.873109 | 0.679505 | 2.90 |

训练耗时来自各一次运行，第二轮可能受 JVM 预热和缓存影响，不构成模型速度或扩展性的结论。

| 模型 | 测试 RMSE | 测试 MAE |
|---|---:|---:|
| ALS | 0.881195 | 0.684580 |
| 训练评分全局均值 | 1.035563 | 0.821658 |

两个方法均在相同的 9,863 条可预测测试记录上评估。测试集共 10,219 条，ALS 有效预测覆盖率为 96.52%，另有 356 条冷启动记录被 `coldStartStrategy='drop'` 排除。均值基线本可覆盖冷启动记录，此处限制到相同子集保证误差比较口径一致。ALS 的 RMSE 相对该基线下降约 14.91%，不能据此宣称 Top K 推荐排序准确率提升14.91%。RMSE、MAE 保留模型原始预测值，不裁剪到评分范围。

总流程计时为46.74秒，范围从 SparkSession 创建之后开始，到写 JSON 之前结束，涵盖数据清洗、HDFS写出、两次拟合、评估、模型保存、推荐与校验，不包含下载和 Spark 启动。具体计时见 results.json。

推荐演示用户为1和414。候选电影须在训练集中至少有20次评分，共1,051部；候选中排除用户全部已知评分电影。完整历史只用于演示阶段排除已知电影，不用于训练或参数选择。分别输出10部电影，已校验无已评分电影混入。ALS 不限制分数上下界，用户1的部分原始预测超过5分，用作排序依据。

## 复现运行

本代码复用已有容器和HDFS，不提供全新集群的一键创建器。要求 Docker 容器 master、worker1、worker2 已启动，HDFS正常，master内存在 Java11、Python3.12及 Hadoop 命令，HDFS地址为 `hdfs://master:9000`。不要格式化已有 NameNode，也不要删除原有 Docker 数据卷。

在上述 Ubuntu 主机中解压本项目，将当前目录切换到包含 source、data、results 的项目根目录：

```bash
# 首次在master安装与实验一致的Spark和Python依赖，需要网络。
bash source/setup_spark.sh
# 执行数据上传、清洗、训练、评估和导出。
bash source/run_project.sh
```

运行脚本会从官方地址下载数据集至项目根目录的 `data/ml-latest-small.zip`。

主程序也可以在 master 中直接运行：

```bash
export SPARK_HOME=/usr/local/spark
export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop
export PYSPARK_PYTHON=/usr/bin/python3
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/usr/local/spark/bin/spark-submit --master 'local[2]' --driver-memory 2g \
  /project/movie_recommender.py --output /project/results
```

HDFS路径：

- 原始文件：`/course-movie/raw/ratings.csv`、`movies.csv`
- 清洗Parquet：`/course-movie/output/clean_ratings`
- 推荐Parquet：`/course-movie/output/recommendations`
- 模型：`/course-movie/output/als_model`

本地结果 JSON 和 CSV 在 master 的 `/project/results`。`run_project.sh` 会将结果、日志与模型复制回 Ubuntu 主机当前目录的 results。重复运行会覆盖本项目专用输出，不影响前期实验路径。

## 备份内容

- `source/`：主程序、环境安装、运行脚本与结果展示工具。
- 官方数据集：通过本页开头的官方链接下载。
- `results/`：最终 JSON、推荐 CSV、正式运行日志与模型备份压缩包。
- `evidence/`：VMware 从真实虚拟机获取的原始截图与裁剪副本。裁剪仅去除空白边缘，未修改终端内容。

`show_results.py` 读取真实结果 JSON 并在终端格式化展示，用于截图。指标页和推荐页是对已完成正式运行结果的回显，不能理解为截图时模型正在训练。原始训练日志与应用ID可用于核对。容器日志使用UTC，虚拟机控制台使用北京时间。

## 限制与资源

样本规模小，单机计算，未验证生产吞吐与横向扩展；固定哈希划分不是按时间划分；未实现冷启动回退、线上服务或Top K排序指标。两组参数仅构成基础对比，未声称找到全局最优参数。没有单独的网页或额外报告。

Spark ALS 文档：https://spark.apache.org/docs/3.5.7/ml-collaborative-filtering.html

代码位于 source 目录，训练后模型位于 results/als_model.tar.gz，可直接下载。数据集请使用官方链接。

工具说明：代码实现与汇报制作使用 AI 辅助，实验结果来自实际运行。
