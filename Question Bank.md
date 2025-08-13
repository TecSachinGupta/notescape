# Comprehensive Spark & Data Engineering Interview Guide

## 🔥 Spark Architecture & Concepts

### ❓ Question 1. What is Spark Architecture?
#### Answer:
Apache Spark follows a master-slave architecture with the following components:
- **Driver Program**: Contains the main() function and creates SparkContext
- **Cluster Manager**: Allocates resources across applications (YARN, Mesos, Standalone)
- **Executors**: Run tasks and store data in memory/disk
- **SparkContext**: Entry point that coordinates with cluster manager
- **DAG Scheduler**: Creates DAG of stages
- **Task Scheduler**: Submits tasks to executors

### ❓ Question 2. Explain Spark Context vs Spark Session
#### Answer:
**SparkContext**:
- Entry point for Spark Core functionality
- Creates RDDs and accumulators
- One per JVM, cannot create multiple in same application

**SparkSession**:
- Unified entry point introduced in Spark 2.0
- Combines SparkContext, SQLContext, and HiveContext
- Entry point for DataFrame and Dataset APIs
- Can create multiple sessions in same application

### ❓ Question 3. What is DAG in Spark and why is it needed?
#### Answer:
**DAG (Directed Acyclic Graph)**:
- Represents the sequence of computations performed on data
- Nodes represent RDDs/DataFrames, edges represent transformations
- No cycles, ensuring fault tolerance through lineage

**Why needed**:
- Enables lazy evaluation and optimization
- Fault tolerance through lineage reconstruction
- Stage division for optimal execution
- Catalyst optimizer uses DAG for query planning

### ❓ Question 4. Explain lazy evaluation and how it improves performance
#### Answer:
**Lazy Evaluation**: Transformations are not executed immediately but only when an action is called.

**Performance Benefits**:
- **Optimization**: Multiple transformations are combined into optimized execution plans
- **Predicate Pushdown**: Filters are pushed down to data sources
- **Projection Pushdown**: Only required columns are read
- **Memory Management**: Avoids creating intermediate datasets unnecessarily
- **Fault Tolerance**: Lineage information is maintained for recovery

### ❓ Question 5. Difference between logical and physical plan
#### Answer:
**Logical Plan**:
- High-level description of what needs to be computed
- Created from SQL queries or DataFrame operations
- Independent of execution strategy
- Optimized by Catalyst optimizer

**Physical Plan**:
- How the computation will actually be executed
- Specifies algorithms, data formats, and physical operators
- Multiple physical plans possible for one logical plan
- Cost-based optimizer selects best physical plan

## 🔄 Transformations & Actions

### ❓ Question 6. Difference between transformations and actions in Spark
#### Answer:
**Transformations**:
- Lazy operations that create new RDD/DataFrame from existing ones
- Return new RDD/DataFrame
- Examples: map, filter, groupBy, join
- Two types: Narrow (no shuffling) and Wide (requires shuffling)

**Actions**:
- Eager operations that trigger computation
- Return values to driver or write data to external systems
- Examples: collect, count, save, show, take

### ❓ Question 7. Difference between narrow and wide transformations
#### Answer:
**Narrow Transformations**:
- Each input partition contributes to only one output partition
- No shuffling required
- Examples: map, filter, flatMap
- Fast and memory efficient

**Wide Transformations**:
- Each input partition contributes to multiple output partitions
- Requires shuffling across network
- Examples: groupBy, join, distinct, repartition
- Slower due to network I/O

### ❓ Question 8. Is repartition a wide or narrow transformation?
#### Answer:
**Repartition is a wide transformation** because:
- It redistributes data across different partitions
- Requires shuffling data across the network
- Each input partition can contribute to multiple output partitions
- Creates stage boundaries in DAG

## 💾 Memory Management & Optimization

### ❓ Question 9. Explain memory management in Spark
#### Answer:
Spark memory is divided into:
- **Storage Memory (60%)**: For caching RDDs/DataFrames
- **Execution Memory (20%)**: For shuffles, joins, sorts, aggregations  
- **Reserved Memory (20%)**: For system operations

**Unified Memory Management**: Storage and execution memory can borrow from each other when needed.

### ❓ Question 10. Difference between cache and persist
#### Answer:
**Cache**:
- Uses default storage level (MEMORY_ONLY)
- Shortcut for persist()
- Data stored in memory only

**Persist**:
- Allows specifying storage level
- Options: MEMORY_ONLY, MEMORY_AND_DISK, DISK_ONLY, etc.
- More flexible caching strategy
- Can specify serialization and replication

### ❓ Question 11. Explain OOM errors and how to handle them
#### Answer:
**Out Of Memory Errors occur due to**:
- Data skewness
- Insufficient executor memory
- Large broadcast variables
- Too many partitions causing small file problem

**Solutions**:
- Increase executor memory (`--executor-memory`)
- Optimize data skewness with salting
- Use appropriate storage levels
- Repartition data evenly
- Use broadcast joins for small tables
- Tune `spark.sql.adaptive.enabled`

### ❓ Question 12. How to decide number of executors and cores?
#### Answer:
**Formula approach**:
```
Total cores available = Nodes × Cores per node
Cores per executor = 5 (optimal for HDFS throughput)
Executors per node = (Cores per node - 1) / Cores per executor
Total executors = Executors per node × Number of nodes
```

**Considerations**:
- Leave 1 core per node for OS and Hadoop daemons
- Memory per executor = (Total memory - 1GB) / Executors per node
- Monitor resource utilization and adjust accordingly

## 🔗 Joins & Partitioning

### ❓ Question 13. Types of joins in Spark
#### Answer:
**Physical Join Strategies**:
- **Broadcast Hash Join**: Small table broadcast to all nodes
- **Sort Merge Join**: Both tables sorted and merged
- **Shuffle Hash Join**: Shuffle both tables on join keys

**Logical Join Types**:
- Inner, Left Outer, Right Outer, Full Outer
- Left Semi, Left Anti
- Cross Join

### ❓ Question 14. What is broadcast join and when to use it?
#### Answer:
**Broadcast Join**: Small table is copied to all executor nodes to avoid shuffling the large table.

**When to use**:
- When one table is small (< 10MB by default)
- To avoid expensive shuffle operations
- For dimension table joins in star schema

**Configuration**:
```python
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10MB")
```

### ❓ Question 15. Difference between bucketing and partitioning
#### Answer:
**Partitioning**:
- Divides data into directories based on column values
- Enables partition pruning
- Good for range queries on partition column

**Bucketing**:
- Divides data into fixed number of buckets using hash function
- Co-locates data with same hash values
- Improves join performance when joining on bucketed columns
- Reduces shuffling in joins and aggregations

## 🗃️ Data Formats & Storage

### ❓ Question 16. Star schema vs Snowflake schema
#### Answer:
**Star Schema**:
- Fact table surrounded by denormalized dimension tables
- Simple joins, faster query performance
- More storage space due to denormalization

**Snowflake Schema**:
- Dimension tables are normalized into multiple tables
- Complex joins, slower query performance  
- Less storage space, eliminates data redundancy

### ❓ Question 17. What are fact and dimension tables?
#### Answer:
**Fact Tables**:
- Contain quantitative data (measures/metrics)
- Large tables with foreign keys to dimensions
- Examples: Sales amount, quantity, revenue

**Dimension Tables**:
- Contain descriptive attributes
- Smaller tables with primary keys
- Examples: Customer, Product, Time dimensions

## 🐍 Python Concepts

### ❓ Question 18. Difference between list and tuple in Python
#### Answer:
**List**:
- Mutable (can be changed after creation)
- Uses square brackets []
- More memory overhead
- Slower for large datasets

**Tuple**:
- Immutable (cannot be changed after creation)
- Uses parentheses ()
- Less memory overhead
- Faster for large datasets
- Can be used as dictionary keys

### ❓ Question 19. Explain Python decorators with example
#### Answer:
**Decorators**: Functions that modify the behavior of other functions without changing their code.

```python
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time}")
        return result
    return wrapper

@timer_decorator
def slow_function():
    time.sleep(2)
    return "Done"
```

### ❓ Question 20. What are Python generators?
#### Answer:
**Generators**: Functions that return iterators and yield values one at a time instead of returning all at once.

```python
def fibonacci_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Usage
for num in fibonacci_generator(10):
    print(num)
```

**Benefits**: Memory efficient, lazy evaluation, can represent infinite sequences.

## 📊 SQL & Data Analysis

### ❓ Question 21. Difference between RANK() and DENSE_RANK()
#### Answer:
**RANK()**:
- Leaves gaps in ranking when there are ties
- Example: 1, 2, 2, 4, 5

**DENSE_RANK()**:
- No gaps in ranking
- Example: 1, 2, 2, 3, 4

```sql
SELECT name, salary,
       RANK() OVER (ORDER BY salary DESC) as rank,
       DENSE_RANK() OVER (ORDER BY salary DESC) as dense_rank
FROM employees;
```

### ❓ Question 22. Difference between GROUP BY and window functions
#### Answer:
**GROUP BY**:
- Aggregates data and reduces number of rows
- Returns one row per group
- Cannot access individual row details

**Window Functions**:
- Performs calculations across related rows
- Maintains original number of rows
- Can access both aggregated and individual row data
- Uses OVER() clause

## 🏗️ Architecture & Design

### ❓ Question 23. Design end-to-end data pipeline
#### Answer:
**Components**:
1. **Data Ingestion**: Apache Kafka, AWS Kinesis, batch files
2. **Data Processing**: Apache Spark, Apache Flink
3. **Data Storage**: HDFS, S3, Delta Lake
4. **Data Warehouse**: Snowflake, Redshift, BigQuery
5. **Orchestration**: Apache Airflow, AWS Step Functions
6. **Monitoring**: Prometheus, Grafana, DataDog
7. **Data Quality**: Great Expectations, Apache Griffin

**Pipeline Flow**:
Raw Data → Ingestion → Processing → Storage → Analytics → Visualization

### ❓ Question 24. Explain ETL pipeline in AWS
#### Answer:
**AWS ETL Pipeline**:
1. **Extract**: AWS Glue Crawlers, Lambda, Kinesis
2. **Transform**: AWS Glue, EMR, Lambda
3. **Load**: S3, Redshift, RDS, DynamoDB

**Services**:
- **AWS Glue**: Serverless ETL service
- **AWS EMR**: Managed Hadoop/Spark clusters
- **AWS Step Functions**: Workflow orchestration
- **AWS Lambda**: Serverless computing
- **Amazon S3**: Data lake storage

## ⚡ Performance Optimization

### ❓ Question 25. Spark optimization techniques
#### Answer:
**Data-level optimizations**:
- Use appropriate file formats (Parquet, Delta)
- Partition data properly
- Use bucketing for frequent joins
- Enable predicate pushdown

**Code-level optimizations**:
- Cache frequently used DataFrames
- Use broadcast joins for small tables
- Avoid UDFs when possible
- Use vectorized operations

**Configuration optimizations**:
- Tune executor memory and cores
- Enable adaptive query execution
- Configure shuffle partitions
- Use appropriate storage levels

### ❓ Question 26. How to handle data skewness?
#### Answer:
**Identification**:
- Monitor task execution times
- Check partition sizes
- Use Spark UI to identify bottlenecks

**Solutions**:
- **Salting**: Add random prefix to skewed keys
- **Bucketing**: Pre-partition data by skewed columns
- **Broadcast joins**: For small-to-large table joins
- **Custom partitioning**: Implement custom partitioner
- **Repartitioning**: Redistribute data evenly

## 🔍 Troubleshooting & Debugging

### ❓ Question 27. How to debug long-running Spark jobs?
#### Answer:
**Tools and techniques**:
- **Spark UI**: Analyze stages, tasks, and storage
- **Spark History Server**: Review completed applications
- **Application logs**: Check driver and executor logs
- **Ganglia/Prometheus**: Monitor cluster resources

**Common issues**:
- Data skewness → Use salting or different join strategies  
- Memory issues → Increase executor memory or optimize code
- Shuffle operations → Minimize wide transformations
- Small files → Use coalesce or repartition

### ❓ Question 28. How to find errors in PySpark logs?
#### Answer:
**Log locations**:
- Driver logs: Check application driver output
- Executor logs: YARN logs or Spark standalone logs
- Application logs: Spark History Server

**Key error patterns**:
```bash
# OutOfMemoryError
grep -i "OutOfMemoryError" spark-logs/*

# Task failures
grep -i "Task failed" spark-logs/*

# Serialization issues
grep -i "NotSerializableException" spark-logs/*
```

## 📋 Practical Coding Questions

### ❓ Question 29. Find 3rd highest salary without window functions (PySpark)
#### Answer:
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import desc

# Create sample data
data = [(101, 15000), (102, 17000), (103, 10000), (104, 13000)]
df = spark.createDataFrame(data, ["emp_id", "salary"])

# Method 1: Using orderBy and limit
third_highest = df.orderBy(desc("salary")).limit(3).collect()[2]

# Method 2: Using distinct and collect
salaries = df.select("salary").distinct().orderBy(desc("salary")).collect()
third_salary = salaries[2][0]
result = df.filter(df.salary == third_salary)
```

### ❓ Question 30. Get monthly average using DataFrame operations
#### Answer:
```python
from pyspark.sql import functions as F

# Sample data with date column
df = spark.createDataFrame([
    ("2023-01-15", 1000),
    ("2023-01-20", 1500), 
    ("2023-02-10", 2000)
], ["date", "balance"])

# Extract year-month and calculate average
monthly_avg = df.withColumn("year_month", F.date_format("date", "yyyy-MM")) \
                .groupBy("year_month") \
                .agg(F.avg("balance").alias("avg_balance"))
                
monthly_avg.show()
```

### ❓ Question 31. Reverse string operations in Python
#### Answer:
```python
s = "i like this program very much"

# i) Reverse characters and words
def reverse_all(text):
    return text[::-1]

# ii) Reverse only characters, keep word positions
def reverse_chars_only(text):
    words = text.split()
    reversed_words = [word[::-1] for word in words]
    return ' '.join(reversed_words)

print(reverse_all(s))  # "hcum yrev margorp siht ekil i"
print(reverse_chars_only(s))  # "i ekil siht margorp yrev hcum"
```

### ❓ Question 32. Find non-matching parts of strings
#### Answer:
```python
def find_non_matching(a, b):
    words_a = set(a.split())
    words_b = set(b.split())
    
    # Find words in b but not in a
    non_matching = words_b - words_a
    return list(non_matching)

a = "i stay in ABC and love India"
b = "i stay in XYZ"
result = find_non_matching(a, b)
print(result)  # ['XYZ']
```

### ❓ Question 33. Replace None values in list
#### Answer:
```python
def replace_none_with_previous(lst):
    result = []
    last_valid = None
    
    for item in lst:
        if item is not None:
            last_valid = item
            result.append(item)
        else:
            result.append(last_valid)
    
    return result

l2 = [1, None, 2, 3, None, None, 4, 5, None]
output = replace_none_with_previous(l2)
print(output)  # [1, 1, 2, 3, 3, 3, 4, 5, 5]
```

## 🔧 Advanced Topics

### ❓ Question 34. What are UDFs and when to avoid them?
#### Answer:
**UDF (User Defined Functions)**: Custom functions to extend Spark's built-in functions.

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

def categorize_salary(salary):
    if salary > 50000:
        return "High"
    elif salary > 30000:
        return "Medium"
    else:
        return "Low"

categorize_udf = udf(categorize_salary, StringType())
df.withColumn("salary_category", categorize_udf("salary"))
```

**When to avoid**:
- Performance overhead due to serialization
- Cannot utilize Catalyst optimizer
- Use built-in functions when possible
- Consider Pandas UDFs for better performance

### ❓ Question 35. Explain Delta Lake features
#### Answer:
**Delta Lake**: Open-source storage layer that brings ACID transactions to Apache Spark.

**Key Features**:
- **ACID Transactions**: Ensures data consistency
- **Schema Evolution**: Handle schema changes gracefully  
- **Time Travel**: Query historical versions of data
- **Upserts**: Efficient merge operations
- **Data Quality**: Schema validation and constraints

```python
# Write to Delta table
df.write.format("delta").save("/path/to/delta-table")

# Time travel
df_yesterday = spark.read.format("delta") \
    .option("versionAsOf", 1) \
    .load("/path/to/delta-table")
```

### ❓ Question 36. Explain different types of SCD (Slowly Changing Dimensions)
#### Answer:
**SCD Type 1**: Overwrite old data with new data
- Simple but loses historical data
- Used when history is not important

**SCD Type 2**: Maintain historical data with versioning
- Add new records for changes
- Use effective dates or version numbers
- Most commonly used

**SCD Type 3**: Limited history tracking
- Add columns for previous values
- Limited to specific number of historical values

## 🐳 DevOps & Deployment

### ❓ Question 37. Explain CI/CD pipeline for Spark applications
#### Answer:
**Pipeline Stages**:
1. **Source Control**: Git with feature branches
2. **Build**: Package Spark application (JAR/Python wheel)
3. **Test**: Unit tests, integration tests, data quality tests
4. **Deploy**: Deploy to staging/production clusters
5. **Monitor**: Application performance and data quality

**Tools**:
- **Jenkins/GitLab CI**: Build automation
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **Terraform**: Infrastructure as code
- **Airflow**: Workflow orchestration

### ❓ Question 38. Cron job scheduling patterns
#### Answer:
**Cron Format**: `minute hour day month day-of-week`

**Common patterns**:
```bash
# Every day at 2 AM
0 2 * * *

# Every Monday at 9 AM  
0 9 * * 1

# Every 15 minutes
*/15 * * * *

# Last day of month
0 2 L * *

# Weekdays only
0 9 * * 1-5
```

## 🧮 Mathematical & Algorithm Questions

### ❓ Question 39. Time and Space Complexity concepts
#### Answer:
**Time Complexity**: Measure of algorithm execution time growth
- O(1): Constant time
- O(log n): Logarithmic time  
- O(n): Linear time
- O(n²): Quadratic time

**Space Complexity**: Measure of memory usage growth
- Includes auxiliary space used by algorithm
- Important for memory-constrained environments

### ❓ Question 40. Mathematical puzzle - Make 120 from five zeros
#### Answer:
**Solution**: (0! + 0! + 0! + 0! + 0!)! = 5! = 120

**Explanation**: 
- 0! = 1 (factorial of zero is one)
- (1+1+1+1+1)! = 5! = 120
