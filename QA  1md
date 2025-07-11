# 🧠 Frequently Asked Questions (FAQ)

### ❓ Question 1: What is the difference between Spark Context and Spark Session?
#### Answer:

**Spark Context:**
- Entry point for Spark Core functionality
- Creates RDDs, accumulators, and broadcast variables
- Coordinates with cluster manager
- One SparkContext per JVM
- Lower-level API

**Spark Session:**
- Unified entry point for Spark 2.0+
- Encapsulates SparkContext, SQLContext, and HiveContext
- Provides DataFrame/Dataset API
- Supports SQL queries
- Higher-level abstraction

**Key Differences:**
- SparkSession is the modern, recommended approach
- SparkContext is more low-level and primarily for RDD operations
- SparkSession provides better integration with structured APIs

**Code Example:**
```scala
// Spark Context (older approach)
val conf = new SparkConf().setAppName("MyApp")
val sc = new SparkContext(conf)

// Spark Session (modern approach)
val spark = SparkSession.builder()
  .appName("MyApp")
  .getOrCreate()
```

---

### ❓ Question 2: What are the key features of Scala and Python?
#### Answer:

**Scala Features:**
- Functional and Object-Oriented programming
- Static typing with type inference
- Pattern matching
- Immutability by default
- Actor model (Akka)
- Traits for multiple inheritance
- Case classes
- Higher-order functions
- Lazy evaluation
- Compiles to JVM bytecode

**Python Features:**
- Dynamic typing
- Interpreted language
- Simple and readable syntax
- Extensive standard library
- Duck typing
- List comprehensions
- Generators and iterators
- Decorators
- Context managers
- Multiple inheritance
- Interactive shell (REPL)

**For Spark:**
- Scala: Better performance, native integration
- Python: Easier to learn, rich ecosystem for data science

---

### ❓ Question 3: Explain Scala Traits and Closures with examples.
#### Answer:

**Traits:**
- Similar to interfaces but can contain concrete methods
- Support multiple inheritance
- Can have fields and constructors
- Used for mixins

```scala
trait Printable {
  def print(): Unit = println("Printing...")
}

trait Drawable {
  def draw(): Unit
}

class Document extends Printable with Drawable {
  def draw(): Unit = println("Drawing document")
}
```

**Closures:**
- Functions that capture variables from their enclosing scope
- Variables are "closed over" by the function
- Can access and modify variables from outer scope

```scala
def multiplier(factor: Int) = (x: Int) => x * factor

val double = multiplier(2)  // Closure captures 'factor'
val triple = multiplier(3)

println(double(5))  // Output: 10
println(triple(5))  // Output: 15
```

-----
### ❓ Question 4: Explain Python Decorators and Generators with examples.
#### Answer:

**Decorators:**
- Functions that modify or extend other functions
- Use `@` syntax
- Common for logging, timing, authentication

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Finished {func.__name__}")
        return result
    return wrapper

@my_decorator
def greet(name):
    return f"Hello, {name}!"

# Usage
greet("Alice")  # Prints decorator messages + greeting
```

**Generators:**
- Functions that yield values one at a time
- Memory efficient for large datasets
- Use `yield` keyword
- Return generator objects

```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Usage
for num in fibonacci(10):
    print(num)

# Generator expression
squares = (x**2 for x in range(10))
```

---

### ❓ Question 5: Explain the end-to-end CI/CD pipeline in your project.
#### Answer:**

**CI/CD Pipeline Components:**

1. **Source Control:** Git repository (GitHub/GitLab)
2. **Build Trigger:** Webhook on code push/PR
3. **Build Stage:** 
   - Code compilation
   - Unit tests
   - Code quality checks (SonarQube)
   - Security scanning
4. **Package Stage:**
   - Docker image creation
   - Artifact repository (Nexus/Artifactory)
5. **Deploy Stage:**
   - DEV → QA → STAGING → PRODUCTION
   - Infrastructure as Code (Terraform)
   - Configuration management
6. **Testing:**
   - Integration tests
   - Performance tests
   - Smoke tests
7. **Monitoring:**
   - Application monitoring
   - Log aggregation
   - Alerting

**Tools Used:**
- Jenkins/GitLab CI/GitHub Actions
- Docker/Kubernetes
- AWS/Azure cloud services
- Terraform for infrastructure
- Monitoring: Prometheus/Grafana

---

### ❓ Question 6: Explain Spark Architecture in detail.
#### Answer:

**Spark Architecture Components:**

1. **Driver Program:**
   - Contains main() function
   - Creates SparkContext
   - Converts user program into tasks
   - Schedules tasks on executors

2. **Cluster Manager:**
   - Allocates resources across applications
   - Types: Standalone, YARN, Mesos, Kubernetes

3. **Worker Nodes:**
   - Run executor processes
   - Execute tasks and store data

4. **Executors:**
   - Run on worker nodes
   - Execute tasks assigned by driver
   - Store data in memory/disk cache

**Execution Flow:**
1. User submits application to driver
2. Driver requests resources from cluster manager
3. Cluster manager allocates executors
4. Driver sends tasks to executors
5. Executors run tasks and return results
6. Driver collects results and presents to user

**Key Concepts:**
- DAG (Directed Acyclic Graph) execution
- Lazy evaluation
- In-memory computing
- Fault tolerance through lineage

---

### ❓ Question 7: How do you deploy Spark applications in AWS?
#### Answer:

**AWS Deployment Options:**

1. **Amazon EMR (Elastic MapReduce):**
   - Managed Hadoop/Spark clusters
   - Auto-scaling capabilities
   - Integration with S3, DynamoDB
   - Cost-effective for batch processing

2. **AWS Glue:**
   - Serverless ETL service
   - Managed Spark environment
   - Pay-per-use pricing
   - Integration with Data Catalog

3. **EKS (Elastic Kubernetes Service):**
   - Kubernetes-based deployment
   - Container orchestration
   - Better for microservices architecture

4. **EC2 with Standalone Mode:**
   - Manual cluster setup
   - Full control over configuration
   - Custom networking and security

**Best Practices:**
- Use S3 for data storage
- Implement proper IAM roles
- Configure VPC for security
- Use spot instances for cost optimization
- Enable CloudWatch monitoring
- Implement data partitioning strategies

---

### ❓ Question 8: Explain Time and Space Complexity with examples.
#### Answer:

**Time Complexity:**
- Measures algorithm's execution time relative to input size
- Common complexities: O(1), O(log n), O(n), O(n log n), O(n²)

**Space Complexity:**
- Measures memory usage relative to input size
- Includes auxiliary space used by algorithm

**Examples:**

```python
# O(1) - Constant time
def get_first_element(arr):
    return arr[0] if arr else None

# O(n) - Linear time
def find_max(arr):
    max_val = arr[0]
    for num in arr[1:]:
        if num > max_val:
            max_val = num
    return max_val

# O(n²) - Quadratic time
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

# O(log n) - Logarithmic time
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

**Space Complexity Examples:**
- O(1): Using fixed variables
- O(n): Creating array of size n
- O(n²): Creating 2D matrix

---

### ❓ Question 9: Write code to find the top 3 highest-paid employees in each department.
#### Answer:

**SQL Solution:**
```sql
WITH ranked_employees AS (
    SELECT 
        employee_id,
        employee_name,
        department,
        salary,
        ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rn
    FROM employees
)
SELECT 
    employee_id,
    employee_name,
    department,
    salary
FROM ranked_employees
WHERE rn <= 3
ORDER BY department, salary DESC;
```

**PySpark Solution:**
```python
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, desc

spark = SparkSession.builder.appName("TopEmployees").getOrCreate()

# Assuming we have a DataFrame 'employees'
# employees = spark.read.table("employees")

# Define window specification
window_spec = Window.partitionBy("department").orderBy(desc("salary"))

# Apply window function
result = employees.withColumn("rn", row_number().over(window_spec)) \
                 .filter("rn <= 3") \
                 .select("employee_id", "employee_name", "department", "salary") \
                 .orderBy("department", desc("salary"))

result.show()
```

**Alternative PySpark with DataFrame API:**
```python
# Using DataFrame operations
result = employees.groupBy("department") \
                 .apply(lambda x: x.orderBy(desc("salary")).limit(3)) \
                 .select("employee_id", "employee_name", "department", "salary")
```

---

### ❓ Question 10: Explain memory management in Spark.
#### Answer:

**Spark Memory Model:**

1. **Execution Memory:**
   - Used for computations (joins, aggregations, sorting)
   - Temporary storage during task execution
   - Can spill to disk if insufficient

2. **Storage Memory:**
   - Used for caching RDDs/DataFrames
   - Broadcast variables
   - Can be evicted using LRU policy

3. **User Memory:**
   - User data structures
   - UDFs and user code
   - Spark metadata

**Memory Allocation:**
- **Total Memory = Executor Memory - Reserved Memory**
- **Unified Memory = Total Memory × 0.6** (configurable)
- **Execution Memory = Unified Memory × 0.5** (initially)
- **Storage Memory = Unified Memory × 0.5** (initially)

**Key Configurations:**
```scala
spark.conf.set("spark.executor.memory", "4g")
spark.conf.set("spark.executor.memoryFraction", "0.6")
spark.conf.set("spark.storage.memoryFraction", "0.5")
spark.conf.set("spark.storage.storageFraction", "0.5")
```

**Memory Management Strategies:**
1. **Appropriate partitioning**
2. **Efficient serialization** (Kryo vs Java)
3. **Proper caching levels** (MEMORY_ONLY, MEMORY_AND_DISK, etc.)
4. **Avoiding data skew**
5. **Garbage collection tuning**
6. **Broadcasting small datasets**

**Memory Issues & Solutions:**
- **OOM Errors:** Increase executor memory, reduce partition size
- **Spilling:** Optimize joins, increase shuffle partitions
- **GC Pressure:** Tune GC settings, reduce object creation

---

### ❓ Question 11: What is the difference between Star Schema and Snowflake Schema?
#### Answer:

**Star Schema:**
- Central fact table surrounded by dimension tables
- Dimension tables are denormalized
- Simple structure with fewer joins
- Better query performance
- Higher storage requirements due to redundancy

**Snowflake Schema:**
- Dimension tables are normalized into multiple related tables
- More complex structure with multiple levels
- Reduces data redundancy
- Lower storage requirements
- More complex queries with multiple joins

**Comparison:**

| Aspect | Star Schema | Snowflake Schema |
|--------|-------------|------------------|
| Structure | Simple, flat | Complex, hierarchical |
| Joins | Fewer joins | More joins |
| Query Performance | Faster | Slower |
| Storage | Higher | Lower |
| Maintenance | Easier | More complex |
| Data Redundancy | Higher | Lower |

**Example:**
```sql
-- Star Schema
FACT_SALES (sale_id, product_id, customer_id, date_id, amount)
DIM_PRODUCT (product_id, product_name, category, brand, price)
DIM_CUSTOMER (customer_id, customer_name, city, state, country)

-- Snowflake Schema
FACT_SALES (sale_id, product_id, customer_id, date_id, amount)
DIM_PRODUCT (product_id, product_name, category_id, brand_id, price)
DIM_CATEGORY (category_id, category_name)
DIM_BRAND (brand_id, brand_name)
DIM_CUSTOMER (customer_id, customer_name, city_id)
DIM_CITY (city_id, city_name, state_id)
DIM_STATE (state_id, state_name, country_id)
DIM_COUNTRY (country_id, country_name)
```

---

### ❓ Question 12: Describe an ETL pipeline implementation in AWS.
#### Answer:

**AWS ETL Pipeline Components:**

1. **Data Ingestion:**
   - **AWS Kinesis:** Real-time streaming data
   - **AWS S3:** Batch file uploads
   - **AWS DMS:** Database migration/replication
   - **AWS Lambda:** Event-driven ingestion

2. **Data Processing:**
   - **AWS Glue:** Serverless ETL service
   - **Amazon EMR:** Managed Hadoop/Spark clusters
   - **AWS Batch:** Batch processing jobs
   - **AWS Step Functions:** Workflow orchestration

3. **Data Storage:**
   - **AWS S3:** Data lake storage
   - **Amazon Redshift:** Data warehouse
   - **Amazon RDS:** Relational databases
   - **Amazon DynamoDB:** NoSQL database

4. **Data Cataloging:**
   - **AWS Glue Data Catalog:** Metadata repository
   - **AWS Lake Formation:** Data lake governance

**Sample ETL Pipeline Architecture:**
```
Source Systems → Kinesis/S3 → Glue ETL → S3 Data Lake → Redshift → Analytics Tools
```

**AWS Glue ETL Job Example:**
```python
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

# Initialize Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

# Read data from S3
datasource = glueContext.create_dynamic_frame.from_catalog(
    database="my_database",
    table_name="raw_data"
)

# Transform data
transformed = ApplyMapping.apply(
    frame=datasource,
    mappings=[
        ("id", "string", "customer_id", "string"),
        ("name", "string", "customer_name", "string"),
        ("amount", "double", "purchase_amount", "double")
    ]
)

# Write to S3
glueContext.write_dynamic_frame.from_options(
    frame=transformed,
    connection_type="s3",
    connection_options={"path": "s3://my-bucket/processed/"},
    format="parquet"
)
```

**Best Practices:**
- Use partitioning for better performance
- Implement data quality checks
- Enable monitoring and logging
- Use IAM roles for security
- Implement retry mechanisms
- Cost optimization with spot instances

---

### ❓ Question 13: Given five zeroes (0 0 0 0 0), use any mathematical operators to make it equal to 120.
#### Answer: 

**Solutions:**

**Solution 1: Using Factorial**
```
(0! + 0! + 0! + 0! + 0!)! = 120
```
Explanation: 0! = 1, so (1+1+1+1+1)! = 5! = 120

**Solution 2: Using Powers and Factorials**
```
((0! + 0!)^(0! + 0! + 0!))! = 120
```
Explanation: (2^3)! = 8! = 40,320... wait, this is wrong.

**Correct Solution 2:**
```
(0! + 0! + 0! + 0! + 0!)! = 5! = 120
```

**Solution 3: Using Ceiling/Floor Functions**
```
⌈0.0! + 0.0! + 0.0! + 0.0! + 0.0!⌉! × (0! + 0! + 0! + 0!)!
```

**Most Common Answer:**
```
(0! + 0! + 0! + 0! + 0!)! = 5! = 120
```

**Step-by-step breakdown:**
1. 0! = 1 (by definition)
2. 0! + 0! + 0! + 0! + 0! = 1 + 1 + 1 + 1 + 1 = 5
3. 5! = 5 × 4 × 3 × 2 × 1 = 120

---

### ❓ Question 14: Explain the difference between Transformations and Actions in Spark.
#### Answer: 

**Transformations:**
- Lazy operations that create new RDDs
- Not executed immediately
- Build up a lineage graph (DAG)
- Return RDD/DataFrame/Dataset

**Actions:**
- Eager operations that trigger computation
- Execute immediately
- Return values to driver program
- Trigger the execution of transformation chain

**Transformation Examples:**
```scala
// Narrow Transformations (no shuffle)
val rdd1 = sc.parallelize(1 to 10)
val rdd2 = rdd1.map(_ * 2)          // map
val rdd3 = rdd1.filter(_ > 5)       // filter
val rdd4 = rdd1.flatMap(x => List(x, x * 2))  // flatMap

// Wide Transformations (require shuffle)
val rdd5 = rdd1.groupBy(_ % 2)      // groupBy
val rdd6 = rdd1.distinct()          // distinct
val rdd7 = rdd1.sortBy(x => x)      // sortBy
val rdd8 = rdd1.reduceByKey(_ + _)  // reduceByKey
```

**Action Examples:**
```scala
// Actions that return values
val count = rdd1.count()            // count
val firstElement = rdd1.first()     // first
val allElements = rdd1.collect()    // collect
val sum = rdd1.reduce(_ + _)        // reduce
val taken = rdd1.take(5)           // take

// Actions that save data
rdd1.saveAsTextFile("path")         // saveAsTextFile
rdd1.foreach(println)               // foreach
```

**DataFrame/Dataset Examples:**
```scala
// Transformations
val df1 = df.select("name", "age")
val df2 = df.filter($"age" > 18)
val df3 = df.groupBy("department").count()
val df4 = df.orderBy($"salary".desc)

// Actions
df.show()                           // show
df.count()                          // count
df.collect()                        // collect
df.write.parquet("path")           // write
```

**Key Differences:**
- **Execution:** Transformations are lazy, Actions are eager
- **Return Type:** Transformations return RDD/DataFrame, Actions return values
- **Performance:** Transformations are optimized together, Actions trigger execution
- **Caching:** Only makes sense with transformations

---

### ❓ Question 15: Explain common searching and sorting algorithms with their complexities.
#### Answer: 

**Searching Algorithms:**

**1. Linear Search:**
```python
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Time: O(n), Space: O(1)
```

**2. Binary Search:**
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Time: O(log n), Space: O(1)
# Requires sorted array
```

**Sorting Algorithms:**

**1. Bubble Sort:**
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

# Time: O(n²), Space: O(1)
```

**2. Selection Sort:**
```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

# Time: O(n²), Space: O(1)
```

**3. Insertion Sort:**
```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

# Time: O(n²), Space: O(1)
# Best case: O(n) for nearly sorted arrays
```

**4. Merge Sort:**
```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Time: O(n log n), Space: O(n)
```

**5. Quick Sort:**
```python
def quick_sort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi - 1)
        quick_sort(arr, pi + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Average: O(n log n), Worst: O(n²), Space: O(log n)
```

**Algorithm Complexity Summary:**

| Algorithm | Time (Best) | Time (Average) | Time (Worst) | Space |
|-----------|-------------|----------------|--------------|-------|
| Linear Search | O(1) | O(n) | O(n) | O(1) |
| Binary Search | O(1) | O(log n) | O(log n) | O(1) |
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) |

---

### ❓ Question 16: Explain the difference between DENSE_RANK and RANK functions.
#### Answer: 

**RANK Function:**
- Assigns ranks with gaps
- Same values get same rank
- Next rank skips numbers based on count of ties

**DENSE_RANK Function:**
- Assigns ranks without gaps
- Same values get same rank
- Next rank is always consecutive

**Example Data:**
```sql
-- Sample table: employees
| name    | salary |
|---------|--------|
| Alice   | 5000   |
| Bob     | 4000   |
| Charlie | 5000   |
| David   | 3000   |
| Eve     | 4000   |
```

**SQL Query:**
```sql
SELECT 
    name,
    salary,
    RANK() OVER (ORDER BY salary DESC) as rank_num,
    DENSE_RANK() OVER (ORDER BY salary DESC) as dense_rank_num,
    ROW_NUMBER() OVER (ORDER BY salary DESC) as row_num
FROM employees
ORDER BY salary DESC;
```

**Results:**
```
| name    | salary | rank_num | dense_rank_num | row_num |
|---------|--------|----------|----------------|---------|
| Alice   | 5000   | 1        | 1              | 1       |
| Charlie | 5000   | 1        | 1              | 2       |
| Bob     | 4000   | 3        | 2              | 3       |
| Eve     | 4000   | 3        | 2              | 4       |
| David   | 3000   | 5        | 3              | 5       |
```

**Key Differences:**
- **RANK:** 1, 1, 3, 3, 5 (gaps exist)
- **DENSE_RANK:** 1, 1, 2, 2, 3 (no gaps)
- **ROW_NUMBER:** 1, 2, 3, 4, 5 (unique numbers)

**PySpark Implementation:**
```python
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, dense_rank, row_number, desc

spark = SparkSession.builder.appName("RankDemo").getOrCreate()

# Create sample data
data = [("Alice", 5000), ("Bob", 4000), ("Charlie", 5000), 
        ("David", 3000), ("Eve", 4000)]
df = spark.createDataFrame(data, ["name", "salary"])

# Define window specification
window_spec = Window.orderBy(desc("salary"))

# Apply ranking functions
result = df.withColumn("rank", rank().over(window_spec)) \
          .withColumn("dense_rank", dense_rank().over(window_spec)) \
          .withColumn("row_number", row_number().over(window_spec))

result.show()
```

**Use Cases:**
- **RANK:** Tournament standings, academic rankings
- **DENSE_RANK:** Grade assignments, performance tiers
- **ROW_NUMBER:** Creating unique identifiers, pagination

---

### ❓ Question 17: Provide examples of common SQL queries and their solutions.
#### Answer: 

**1. Find Second Highest Salary:**
```sql
-- Method 1: Using LIMIT and OFFSET
SELECT salary 
FROM employees 
ORDER BY salary DESC 
LIMIT 1 OFFSET 1;

-- Method 2: Using Subquery
SELECT MAX(salary) 
FROM employees 
WHERE salary < (SELECT MAX(salary) FROM employees);

-- Method 3: Using Window Functions
SELECT DISTINCT salary
FROM (
    SELECT salary, 
           DENSE_RANK() OVER (ORDER BY salary DESC) as rank
    FROM employees
) ranked
WHERE rank = 2;
```

**2. Find Employees with Salary Above Department Average:**
```sql
SELECT e.name, e.salary, e.department
FROM employees e
WHERE e.salary > (
    SELECT AVG(salary) 
    FROM employees e2 
    WHERE e2.department = e.department
);
```

**3. Find Duplicate Records:**
```sql
-- Find duplicates
SELECT name, email, COUNT(*)
FROM users
GROUP BY name, email
HAVING COUNT(*) > 1;

-- Remove duplicates (keep one)
DELETE FROM users
WHERE id NOT IN (
    SELECT MIN(id)
    FROM users
    GROUP BY name, email
);
```

**4. Consecutive Numbers Problem:**
```sql
-- Find 3 consecutive numbers
SELECT DISTINCT l1.num
FROM logs l1
JOIN logs l2 ON l1.id = l2.id - 1 AND l1.num = l2.num
JOIN logs l3 ON l2.id = l3.id - 1 AND l2.num = l3.num;
```

**5. N-th Highest Salary Function:**
```sql
-- Generic function for N-th highest
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
    DECLARE M INT;
    SET M = N - 1;
    RETURN (
        SELECT DISTINCT salary
        FROM employees
        ORDER BY salary DESC
        LIMIT 1 OFFSET M
    );
END
```

**6. Running Total:**
```sql
SELECT 
    date,
    amount,
    SUM(amount) OVER (ORDER BY date) as running_total
FROM transactions
ORDER BY date;
```

**7. Find Gaps in Sequence:**
```sql
SELECT 
    id + 1 as gap_start,
    next_id - 1 as gap_end
FROM (
    SELECT id, 
           LEAD(id) OVER (ORDER BY id) as next_id
    FROM sequence_table
) t
WHERE next_id - id > 1;
```

---

### ❓ Question 18: Explain common DataFrame operations like groupby, first, max with examples.
#### Answer: 

**PySpark DataFrame Operations:**

**1. GroupBy Operations:**
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("DataFrameOps").getOrCreate()

# Sample data
data = [("Alice", "Sales", 5000), ("Bob", "IT", 6000), 
        ("Charlie", "Sales", 4500), ("David", "IT", 5500)]
df = spark.createDataFrame(data, ["name", "department", "salary"])

# Basic groupBy
dept_stats = df.groupBy("department").agg(
    avg("salary").alias("avg_salary"),
    max("salary").alias("max_salary"),
    min("salary").alias("min_salary"),
    count("*").alias("employee_count")
)

# Multiple groupBy columns
df.groupBy("department", "salary").count().show()
```

**2. First() Function:**
```python
# Get first row
first_row = df.first()
print(first_row)

# Get first value in each group
first_in_group = df.groupBy("department").agg(
    first("name").alias("first_employee"),
    first("salary").alias("first_salary")
)

# First with ordering
from pyspark.sql.window import Window

window_spec = Window.partitionBy("department").orderBy("salary")
df.withColumn("first_in_dept", 
              first("name").over(window_spec)).show()
```

**3. Max() Function:**
```python
# Global max
max_salary = df.agg(max("salary")).collect()[0][0]

# Max by group
max_by_dept = df.groupBy("department").agg(max("salary").alias("max_salary"))

# Row with max value
from pyspark.sql.window import Window
window_spec = Window.partitionBy("department").orderBy(desc("salary"))

df.withColumn("rank", rank().over(window_spec)) \
  .filter(col("rank") == 1) \
  .drop("rank") \
  .show()
```

**4. Advanced DataFrame Operations:**
```python
# Pivot operations
pivot_df = df.groupBy("department").pivot("name").sum("salary")

# Window functions
window_spec = Window.partitionBy("department").orderBy("salary")
df.withColumn("running_total", sum("salary").over(window_spec)) \
  .withColumn("lag_salary", lag("salary").over(window_spec)) \
  .withColumn("lead_salary", lead("salary").over(window_spec)) \
  .show()

# Multiple aggregations
multi_agg = df.groupBy("department").agg(
    count("*").alias("count"),
    sum("salary").alias("total_salary"),
    avg("salary").alias("avg_salary"),
    stddev("salary").alias("stddev_salary")
)
```

**5. Complex GroupBy with Custom Functions:**
```python
from pyspark.sql.types import *

# Custom aggregation function
def salary_range(salaries):
    return max(salaries) - min(salaries)

# Register UDF
salary_range_udf = udf(salary_range, IntegerType())

# Apply custom aggregation
df.groupBy("department").agg(
    collect_list("salary").alias("all_salaries")
).withColumn("salary_range", 
             salary_range_udf(col("all_salaries"))) \
 .drop("all_salaries") \
 .show()
```

**6. DataFrame Joins and Transformations:**
```python
# Join operations
dept_info = spark.createDataFrame([
    ("Sales", "Building A"), ("IT", "Building B")
], ["department", "location"])

joined_df = df.join(dept_info, "department", "inner")

# Column operations
df.select("name", "salary", 
          (col("salary") * 1.1).alias("new_salary")) \
  .withColumn("salary_category", 
              when(col("salary") > 5000, "High")
              .when(col("salary") > 4000, "Medium")
              .otherwise("Low")) \
  .show()
```

**Common Patterns:**
- **GroupBy + Agg:** Statistical operations per group
- **Window Functions:** Ranking, running totals, lag/lead
- **Pivot:** Transform rows to columns
- **Join:** Combine multiple DataFrames
- **Filter + Select:** Data subset operations

---

### ❓ Question 19: Explain logical and physical plans in Spark query execution.
#### Answer: 

**Spark Query Execution Process:**

1. **SQL/DataFrame API** → **Logical Plan** → **Physical Plan** → **Execution**

**Logical Plan:**
- Abstract representation of computation
- Describes WHAT needs to be done
- Independent of execution strategy
- Created by Catalyst optimizer

**Physical Plan:**
- Concrete execution strategy
- Describes HOW computation will be executed
- Includes specific operators and algorithms
- Considers available resources

**Example:**
```scala
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().appName("PlanExample").getOrCreate()

// Create sample DataFrames
val employees = spark.read.parquet("employees.parquet")
val departments = spark.read.parquet("departments.parquet")

// Query
val result = employees
  .filter($"salary" > 5000)
  .join(departments, "dept_id")
  .select($"name", $"salary", $"dept_name")
  .orderBy($"salary".desc)
```

**View Plans:**
```scala
// Logical plan
result.explain(true)

// Or more detailed
result.queryExecution.logical
result.queryExecution.optimizedPlan
result.queryExecution.executedPlan
```

**Logical Plan Stages:**

1. **Parsed Logical Plan:**
```
Sort [salary DESC]
+- Project [name, salary, dept_name]
   +- Join [dept_id]
      :- Filter (salary > 5000)
      :  +- Relation employees
      +- Relation departments
```

2. **Analyzed Logical Plan:**
- Resolves column references
- Checks data types
- Validates table existence

3. **Optimized Logical Plan:**
- Catalyst optimizer applies rules
- Predicate pushdown
- Projection pruning
- Constant folding

```
Sort [salary DESC]
+- Project [name, salary, dept_name]
   +- Join [dept_id]
      :- Project [name, salary, dept_id]
      :  +- Filter (salary > 5000)
      :     +- Relation employees
      +- Relation departments
```

**Physical Plan:**
```
*(3) Sort [salary DESC]
+- Exchange rangepartitioning(salary DESC)
   +- *(2) Project [name, salary, dept_name]
      +- *(2) BroadcastHashJoin [dept_id]
         :- *(2) Project [name, salary, dept_id]
         :  +- *(2) Filter (salary > 5000)
         :     +- *(2) FileScan parquet employees
         +- BroadcastExchange
            +- *(1) FileScan parquet departments
```

**Key Physical Operators:**
- **FileScan:** Read from storage
- **Filter:** Apply predicates
- **Project:** Select columns
- **BroadcastHashJoin:** Join with broadcast
- **Sort:** Sort data
- **Exchange:** Shuffle data between partitions

**Catalyst Optimizer Rules:**
```scala
// Common optimizations
1. Predicate Pushdown: Move filters closer to data source
2. Projection Pruning: Read only required columns
3. Constant Folding: Evaluate constants at compile time
4. Join Reordering: Optimize join order
5. Broadcast Join: Use broadcast for small tables
```

**Code Analysis:**
```scala
// Enable extended explain
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

// Analyze query execution
val df = spark.sql("""
    SELECT e.name, e.salary, d.dept_name
    FROM employees e
    JOIN departments d ON e.dept_id = d.dept_id
    WHERE e.salary > 5000
    ORDER BY e.salary DESC
""")

// Show all plans
df.explain("extended")
```

**Performance Implications:**
- **Logical Plan:** Determines optimization opportunities
- **Physical Plan:** Affects actual execution performance
- **Catalyst Optimizer:** Automatically improves query performance
- **Adaptive Query Execution:** Runtime optimizations based on statistics

---

### ❓ Question 20: Explain Entity-Relationship (E-R) models in SQL with examples.
#### Answer: 

**Entity-Relationship Model Components:**

**1. Entities:**
- Real-world objects or concepts
- Represented as tables in database
- Have attributes (columns)

**2. Attributes:**
- Properties of entities
- Columns in tables
- Types: Simple, Composite, Derived, Multi-valued

**3. Relationships:**
- Associations between entities
- Types: One-to-One, One-to-Many, Many-to-Many

**4. Keys:**
- **Primary Key:** Unique identifier
- **Foreign Key:** Reference to another table
- **Composite Key:** Multiple columns as key

**E-R Diagram Example: University System**

**Entities:**
- Student (StudentID, Name, Email, DOB)
- Course (CourseID, CourseName, Credits)
- Professor (ProfessorID, Name, Department)
- Enrollment (StudentID, CourseID, Grade, Semester)

**Relationships:**
- Student enrolls in Course (Many-to-Many)
- Professor teaches Course (One-to-Many)
- Student has Enrollment (One-to-Many)

**SQL Implementation:**
```sql
-- Entity Tables
CREATE TABLE Student (
    StudentID INT PRIMARY KEY,
    Name VARCHAR(100) NOT NULL,
    Email VARCHAR(100) UNIQUE,
    DOB DATE,
    Department VARCHAR(50)
);

CREATE TABLE Professor (
    ProfessorID INT PRIMARY KEY,
    Name VARCHAR(100) NOT NULL,
    Department VARCHAR(50),
    HireDate DATE
);

CREATE TABLE Course (
    CourseID INT PRIMARY KEY,
    CourseName VARCHAR(100) NOT NULL,
    Credits INT,
    ProfessorID INT,
    FOREIGN KEY (ProfessorID) REFERENCES Professor(ProfessorID)
);

-- Relationship Table (Many-to-Many)
CREATE TABLE Enrollment (
    StudentID INT,
    CourseID INT,
    Grade CHAR(2),
    Semester VARCHAR(20),
    PRIMARY KEY (StudentID, CourseID, Semester),
    FOREIGN KEY (StudentID) REFERENCES Student(StudentID),
    FOREIGN KEY (CourseID) REFERENCES Course(CourseID)
);
```

**Cardinality Examples:**

**1. One-to-One (1:1):**
```sql
-- Person has one Passport
CREATE TABLE Person (
    PersonID INT PRIMARY KEY,
    Name VARCHAR(100)
);

CREATE TABLE Passport (
    PassportID INT PRIMARY KEY,
    PassportNumber VARCHAR(20),
    PersonID INT UNIQUE,
    FOREIGN KEY (PersonID) REFERENCES Person(PersonID)
);
```

**2. One-to-Many (1:N):**
```sql
-- Department has many Employees
CREATE TABLE Department (
    DeptID INT PRIMARY KEY,
    DeptName VARCHAR(100)
);

CREATE TABLE Employee (
    EmpID INT PRIMARY KEY,
    Name VARCHAR(100),
    DeptID INT,
    FOREIGN KEY (DeptID) REFERENCES Department(DeptID)
);
```

**3. Many-to-Many (M:N):**
```sql
-- Authors write Books (resolved with junction table)
CREATE TABLE Author (
    AuthorID INT PRIMARY KEY,
    Name VARCHAR(100)
);

CREATE TABLE Book (
    BookID INT PRIMARY KEY,
    Title VARCHAR(200)
);

CREATE TABLE BookAuthor (
    AuthorID INT,
    BookID INT,
    PRIMARY KEY (AuthorID, BookID),
    FOREIGN KEY (AuthorID) REFERENCES Author(AuthorID),
    FOREIGN KEY (BookID) REFERENCES Book(BookID)
);
```

**Advanced E-R Concepts:**

**1. Weak Entity:**
```sql
-- Dependent entity that depends on strong entity
CREATE TABLE Order (
    OrderID INT PRIMARY KEY,
    CustomerID INT,
    OrderDate DATE
);

CREATE TABLE OrderItem (
    OrderID INT,
    ItemNumber INT,
    ProductName VARCHAR(100),
    Quantity INT,
    Price DECIMAL(10,2),
    PRIMARY KEY (OrderID, ItemNumber),
    FOREIGN KEY (OrderID) REFERENCES Order(OrderID)
);
```

**2. Inheritance/Specialization:**
```sql
-- Superclass
CREATE TABLE Vehicle (
    VehicleID INT PRIMARY KEY,
    Brand VARCHAR(50),
    Model VARCHAR(50),
    Year INT
);

-- Subclasses
CREATE TABLE Car (
    VehicleID INT PRIMARY KEY,
    NumDoors INT,
    FuelType VARCHAR(20),
    FOREIGN KEY (VehicleID) REFERENCES Vehicle(VehicleID)
);

CREATE TABLE Truck (
    VehicleID INT PRIMARY KEY,
    LoadCapacity DECIMAL(10,2),
    NumAxles INT,
    FOREIGN KEY (VehicleID) REFERENCES Vehicle(VehicleID)
);
```

**3. Aggregation/Composition:**
```sql
-- Aggregation: Department HAS Employees
CREATE TABLE Department (
    DeptID INT PRIMARY KEY,
    DeptName VARCHAR(100),
    Budget DECIMAL(15,2)
);

CREATE TABLE Employee (
    EmpID INT PRIMARY KEY,
    Name VARCHAR(100),
    DeptID INT,
    FOREIGN KEY (DeptID) REFERENCES Department(DeptID)
);

-- Composition: House HAS Rooms (rooms cannot exist without house)
CREATE TABLE House (
    HouseID INT PRIMARY KEY,
    Address VARCHAR(200),
    OwnerName VARCHAR(100)
);

CREATE TABLE Room (
    RoomID INT PRIMARY KEY,
    HouseID INT NOT NULL,
    RoomType VARCHAR(50),
    Area DECIMAL(8,2),
    FOREIGN KEY (HouseID) REFERENCES House(HouseID) ON DELETE CASCADE
);
```

**E-R Model Best Practices:**

1. **Normalization:**
   - 1NF: Atomic values, no repeating groups
   - 2NF: Remove partial dependencies
   - 3NF: Remove transitive dependencies

2. **Naming Conventions:**
   - Use descriptive names
   - Consistent naming patterns
   - Avoid reserved keywords

3. **Constraints:**
   - Primary keys for uniqueness
   - Foreign keys for referential integrity
   - Check constraints for data validation
   - Not null constraints where appropriate

**Complex E-R Example: E-commerce System**
```sql
-- Customer entity
CREATE TABLE Customer (
    CustomerID INT PRIMARY KEY,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Email VARCHAR(100) UNIQUE,
    Phone VARCHAR(20),
    RegistrationDate DATE
);

-- Product entity
CREATE TABLE Product (
    ProductID INT PRIMARY KEY,
    ProductName VARCHAR(100),
    Description TEXT,
    Price DECIMAL(10,2),
    StockQuantity INT,
    CategoryID INT
);

-- Order entity
CREATE TABLE CustomerOrder (
    OrderID INT PRIMARY KEY,
    CustomerID INT,
    OrderDate DATE,
    TotalAmount DECIMAL(10,2),
    Status VARCHAR(20),
    FOREIGN KEY (CustomerID) REFERENCES Customer(CustomerID)
);

-- Many-to-Many relationship: Order contains Products
CREATE TABLE OrderDetail (
    OrderID INT,
    ProductID INT,
    Quantity INT,
    UnitPrice DECIMAL(10,2),
    PRIMARY KEY (OrderID, ProductID),
    FOREIGN KEY (OrderID) REFERENCES CustomerOrder(OrderID),
    FOREIGN KEY (ProductID) REFERENCES Product(ProductID)
);

-- Address entity (One-to-Many with Customer)
CREATE TABLE Address (
    AddressID INT PRIMARY KEY,
    CustomerID INT,
    AddressType VARCHAR(20), -- 'Billing', 'Shipping'
    Street VARCHAR(100),
    City VARCHAR(50),
    State VARCHAR(50),
    ZipCode VARCHAR(10),
    Country VARCHAR(50),
    FOREIGN KEY (CustomerID) REFERENCES Customer(CustomerID)
);
```

**ER Model Queries:**
```sql
-- Find all customers who ordered in last 30 days
SELECT DISTINCT c.CustomerID, c.FirstName, c.LastName
FROM Customer c
JOIN CustomerOrder co ON c.CustomerID = co.CustomerID
WHERE co.OrderDate >= DATE_SUB(CURDATE(), INTERVAL 30 DAY);

-- Find products never ordered
SELECT p.ProductID, p.ProductName
FROM Product p
LEFT JOIN OrderDetail od ON p.ProductID = od.ProductID
WHERE od.ProductID IS NULL;

-- Get total sales by customer
SELECT c.CustomerID, c.FirstName, c.LastName, SUM(co.TotalAmount) as TotalSales
FROM Customer c
JOIN CustomerOrder co ON c.CustomerID = co.CustomerID
GROUP BY c.CustomerID, c.FirstName, c.LastName
ORDER BY TotalSales DESC;
```

**E-R Model Advantages:**
- Clear representation of business rules
- Facilitates database design
- Ensures data integrity
- Supports complex queries
- Enables efficient data retrieval

**Common E-R Mistakes to Avoid:**
- Many-to-Many relationships without junction tables
- Missing foreign key constraints
- Redundant data storage
- Inappropriate normalization level
- Weak entity identification errors

---

### ❓ Question 21. Groupby and Having
#### Answer:

**GROUP BY** is used to group rows that have the same values in specified columns into summary rows. **HAVING** is used to filter groups created by GROUP BY based on aggregate conditions.

**Key Differences:**
- WHERE filters rows before grouping
- HAVING filters groups after grouping
- HAVING can use aggregate functions, WHERE cannot

**Example:**
```sql
SELECT department, COUNT(*) as emp_count, AVG(salary) as avg_salary
FROM employees
WHERE salary > 30000
GROUP BY department
HAVING COUNT(*) > 5 AND AVG(salary) > 50000;
```

---

### ❓ Question 22. End to End Pipeline of the Project
#### Answer: 

**Typical Data Engineering Pipeline:**

1. **Data Ingestion**
   - Batch: HDFS, S3, databases
   - Streaming: Kafka, Kinesis, Pub/Sub

2. **Data Processing**
   - Raw data cleaning and validation
   - Transformations using Spark
   - Data quality checks

3. **Data Storage**
   - Data Lake: Parquet, Delta Lake
   - Data Warehouse: Hive, Redshift, BigQuery

4. **Data Serving**
   - APIs, dashboards, reports
   - ML model serving

5. **Monitoring & Orchestration**
   - Airflow, Prefect for scheduling
   - Logging, alerting, and monitoring

---

### ❓ Question 23. Optimization in Spark
#### Answer: 

**Performance Optimization Techniques:**

1. **Caching & Persistence**
   ```python
   df.cache()  # or df.persist(StorageLevel.MEMORY_AND_DISK)
   ```

2. **Partitioning**
   - Use appropriate partition columns
   - Avoid small files problem
   - Use repartition() or coalesce()

3. **Broadcast Variables**
   ```python
   broadcast_var = spark.sparkContext.broadcast(small_dict)
   ```

4. **Avoid Shuffling**
   - Use broadcast joins for small tables
   - Partition data appropriately
   - Use bucketing

5. **Serialization**
   - Use Kryo serialization
   - Avoid UDFs when possible

6. **Memory Management**
   - Tune executor memory
   - Use appropriate storage levels

---

### ❓ Question 24. Transformations & Actions
#### Answer: 

**Transformations (Lazy Evaluation):**
- Return new RDD/DataFrame
- Not executed until action is called
- Examples: map(), filter(), select(), groupBy(), join()

**Actions (Eager Evaluation):**
- Trigger computation and return results
- Examples: collect(), count(), show(), save(), take()

**Example:**
```python
# Transformations (lazy)
df_filtered = df.filter(df.age > 18)
df_selected = df_filtered.select("name", "age")

# Action (triggers execution)
df_selected.show()
```

---

### ❓ Question 25. Spark Architecture and DAG Formation
#### Answer: 

**Spark Architecture:**

1. **Driver Program**
   - Contains SparkContext
   - Coordinates execution
   - Maintains cluster state

2. **Cluster Manager**
   - YARN, Mesos, Kubernetes, Standalone
   - Resource allocation

3. **Executors**
   - Run tasks
   - Store data for caching
   - Report status to driver

**DAG Formation:**
- Spark builds Directed Acyclic Graph of operations
- Transformations create DAG nodes
- Actions trigger DAG execution
- Catalyst optimizer optimizes logical plan
- Code generation creates physical plan

---

### ❓ Question 26. How to Decide Number of Executors and Cores
#### Answer: 

**Guidelines:**

1. **Cores per Executor:**
   - Recommended: 4-5 cores per executor
   - Avoid > 5 cores (diminishing returns)

2. **Memory per Executor:**
   - Leave 1GB for OS overhead
   - Formula: (Total RAM - 1GB) / Number of executors

3. **Number of Executors:**
   - Formula: (Total cores / cores per executor) - 1 (for driver)

**Example Calculation:**
```
Cluster: 10 nodes, 16 cores each, 64GB RAM each
- Total cores: 160
- Cores per executor: 5
- Executors per node: 16/5 = 3
- Total executors: 10 * 3 = 30
- Executor memory: (64-1)/3 = ~21GB
```

---

### ❓ Question 27. Difference Between Window Functions and Group By
#### Answer: 

**GROUP BY:**
- Reduces number of rows
- Aggregates data into groups
- Cannot access individual row data after grouping

**Window Functions:**
- Maintains original row count
- Performs calculations across related rows
- Can access both aggregate and individual row data

**Example:**
```sql
-- GROUP BY (reduces rows)
SELECT department, AVG(salary) as avg_salary
FROM employees
GROUP BY department;

-- Window Function (maintains rows)
SELECT name, salary, department,
       AVG(salary) OVER (PARTITION BY department) as dept_avg
FROM employees;
```

---

### ❓ Question 28. Broadcast Variables
#### Answer: 

**Purpose:**
- Distribute read-only data to all executors
- Avoid sending large variables with each task
- Improve performance for joins with small tables

**Usage:**
```python
# Create broadcast variable
broadcast_dict = spark.sparkContext.broadcast(small_lookup_dict)

# Use in transformations
def lookup_function(value):
    return broadcast_dict.value.get(value, "Unknown")

df.withColumn("category", lookup_function(df.code))
```

**Benefits:**
- Reduces network traffic
- Faster joins with small datasets
- Cached on each executor

---

### ❓ Question 29. List and Tuple in Python
#### Answer: 

**List:**
- Mutable (can be modified)
- Ordered collection
- Syntax: `[1, 2, 3]`
- Methods: append(), remove(), pop(), etc.

**Tuple:**
- Immutable (cannot be modified)
- Ordered collection
- Syntax: `(1, 2, 3)`
- Methods: count(), index()

**Key Differences:**
```python
# List - mutable
my_list = [1, 2, 3]
my_list.append(4)  # Works
my_list[0] = 10    # Works

# Tuple - immutable
my_tuple = (1, 2, 3)
# my_tuple.append(4)  # Error!
# my_tuple[0] = 10    # Error!
```

**Use Cases:**
- List: When you need to modify data
- Tuple: When you need fixed data (coordinates, database records)

---

### ❓ Question 30. Get Monthly Average of Balance Amount
#### Answer: 

**Using DataFrame API:**
```python
from pyspark.sql import functions as F

# Sample data
df = spark.createDataFrame([
    ("2023-01-15", 1000.0),
    ("2023-01-20", 1500.0),
    ("2023-02-10", 2000.0),
    ("2023-02-25", 1800.0)
], ["date", "balance"])

# Convert date string to date type and extract year-month
result = df.withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd")) \
          .withColumn("year_month", F.date_format(F.col("date"), "yyyy-MM")) \
          .groupBy("year_month") \
          .agg(F.avg("balance").alias("monthly_avg_balance")) \
          .orderBy("year_month")

result.show()
```

**Using Spark SQL:**
```python
# Register DataFrame as temp view
df.createOrReplaceTempView("transactions")

# SQL query
monthly_avg = spark.sql("""
    SELECT 
        DATE_FORMAT(TO_DATE(date, 'yyyy-MM-dd'), 'yyyy-MM') as year_month,
        AVG(balance) as monthly_avg_balance
    FROM transactions
    GROUP BY DATE_FORMAT(TO_DATE(date, 'yyyy-MM-dd'), 'yyyy-MM')
    ORDER BY year_month
""")

monthly_avg.show()
```

**Alternative using substring:**
```python
# Using substring for year-month extraction
result = df.withColumn("year_month", F.substring("date", 1, 7)) \
          .groupBy("year_month") \
          .agg(F.avg("balance").alias("monthly_avg_balance")) \
          .orderBy("year_month")
```

**Expected Output:**
```
+----------+-------------------+
|year_month|monthly_avg_balance|
+----------+-------------------+
|   2023-01|             1250.0|
|   2023-02|             1900.0|
+----------+-------------------+
```
---

### ❓ Question 31. Explain Entity-Relationship (E-R) models in SQL with examples.
#### Answer: 
Entity-Relationship (E-R) models are conceptual database design tools that represent entities, their attributes, and relationships between entities. They form the foundation for relational database design.

**Key Components:**
- **Entities**: Objects or concepts (e.g., Customer, Order, Product)
- **Attributes**: Properties of entities (e.g., CustomerID, Name, Email)
- **Relationships**: Associations between entities (e.g., Customer places Order)

**Example E-R Model:**
```sql
-- Customer Entity
CREATE TABLE Customers (
    CustomerID INT PRIMARY KEY,
    Name VARCHAR(100),
    Email VARCHAR(100),
    Phone VARCHAR(15)
);

-- Order Entity
CREATE TABLE Orders (
    OrderID INT PRIMARY KEY,
    CustomerID INT,
    OrderDate DATE,
    TotalAmount DECIMAL(10,2),
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);

-- Product Entity
CREATE TABLE Products (
    ProductID INT PRIMARY KEY,
    ProductName VARCHAR(100),
    Price DECIMAL(10,2),
    Category VARCHAR(50)
);

-- Many-to-Many Relationship (Order-Product)
CREATE TABLE OrderDetails (
    OrderID INT,
    ProductID INT,
    Quantity INT,
    PRIMARY KEY (OrderID, ProductID),
    FOREIGN KEY (OrderID) REFERENCES Orders(OrderID),
    FOREIGN KEY (ProductID) REFERENCES Products(ProductID)
);
```

**Relationship Types:**
- **One-to-One**: Customer → Profile
- **One-to-Many**: Customer → Orders
- **Many-to-Many**: Orders → Products (via OrderDetails)

---

### ❓ Question 32. Explain HBase architecture and why reads are fast in HBase.
#### Answer: 
HBase is a distributed, column-oriented NoSQL database built on top of Hadoop HDFS.

**HBase Architecture:**
- **HMaster**: Manages region assignments, schema changes, and load balancing
- **RegionServer**: Handles read/write requests for specific regions
- **Region**: Horizontal partition of table data
- **Store**: Column family data within a region
- **MemStore**: In-memory write buffer
- **HFile**: Immutable storage files on HDFS
- **WAL (Write-Ahead Log)**: Ensures durability

**Why Reads are Fast:**
1. **Row Key Sorting**: Data is sorted by row key, enabling binary search
2. **Bloom Filters**: Probabilistic data structure that quickly determines if a row key might exist in an HFile
3. **Block Cache**: LRU cache that stores frequently accessed data blocks in memory
4. **Data Locality**: RegionServers are co-located with HDFS DataNodes
5. **Compression**: Reduces I/O overhead
6. **Skip Lists**: Efficient in-memory data structure for MemStore
7. **Single Row Atomicity**: No complex locking mechanisms needed

**Read Path:**
1. Check MemStore first
2. Check Block Cache
3. Use Bloom filters to identify relevant HFiles
4. Read from HFiles with binary search optimization

---

### ❓ Question 33. Explain lazy evaluation and how it improves performance.
#### Answer: 
Lazy evaluation is a programming strategy where expressions are not evaluated until their values are actually needed. It's a key concept in functional programming and distributed computing frameworks like Spark.

**How Lazy Evaluation Works:**
- Operations are recorded as a computation graph
- Execution is deferred until an action is triggered
- Multiple operations can be optimized together

**Performance Benefits:**
1. **Query Optimization**: Entire computation pipeline can be optimized as a whole
2. **Reduced Memory Usage**: Only necessary data is computed and stored
3. **Fault Tolerance**: Computation graph can be replayed for recovery
4. **Pipelining**: Operations can be combined and executed in parallel
5. **Dead Code Elimination**: Unused computations are never executed

**Spark Example:**
```python
# Transformations (lazy)
df = spark.read.csv("data.csv")
filtered_df = df.filter(df.age > 18)
grouped_df = filtered_df.groupBy("category").count()

# Action (triggers execution)
result = grouped_df.collect()  # Only now the computation happens
```

**Optimization Example:**
```python
# Without lazy evaluation: Multiple passes through data
data.filter(condition1).filter(condition2).map(transform)

# With lazy evaluation: Combined into single pass
# Spark optimizes this into: data.filter(condition1 AND condition2).map(transform)
```

---

### ❓ Question 34. Reasons and solutions for OOM (Out of Memory) errors
#### Answer: 
Out of Memory errors occur when applications exceed available memory resources.

**Common Reasons:**
1. **Large Dataset Processing**: Processing datasets larger than available memory
2. **Memory Leaks**: Objects not properly garbage collected
3. **Inefficient Algorithms**: O(n²) algorithms with large datasets
4. **Excessive Object Creation**: Creating too many temporary objects
5. **Improper Caching**: Caching too much data in memory
6. **Skewed Data**: Uneven data distribution causing hotspots

**Solutions:**

**1. Memory Management:**
```python
# Use generators instead of lists
def process_large_file():
    for line in open('large_file.txt'):  # Generator
        yield process_line(line)

# Instead of loading all at once
lines = list(open('large_file.txt'))  # Memory intensive
```

**2. Batch Processing:**
```python
# Process data in chunks
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
```

**3. Spark Solutions:**
```python
# Increase executor memory
spark.conf.set("spark.executor.memory", "8g")
spark.conf.set("spark.executor.memoryFraction", "0.8")

# Use efficient operations
df.filter(condition).select(columns)  # Instead of select().filter()

# Repartition skewed data
df.repartition(200, "partition_key")
```

**4. JVM Tuning:**
```bash
# Increase heap size
-Xmx8g -Xms2g

# Garbage collection tuning
-XX:+UseG1GC -XX:MaxGCPauseMillis=200
```

---

### ❓ Question 35. Design data pipeline for recommendation system
#### Answer: 
A recommendation system data pipeline requires handling user behavior data, item features, and model training/serving.

**Architecture Components:**

**1. Data Ingestion:**
```python
# Kafka for real-time user events
from kafka import KafkaProducer, KafkaConsumer

# User interaction events
user_events = {
    'user_id': 12345,
    'item_id': 67890,
    'event_type': 'click',
    'timestamp': '2024-01-01T10:00:00Z',
    'context': {'page': 'home', 'device': 'mobile'}
}
```

**2. Data Storage:**
```sql
-- User profiles
CREATE TABLE user_profiles (
    user_id BIGINT,
    demographics JSON,
    preferences JSON,
    last_updated TIMESTAMP
);

-- Item features
CREATE TABLE item_features (
    item_id BIGINT,
    category VARCHAR(50),
    features JSON,
    created_at TIMESTAMP
);

-- User interactions
CREATE TABLE user_interactions (
    user_id BIGINT,
    item_id BIGINT,
    interaction_type VARCHAR(20),
    rating FLOAT,
    timestamp TIMESTAMP
) PARTITIONED BY (DATE(timestamp));
```

**3. Feature Engineering Pipeline:**
```python
# Spark pipeline for feature engineering
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

def create_user_features(interactions_df):
    # Aggregate user behavior
    user_stats = interactions_df.groupBy("user_id").agg(
        count("*").alias("total_interactions"),
        avg("rating").alias("avg_rating"),
        countDistinct("item_id").alias("unique_items")
    )
    
    # Calculate user preferences
    user_preferences = interactions_df.groupBy("user_id", "category").agg(
        count("*").alias("category_interactions")
    ).groupBy("user_id").pivot("category").sum("category_interactions")
    
    return user_stats.join(user_preferences, "user_id")
```

**4. Model Training Pipeline:**
```python
# Collaborative Filtering with ALS
from pyspark.ml.recommendation import ALS

def train_recommendation_model(interactions_df):
    # Prepare data
    als = ALS(
        maxIter=10,
        regParam=0.01,
        userCol="user_id",
        itemCol="item_id",
        ratingCol="rating",
        coldStartStrategy="drop"
    )
    
    # Train model
    model = als.fit(interactions_df)
    return model

# Content-based features
def create_content_features(items_df):
    # TF-IDF for item descriptions
    from pyspark.ml.feature import Tokenizer, HashingTF, IDF
    
    tokenizer = Tokenizer(inputCol="description", outputCol="tokens")
    hashingTF = HashingTF(inputCol="tokens", outputCol="raw_features")
    idf = IDF(inputCol="raw_features", outputCol="features")
    
    # Pipeline execution
    tokens = tokenizer.transform(items_df)
    tf = hashingTF.transform(tokens)
    tfidf = idf.fit(tf).transform(tf)
    
    return tfidf
```

**5. Real-time Serving:**
```python
# Redis for fast lookups
import redis
import json

class RecommendationService:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379)
        
    def get_recommendations(self, user_id, num_recommendations=10):
        # Check cache first
        cached_recs = self.redis_client.get(f"user:{user_id}:recommendations")
        if cached_recs:
            return json.loads(cached_recs)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(user_id, num_recommendations)
        
        # Cache for 1 hour
        self.redis_client.setex(
            f"user:{user_id}:recommendations",
            3600,
            json.dumps(recommendations)
        )
        
        return recommendations
```

**6. Pipeline Orchestration:**
```python
# Airflow DAG
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

dag = DAG(
    'recommendation_pipeline',
    default_args={
        'start_date': datetime(2024, 1, 1),
        'retries': 1,
        'retry_delay': timedelta(minutes=5)
    },
    schedule_interval='@daily'
)

# Tasks
extract_interactions = PythonOperator(
    task_id='extract_interactions',
    python_callable=extract_user_interactions,
    dag=dag
)

feature_engineering = PythonOperator(
    task_id='feature_engineering',
    python_callable=create_features,
    dag=dag
)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=train_recommendation_model,
    dag=dag
)

# Dependencies
extract_interactions >> feature_engineering >> train_model
```

---

### ❓ Question 36. RDD, DataFrame, and Dataset differences
#### Answer: 
RDD, DataFrame, and Dataset are three main data abstractions in Apache Spark, each with distinct characteristics and use cases.

**RDD (Resilient Distributed Dataset):**
- **Definition**: Fundamental data structure in Spark, immutable distributed collection
- **API**: Functional programming style (map, filter, reduce)
- **Optimization**: No built-in optimization
- **Type Safety**: Compile-time type safety
- **Serialization**: Java/Kryo serialization

```python
# RDD Example
rdd = spark.sparkContext.textFile("data.txt")
filtered_rdd = rdd.filter(lambda line: "ERROR" in line)
word_count = filtered_rdd.flatMap(lambda line: line.split()).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
```

**DataFrame:**
- **Definition**: Distributed collection with named columns (like SQL table)
- **API**: SQL-like operations and domain-specific language
- **Optimization**: Catalyst optimizer for query optimization
- **Type Safety**: Runtime type checking
- **Serialization**: Efficient binary format (Tungsten)

```python
# DataFrame Example
df = spark.read.csv("data.csv", header=True)
filtered_df = df.filter(df.age > 18)
result = filtered_df.groupBy("department").avg("salary")
```

**Dataset:**
- **Definition**: Extension of DataFrame with compile-time type safety
- **API**: Combines RDD's type safety with DataFrame's optimizations
- **Optimization**: Catalyst optimizer + Tungsten execution
- **Type Safety**: Compile-time type checking
- **Language**: Primarily available in Scala/Java

```scala
// Dataset Example (Scala)
case class Person(name: String, age: Int, department: String)
val ds = spark.read.csv("data.csv").as[Person]
val filtered = ds.filter(_.age > 18)
val result = filtered.groupBy("department").avg("salary")
```

**Comparison Table:**
| Feature | RDD | DataFrame | Dataset |
|---------|-----|-----------|---------|
| Type Safety | Compile-time | Runtime | Compile-time |
| Optimization | None | Catalyst | Catalyst |
| Performance | Slowest | Fast | Fastest |
| API Style | Functional | SQL-like | Both |
| Memory Usage | High | Optimized | Optimized |
| Debugging | Difficult | Easy | Easy |

**When to Use:**
- **RDD**: Low-level operations, unstructured data, complex transformations
- **DataFrame**: SQL-like operations, structured data, best performance
- **Dataset**: Type safety required, structured data, Scala/Java applications

---

### ❓ Question 37. Design data pipeline with database used by computation system and database used by reporting system
#### Answer: 
This architecture separates operational (OLTP) and analytical (OLAP) workloads to optimize performance for different use cases.

**Architecture Overview:**

**1. Operational Database (OLTP - Online Transaction Processing):**
```sql
-- PostgreSQL for transactional workloads
-- Optimized for writes, ACID compliance

-- Users table
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(user_id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10,2),
    status VARCHAR(20)
);

-- Indexes for operational queries
CREATE INDEX idx_user_email ON users(email);
CREATE INDEX idx_order_user_date ON orders(user_id, order_date);
```

**2. ETL Pipeline:**
```python
# Apache Airflow DAG for data pipeline
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

def extract_operational_data():
    # Extract from operational database
    operational_engine = create_engine('postgresql://user:pass@oltp-db:5432/operational')
    
    # Extract incremental data
    query = """
    SELECT u.user_id, u.email, u.created_at,
           o.order_id, o.order_date, o.total_amount, o.status
    FROM users u
    JOIN orders o ON u.user_id = o.user_id
    WHERE o.order_date >= CURRENT_DATE - INTERVAL '1 day'
    """
    
    df = pd.read_sql(query, operational_engine)
    return df

def transform_data(df):
    # Data transformations
    df['order_month'] = df['order_date'].dt.to_period('M')
    df['customer_lifetime_value'] = df.groupby('user_id')['total_amount'].transform('sum')
    
    # Aggregations for reporting
    monthly_summary = df.groupby(['order_month']).agg({
        'order_id': 'count',
        'total_amount': 'sum',
        'user_id': 'nunique'
    }).reset_index()
    
    return df, monthly_summary

def load_to_warehouse(df, summary_df):
    # Load to data warehouse
    warehouse_engine = create_engine('postgresql://user:pass@olap-db:5432/warehouse')
    
    # Load fact table
    df.to_sql('fact_orders', warehouse_engine, if_exists='append', index=False)
    
    # Load aggregated data
    summary_df.to_sql('monthly_summary', warehouse_engine, if_exists='append', index=False)
```

**3. Data Warehouse (OLAP - Online Analytical Processing):**
```sql
-- Data warehouse schema optimized for analytics
-- Star schema design

-- Dimension tables
CREATE TABLE dim_users (
    user_sk SERIAL PRIMARY KEY,
    user_id INT,
    email VARCHAR(255),
    registration_date DATE,
    user_segment VARCHAR(50),
    scd_start_date DATE,
    scd_end_date DATE,
    is_current BOOLEAN
);

CREATE TABLE dim_date (
    date_sk SERIAL PRIMARY KEY,
    date_actual DATE,
    year INT,
    quarter INT,
    month INT,
    week INT,
    day_of_week INT,
    is_weekend BOOLEAN
);

-- Fact table
CREATE TABLE fact_orders (
    order_sk SERIAL PRIMARY KEY,
    user_sk INT REFERENCES dim_users(user_sk),
    date_sk INT REFERENCES dim_date(date_sk),
    order_id INT,
    total_amount DECIMAL(10,2),
    quantity INT,
    profit_margin DECIMAL(5,2)
);

-- Columnstore indexes for analytics
CREATE INDEX idx_fact_orders_date ON fact_orders(date_sk);
CREATE INDEX idx_fact_orders_user ON fact_orders(user_sk);
```

**4. Real-time Streaming (Optional):**
```python
# Kafka + Spark Streaming for real-time analytics
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark = SparkSession.builder.appName("RealTimeAnalytics").getOrCreate()

# Read from Kafka
stream_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "orders") \
    .load()

# Parse JSON and transform
orders_df = stream_df.select(
    from_json(col("value").cast("string"), order_schema).alias("order")
).select("order.*")

# Real-time aggregations
hourly_stats = orders_df \
    .withWatermark("order_timestamp", "1 hour") \
    .groupBy(window(col("order_timestamp"), "1 hour")) \
    .agg(
        count("*").alias("order_count"),
        sum("total_amount").alias("total_revenue")
    )

# Write to warehouse
query = hourly_stats.writeStream \
    .outputMode("append") \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://warehouse:5432/analytics") \
    .option("dbtable", "real_time_metrics") \
    .trigger(processingTime="10 minutes") \
    .start()
```

**5. Reporting Layer:**
```python
# Reporting API
from flask import Flask, jsonify
import psycopg2

app = Flask(__name__)

@app.route('/api/sales/monthly')
def monthly_sales():
    conn = psycopg2.connect(
        host="olap-db",
        database="warehouse",
        user="reporting_user",
        password="password"
    )
    
    query = """
    SELECT 
        d.year,
        d.month,
        SUM(f.total_amount) as total_sales,
        COUNT(f.order_id) as order_count
    FROM fact_orders f
    JOIN dim_date d ON f.date_sk = d.date_sk
    WHERE d.year = 2024
    GROUP BY d.year, d.month
    ORDER BY d.year, d.month
    """
    
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    
    return jsonify({
        'monthly_sales': [
            {'year': row[0], 'month': row[1], 'sales': row[2], 'orders': row[3]}
            for row in results
        ]
    })
```

**6. Architecture Benefits:**
- **Separation of Concerns**: OLTP optimized for transactions, OLAP for analytics
- **Performance**: No interference between operational and analytical workloads
- **Scalability**: Each system can scale independently
- **Data Quality**: ETL process ensures data consistency
- **Historical Data**: Warehouse maintains historical snapshots

---

### ❓ Question 38. Python Regular Expressions
#### Answer: 
Regular expressions (regex) are powerful pattern-matching tools for text processing in Python using the `re` module.

**Basic Syntax and Metacharacters:**
```python
import re

# Basic patterns
pattern = r'hello'          # Literal string
pattern = r'h.llo'          # . matches any character
pattern = r'h.*llo'         # .* matches any characters (greedy)
pattern = r'h.+llo'         # .+ matches one or more characters
pattern = r'h.?llo'         # .? matches zero or one character
```

**Character Classes:**
```python
# Predefined character classes
pattern = r'\d+'            # One or more digits
pattern = r'\w+'            # One or more word characters
pattern = r'\s+'            # One or more whitespace
pattern = r'[a-zA-Z]+'      # Custom character class
pattern = r'[^0-9]+'        # Negated character class (not digits)

# Examples
text = "Phone: 123-456-7890"
phone = re.search(r'\d{3}-\d{3}-\d{4}', text)
print(phone.group())  # Output: 123-456-7890
```

**Quantifiers:**
```python
# Quantifiers
pattern = r'a{3}'           # Exactly 3 'a's
pattern = r'a{2,5}'         # Between 2 and 5 'a's
pattern = r'a{2,}'          # 2 or more 'a's
pattern = r'a*'             # Zero or more 'a's
pattern = r'a+'             # One or more 'a's
pattern = r'a?'             # Zero or one 'a'

# Example: Validate email
email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
email = "user@example.com"
if re.match(email_pattern, email):
    print("Valid email")
```

**Groups and Capturing:**
```python
# Groups with parentheses
text = "Name: John Doe, Age: 30"
pattern = r'Name: (\w+\s\w+), Age: (\d+)'
match = re.search(pattern, text)

if match:
    name = match.group(1)    # John Doe
    age = match.group(2)     # 30
    print(f"Name: {name}, Age: {age}")

# Named groups
pattern = r'Name: (?P<name>\w+\s\w+), Age: (?P<age>\d+)'
match = re.search(pattern, text)
if match:
    print(match.groupdict())  # {'name': 'John Doe', 'age': '30'}
```

**Common Methods:**
```python
import re

text = "The quick brown fox jumps over the lazy dog"

# re.search() - finds first match
match = re.search(r'brown', text)
print(match.group())  # brown

# re.findall() - finds all matches
words = re.findall(r'\b\w+\b', text)
print(words)  # ['The', 'quick', 'brown', 'fox', ...]

# re.finditer() - iterator of match objects
for match in re.finditer(r'\b\w+\b', text):
    print(f"Word: {match.group()}, Position: {match.start()}-{match.end()}")

# re.sub() - substitute matches
new_text = re.sub(r'brown', 'red', text)
print(new_text)  # The quick red fox jumps over the lazy dog

# re.split() - split by pattern
parts = re.split(r'\s+', text)
print(parts)  # ['The', 'quick', 'brown', 'fox', ...]
```

**Practical Examples:**
```python
# 1. Extract URLs from text
text = "Visit https://example.com or http://test.org"
urls = re.findall(r'https?://[^\s]+', text)
print(urls)  # ['https://example.com', 'http://test.org']

# 2. Parse log files
log_line = '192.168.1.1 - - [25/Dec/2023:10:00:00 +0000] "GET /index.html HTTP/1.1" 200 1024'
log_pattern = r'(\d+\.\d+\.\d+\.\d+).*?\[([^\]]+)\].*?"(\w+)\s+([^"]+)".*?(\d+)\s+(\d+)'
match = re.search(log_pattern, log_line)
if match:
    ip, timestamp, method, path, status, size = match.groups()
    print(f"IP: {ip}, Method: {method}, Path: {path}, Status: {status}")

# 3. Data validation
def validate_input(data):
    patterns = {
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'phone': r'^\+?1?[-.\s]?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})$',
        'ssn': r'^\d{3}-\d{2}-\d{4}$',
        'credit_card': r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$'
    }
    
    for field, pattern in patterns.items():
        if field in data:
            if not re.match(pattern, data[field]):
                return False, f"Invalid {field} format"
    return True, "Valid"

# 4. Text processing
def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Convert to lowercase
    text = text.lower().strip()
    return text

# 5. Extract structured data
html = '<div class="product" data-price="29.99">Product Name</div>'
price_pattern = r'data-price="([\d.]+)"'
name_pattern = r'>([^<]+)</div>'

price = re.search(price_pattern, html).group(1)
name = re.search(name_pattern, html).group(1)
print(f"Product: {name}, Price: ${price}")
```

**Advanced Features:**
```python
# Lookahead and lookbehind
text = "password123"
# Positive lookahead - password followed by digits
pattern = r'password(?=\d+)'
match = re.search(pattern, text)

# Negative lookahead - password not followed by digits
pattern = r'password(?!\d+)'

# Positive lookbehind - digits preceded by password
pattern = r'(?<=password)\d+'

# Non-greedy matching
text = "<div>Hello</div><div>World</div>"
greedy = re.findall(r'<div>.*</div>', text)      # ['<div>Hello</div><div>World</div>']
non_greedy = re.findall(r'<div>.*?</div>', text) # ['<div>Hello</div>', '<div>World</div>']

# Compiled patterns for better performance
pattern = re.compile(r'\d+')
numbers = pattern.findall("123 456 789")
```

---

### ❓ Question 39. Find the errors/issues in PySpark logs
#### Answer: 
Analyzing PySpark logs requires understanding common error patterns and debugging techniques.

**Common PySpark Error Types:**

**1. Memory-Related Errors:**
```python
# Common memory errors in logs
"""
ERROR Executor: Exception in task 0.0 in stage 1.0 (TID 1)
java.lang.OutOfMemoryError: Java heap space
"""

# Solutions:
spark.conf.set("spark.executor.memory", "8g")
spark.conf.set("spark.executor.memoryFraction", "0.8")
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

# Identify memory issues
def analyze_memory_usage(df):
    # Check partition sizes
    partition_sizes = df.rdd.mapPartitions(lambda x: [sum(1 for _ in x)]).collect()
    print(f"Partition sizes: {partition_sizes}")
    
    # Repartition if skewed
    if max(partition_sizes) > 2 * sum(partition_sizes) / len(partition_sizes):
        df = df.repartition(200)
    
    return df
```

**2. Serialization Errors:**
```python
# Log pattern:
"""
ERROR TaskSetManager: Task 0 in stage 1.0 failed 4 times, most recent failure:
NotSerializableException: java.io.NotSerializableException
"""

# Common causes and solutions:
# Bad - referencing non-serializable objects
class DataProcessor:
    def __init__(self):
        self.connection = create_db_connection()  # Not serializable
    
    def process_data(self, df):
        return df.map(lambda x: self.connection.query(x))  # Error!

# Good - create connections inside functions
def process_data(df):
    def process_partition(partition):
        connection = create_db_connection()  # Create per partition
        for row in partition:
            yield connection.query(row)
        connection.close()
    
    return df.mapPartitions(process_partition)
```

**3. Schema and Data Type Errors:**
```python
# Log pattern:
"""
AnalysisException: cannot resolve column_name given input columns
"""

# Debugging schema issues
def debug_schema_issues(df):
    print("Schema:")
    df.printSchema()
    
    print("Columns:")
    print(df.columns)
    
    # Check for null values
    null_counts = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
    null_counts.show()
    
    # Check data types
    for col_name, col_type in df.dtypes:
        print(f"{col_name}: {col_type}")

# Common schema fixes
def fix_schema_issues(df):
    # Cast columns to correct types
    df = df.withColumn("price", col("price").cast("double"))
    df = df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
    
    # Handle missing columns
    required_columns = ["id", "name", "price", "date"]
    for col_name in required_columns:
        if col_name not in df.columns:
            df = df.withColumn(col_name, lit(None))
    
    return df
```

**4. File and Path Errors:**
```python
# Log pattern:
"""
FileNotFoundException: File does not exist: /path/to/file
PermissionDeniedException: Permission denied
"""

# Debugging file issues
def debug_file_issues(spark, file_path):
    import os
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    # Check permissions
    if not os.access(file_path, os.R_OK):
        print(f"No read permission: {file_path}")
        return False
    
    # Check file format
    try:
        df = spark.read.option("header", "true").csv(file_path)
        print(f"Successfully read file: {file_path}")
        return True
    except Exception as e:
        print(f"Error reading file: {e}")
        return False
```

**5. Network and Connection Errors:**
```python
# Log pattern:
"""
ConnectionException: Connection refused
TimeoutException: Timeout waiting for connection
"""

# Connection retry logic
def safe_database_read(spark, jdbc_url, table, retries=3):
    for attempt in range(retries):
        try:
            df = spark.read \
                .format("jdbc") \
                .option("url", jdbc_url) \
                .option("dbtable", table) \
                .option("driver", "org.postgresql.Driver") \
                .load()
            return df
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

**6. Stage and Task Failures:**
```python
# Log pattern:
"""
ERROR TaskSchedulerImpl: Lost executor 1: Remote RPC client disassociated
WARN TaskSetManager: Lost task 0.0 in stage 1.0
"""

# Debugging task failures
def analyze_task_failures(spark):
    # Check executor logs
    print("Executor configuration:")
    print(f"Executor memory: {spark.conf.get('spark.executor.memory')}")
    print(f"Executor cores: {spark.conf.get('spark.executor.cores')}")
    print(f"Number of executors: {spark.conf.get('spark.executor.instances')}")
    
    # Monitor task metrics
    def monitor_stage_progress(df):
        # Add checkpoints for long pipelines
        df.checkpoint()
        
        # Cache intermediate results
        df.cache()
        
        # Force evaluation to check for errors
        df.count()
        
        return df
```

**Log Analysis Script:**
```python
import re
from collections import defaultdict

def analyze_spark_logs(log_file_path):
    """
    Analyze PySpark logs to identify common issues
    """
    error_patterns = {
        'oom': r'OutOfMemoryError|GC overhead limit exceeded',
        'serialization': r'NotSerializableException|Serialization',
        'network': r'Connection refused|TimeoutException|Lost executor',
        'file': r'FileNotFoundException|PermissionDeniedException',
        'schema': r'AnalysisException|cannot resolve|Column.*not found',
        'stage_failure': r'Lost task|Stage.*failed|Task.*failed'
    }
    
    error_counts = defaultdict(int)
    error_details = defaultdict(list)
    
    with open(log_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            for error_type, pattern in error_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    error_counts[error_type] += 1
                    error_details[error_type].append((line_num, line.strip()))
    
    # Generate report
    print("=== PySpark Log Analysis Report ===")
    print(f"Total errors found: {sum(error_counts.values())}")
    print("\nError breakdown:")
    
    for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{error_type}: {count} occurrences")
        
        # Show first few instances
        print("Sample occurrences:")
        for line_num, line in error_details[error_type][:3]:
            print(f"  Line {line_num}: {line[:100]}...")
        print()
    
    return error_counts, error_details

# Usage
error_counts, error_details = analyze_spark_logs("/path/to/spark-logs/app.log")
```

**Performance Debugging:**
```python
def debug_performance_issues(spark, df):
    """
    Debug performance issues in PySpark
    """
    # Check execution plan
    print("=== Execution Plan ===")
    df.explain(True)
    
    # Check partition distribution
    print("\n=== Partition Analysis ===")
    partition_count = df.rdd.getNumPartitions()
    print(f"Number of partitions: {partition_count}")
    
    # Check for data skew
    partition_sizes = df.rdd.mapPartitions(lambda x: [sum(1 for _ in x)]).collect()
    print(f"Partition sizes: {partition_sizes}")
    
    if partition_sizes:
        avg_size = sum(partition_sizes) / len(partition_sizes)
        max_size = max(partition_sizes)
        skew_ratio = max_size / avg_size if avg_size > 0 else 0
        
        print(f"Average partition size: {avg_size:.2f}")
        print(f"Max partition size: {max_size}")
        print(f"Skew ratio: {skew_ratio:.2f}")
        
        if skew_ratio > 2.0:
            print("WARNING: Data skew detected!")
    
    # Memory usage analysis
    print("\n=== Memory Configuration ===")
    memory_configs = [
        'spark.executor.memory',
        'spark.executor.memoryFraction',
        'spark.driver.memory',
        'spark.driver.maxResultSize'
    ]
    
    for config in memory_configs:
        try:
            value = spark.conf.get(config)
            print(f"{config}: {value}")
        except:
            print(f"{config}: not set")

# Common optimization patterns
def optimize_dataframe(df):
    """
    Apply common optimizations to DataFrame
    """
    # Filter early
    df = df.filter(df.status == 'active')
    
    # Project only needed columns
    df = df.select('id', 'name', 'price', 'date')
    
    # Repartition if needed
    if df.rdd.getNumPartitions() > 200:
        df = df.repartition(200)
    
    # Cache if used multiple times
    df.cache()
    
    return df
```

---

### ❓ Question 40. PySpark queries: Avg, min, max price per month using DataFrame or SparkSQL code. Source date field in format yyyy-mm-dd. 2nd, 3rd, 4th highest/lowest price per month
#### Answer: 
Here are PySpark solutions for calculating price statistics per month and finding ranked prices.

**Sample Data Setup:**
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *

# Initialize Spark
spark = SparkSession.builder.appName("PriceAnalysis").getOrCreate()

# Sample data
data = [
    ("2024-01-15", 100.50),
    ("2024-01-20", 150.75),
    ("2024-01-25", 200.00),
    ("2024-02-01", 120.25),
    ("2024-02-10", 180.50),
    ("2024-02-15", 90.75),
    ("2024-03-05", 300.00),
    ("2024-03-12", 250.50),
    ("2024-03-20", 175.25)
]

schema = StructType([
    StructField("date", StringType(), True),
    StructField("price", DoubleType(), True)
])

df = spark.createDataFrame(data, schema)
```

**1. Basic Statistics (Avg, Min, Max) per Month - DataFrame API:**
```python
# Convert date string to date type and extract month
df_with_month = df.withColumn("date", to_date(col("date"), "yyyy-MM-dd")) \
                  .withColumn("year", year(col("date"))) \
                  .withColumn("month", month(col("date"))) \
                  .withColumn("year_month", date_format(col("date"), "yyyy-MM"))

# Calculate basic statistics per month
monthly_stats = df_with_month.groupBy("year_month") \
    .agg(
        avg("price").alias("avg_price"),
        min("price").alias("min_price"),
        max("price").alias("max_price"),
        count("price").alias("record_count")
    ) \
    .orderBy("year_month")

monthly_stats.show()
```

**2. Basic Statistics - SparkSQL:**
```python
# Register DataFrame as temporary view
df.createOrReplaceTempView("price_data")

# SQL query for basic statistics
spark.sql("""
    SELECT 
        DATE_FORMAT(TO_DATE(date, 'yyyy-MM-dd'), 'yyyy-MM') as year_month,
        ROUND(AVG(price), 2) as avg_price,
        MIN(price) as min_price,
        MAX(price) as max_price,
        COUNT(*) as record_count
    FROM price_data
    GROUP BY DATE_FORMAT(TO_DATE(date, 'yyyy-MM-dd'), 'yyyy-MM')
    ORDER BY year_month
""").show()
```

**3. 2nd, 3rd, 4th Highest Prices per Month - DataFrame API:**
```python
# Window function to rank prices within each month
window_spec = Window.partitionBy("year_month").orderBy(col("price").desc())

# Add ranking columns
df_ranked = df_with_month.withColumn("price_rank", row_number().over(window_spec)) \
                         .withColumn("dense_rank", dense_rank().over(window_spec))

# Filter for 2nd, 3rd, 4th highest prices
highest_prices = df_ranked.filter(col("price_rank").isin([2, 3, 4])) \
                         .select("year_month", "date", "price", "price_rank") \
                         .orderBy("year_month", "price_rank")

print("2nd, 3rd, 4th Highest Prices per Month:")
highest_prices.show()

# Pivot to show ranks as columns
highest_prices_pivot = df_ranked.filter(col("price_rank").isin([2, 3, 4])) \
    .groupBy("year_month") \
    .pivot("price_rank") \
    .agg(first("price")) \
    .withColumnRenamed("2", "2nd_highest") \
    .withColumnRenamed("3", "3rd_highest") \
    .withColumnRenamed("4", "4th_highest") \
    .orderBy("year_month")

print("\nHighest Prices Pivot View:")
highest_prices_pivot.show()
```

**4. 2nd, 3rd, 4th Lowest Prices per Month - DataFrame API:**
```python
# Window function to rank prices (ascending for lowest)
window_spec_asc = Window.partitionBy("year_month").orderBy(col("price").asc())

# Add ranking for lowest prices
df_ranked_low = df_with_month.withColumn("price_rank_low", row_number().over(window_spec_asc))

# Filter for 2nd, 3rd, 4th lowest prices
lowest_prices = df_ranked_low.filter(col("price_rank_low").isin([2, 3, 4])) \
                            .select("year_month", "date", "price", "price_rank_low") \
                            .orderBy("year_month", "price_rank_low")

print("2nd, 3rd, 4th Lowest Prices per Month:")
lowest_prices.show()

# Pivot for lowest prices
lowest_prices_pivot = df_ranked_low.filter(col("price_rank_low").isin([2, 3, 4])) \
    .groupBy("year_month") \
    .pivot("price_rank_low") \
    .agg(first("price")) \
    .withColumnRenamed("2", "2nd_lowest") \
    .withColumnRenamed("3", "3rd_lowest") \
    .withColumnRenamed("4", "4th_lowest") \
    .orderBy("year_month")

print("\nLowest Prices Pivot View:")
lowest_prices_pivot.show()
```

**5. Combined Analysis - SparkSQL:**
```python
# Comprehensive SQL query with CTEs
spark.sql("""
    WITH monthly_data AS (
        SELECT 
            DATE_FORMAT(TO_DATE(date, 'yyyy-MM-dd'), 'yyyy-MM') as year_month,
            TO_DATE(date, 'yyyy-MM-dd') as date,
            price
        FROM price_data
    ),
    
    ranked_prices AS (
        SELECT 
            year_month,
            date,
            price,
            ROW_NUMBER() OVER (PARTITION BY year_month ORDER BY price DESC) as rank_desc,
            ROW_NUMBER() OVER (PARTITION BY year_month ORDER BY price ASC) as rank_asc
        FROM monthly_data
    ),
    
    basic_stats AS (
        SELECT 
            year_month,
            ROUND(AVG(price), 2) as avg_price,
            MIN(price) as min_price,
            MAX(price) as max_price,
            COUNT(*) as record_count
        FROM monthly_data
        GROUP BY year_month
    ),
    
    highest_prices AS (
        SELECT 
            year_month,
            MAX(CASE WHEN rank_desc = 2 THEN price END) as 2nd_highest,
            MAX(CASE WHEN rank_desc = 3 THEN price END) as 3rd_highest,
            MAX(CASE WHEN rank_desc = 4 THEN price END) as 4th_highest
        FROM ranked_prices
        GROUP BY year_month
    ),
    
    lowest_prices AS (
        SELECT 
            year_month,
            MAX(CASE WHEN rank_asc = 2 THEN price END) as 2nd_lowest,
            MAX(CASE WHEN rank_asc = 3 THEN price END) as 3rd_lowest,
            MAX(CASE WHEN rank_asc = 4 THEN price END) as 4th_lowest
        FROM ranked_prices
        GROUP BY year_month
    )
    
    SELECT 
        bs.year_month,
        bs.avg_price,
        bs.min_price,
        bs.max_price,
        bs.record_count,
        hp.2nd_highest,
        hp.3rd_highest,
        hp.4th_highest,
        lp.2nd_lowest,
        lp.3rd_lowest,
        lp.4th_lowest
    FROM basic_stats bs
    LEFT JOIN highest_prices hp ON bs.year_month = hp.year_month
    LEFT JOIN lowest_prices lp ON bs.year_month = lp.year_month
    ORDER BY bs.year_month
""").show()
```

**6. Advanced Analysis with Percentiles:**
```python
# Using percentile functions for more robust analysis
from pyspark.sql.functions import expr

# Calculate percentiles and quartiles
percentile_analysis = df_with_month.groupBy("year_month") \
    .agg(
        avg("price").alias("avg_price"),
        min("price").alias("min_price"),
        max("price").alias("max_price"),
        expr("percentile_approx(price, 0.25)").alias("q1_25th"),
        expr("percentile_approx(price, 0.5)").alias("median_50th"),
        expr("percentile_approx(price, 0.75)").alias("q3_75th"),
        expr("percentile_approx(price, 0.9)").alias("p90_90th"),
        count("price").alias("record_count")
    ) \
    .orderBy("year_month")

print("Percentile Analysis:")
percentile_analysis.show()
```

**7. Performance Optimized Version:**
```python
# Optimized version for large datasets
def analyze_monthly_prices(df, date_col="date", price_col="price"):
    """
    Optimized function for monthly price analysis
    """
    # Cache the base DataFrame if used multiple times
    df.cache()
    
    # Single pass through data with window functions
    window_desc = Window.partitionBy("year_month").orderBy(col(price_col).desc())
    window_asc = Window.partitionBy("year_month").orderBy(col(price_col).asc())
    
    # Comprehensive analysis in single transformation
    result = df.withColumn("date", to_date(col(date_col), "yyyy-MM-dd")) \
              .withColumn("year_month", date_format(col("date"), "yyyy-MM")) \
              .withColumn("rank_desc", row_number().over(window_desc)) \
              .withColumn("rank_asc", row_number().over(window_asc)) \
              .groupBy("year_month") \
              .agg(
                  avg(price_col).alias("avg_price"),
                  min(price_col).alias("min_price"),
                  max(price_col).alias("max_price"),
                  count(price_col).alias("record_count"),
                  # Highest prices
                  max(when(col("rank_desc") == 2, col(price_col))).alias("2nd_highest"),
                  max(when(col("rank_desc") == 3, col(price_col))).alias("3rd_highest"),
                  max(when(col("rank_desc") == 4, col(price_col))).alias("4th_highest"),
                  # Lowest prices  
                  max(when(col("rank_asc") == 2, col(price_col))).alias("2nd_lowest"),
                  max(when(col("rank_asc") == 3, col(price_col))).alias("3rd_lowest"),
                  max(when(col("rank_asc") == 4, col(price_col))).alias("4th_lowest")
              ) \
              .orderBy("year_month")
    
    return result

# Usage
comprehensive_analysis = analyze_monthly_prices(df)
comprehensive_analysis.show()
```

**8. Output Format Example:**
```
+----------+---------+---------+---------+------------+------------+------------+------------+-----------+-----------+-----------+
|year_month|avg_price|min_price|max_price|record_count|2nd_highest|3rd_highest|4th_highest|2nd_lowest|3rd_lowest|4th_lowest|
+----------+---------+---------+---------+------------+------------+------------+------------+-----------+-----------+-----------+
|   2024-01|   150.42|   100.50|   200.00|           3|      150.75|      100.50|        null|     150.75|     200.00|      null|
|   2024-02|   130.50|    90.75|   180.50|           3|      120.25|       90.75|        null|     120.25|     180.50|      null|
|   2024-03|   241.92|   175.25|   300.00|           3|      250.50|      175.25|        null|     250.50|     300.00|      null|
+----------+---------+---------+---------+------------+------------+------------+------------+-----------+-----------+-----------+
```

These solutions provide comprehensive monthly price analysis with both DataFrame API and SparkSQL approaches, handling edge cases where fewer than 4 records exist per month.

---

### ❓ Question 41. What is DAG, Difference between logical and physical plan
#### Answer: 
DAG (Directed Acyclic Graph) is a fundamental concept in distributed computing systems like Apache Spark, representing a sequence of computations where data flows in one direction without cycles.

**What is DAG:**
- **Directed**: Edges have direction (data flows from source to destination)
- **Acyclic**: No circular dependencies or loops
- **Graph**: Nodes represent operations, edges represent data dependencies

**DAG in Spark Context:**
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("DAG Example").getOrCreate()

# Example DAG creation
df1 = spark.read.csv("sales.csv", header=True)          # Node 1: Read
df2 = df1.filter(col("amount") > 100)                   # Node 2: Filter
df3 = df2.groupBy("category").sum("amount")             # Node 3: GroupBy
df4 = df3.orderBy(col("sum(amount)").desc())            # Node 4: Sort
df4.show()                                              # Action: Triggers execution

# Visualize the DAG
df4.explain(True)  # Shows both logical and physical plans
```

**Logical Plan:**
The logical plan represents *what* operations need to be performed, focusing on the high-level transformations without considering execution details.

**Characteristics of Logical Plan:**
- High-level operations (filter, join, aggregate)
- No physical execution details
- Platform-independent
- Result of query parsing and analysis

**Example Logical Plan:**
```
== Parsed Logical Plan ==
Sort [sum(amount)#10 DESC NULLS LAST], true
+- Aggregate [category#3], [category#3, sum(amount#4) AS sum(amount)#10]
   +- Filter (amount#4 > 100)
      +- Relation[id#2,category#3,amount#4] csv

== Analyzed Logical Plan ==
category: string, sum(amount): double
Sort [sum(amount)#10 DESC NULLS LAST], true
+- Aggregate [category#3], [category#3, sum(cast(amount#4 as double)) AS sum(amount)#10]
   +- Filter (cast(amount#4 as double) > 100.0)
      +- Relation[id#2,category#3,amount#4] csv
```

**Physical Plan:**
The physical plan represents *how* operations will be executed, including specific algorithms, data partitioning, and resource allocation.

**Characteristics of Physical Plan:**
- Specific execution algorithms (HashAggregate, SortMergeJoin)
- Partitioning and shuffling details
- Resource allocation and parallelization
- Platform-specific optimizations

**Example Physical Plan:**
```
== Physical Plan ==
*(3) Sort [sum(amount)#10 DESC NULLS LAST], true, 0
+- Exchange rangepartitioning(sum(amount)#10 DESC NULLS LAST, 200), ENSURE_REQUIREMENTS, [id=#25]
   +- *(2) HashAggregate(keys=[category#3], functions=[sum(cast(amount#4 as double))])
      +- Exchange hashpartitioning(category#3, 200), ENSURE_REQUIREMENTS, [id=#21]
         +- *(1) HashAggregate(keys=[category#3], functions=[partial_sum(cast(amount#4 as double))])
            +- *(1) Project [category#3, amount#4]
               +- *(1) Filter (isnotnull(amount#4) AND (cast(amount#4 as double) > 100.0))
                  +- FileScan csv [id#2,category#3,amount#4] Batched: false, DataFilters: [isnotnull(amount#4), (cast(amount#4 as double) > 100.0)], Format: CSV, Location: InMemoryFileIndex[file:/path/sales.csv], PartitionFilters: [], PushedFilters: [IsNotNull(amount)], ReadSchema: struct<id:string,category:string,amount:string>
```

**Key Differences:**

| Aspect | Logical Plan | Physical Plan |
|--------|--------------|---------------|
| **Purpose** | What to compute | How to compute |
| **Level** | High-level operations | Low-level execution details |
| **Optimization** | Rule-based optimization | Cost-based optimization |
| **Algorithms** | Generic operations | Specific algorithms (HashJoin, SortMergeJoin) |
| **Partitioning** | Not specified | Explicit partitioning strategies |
| **Resources** | Not considered | Memory, CPU, network usage |
| **Execution** | Platform-independent | Platform-specific |

**DAG Optimization Process:**
```python
# Example of how Spark optimizes DAG
def demonstrate_dag_optimization():
    # Original transformations
    df = spark.read.csv("data.csv", header=True)
    df1 = df.select("id", "name", "age", "salary")
    df2 = df1.filter(col("age") > 25)
    df3 = df2.filter(col("salary") > 50000)
    df4 = df3.select("id", "name", "salary")
    
    # Logical plan will show separate operations
    print("=== Logical Plan ===")
    df4.explain(extended=False)
    
    # Physical plan will show optimized execution
    # - Predicate pushdown: combines filters
    # - Projection pushdown: eliminates unused columns early
    # - Column pruning: only reads necessary columns
    print("=== Optimized Physical Plan ===")
    df4.explain(mode="formatted")

# Catalyst Optimizer stages
"""
1. Logical Plan Creation
2. Logical Plan Optimization (Rule-based)
   - Predicate Pushdown
   - Projection Pushdown
   - Column Pruning
   - Constant Folding
3. Physical Plan Generation
4. Physical Plan Optimization (Cost-based)
   - Join Reordering
   - Algorithm Selection
5. Code Generation
"""
```

**Advanced DAG Concepts:**
```python
# Lazy evaluation and DAG construction
def understand_lazy_evaluation():
    # These operations build the DAG but don't execute
    df = spark.read.csv("large_dataset.csv")
    filtered_df = df.filter(col("status") == "active")
    grouped_df = filtered_df.groupBy("category").count()
    
    # DAG is built but not executed yet
    print("DAG constructed but not executed")
    
    # Action triggers DAG execution
    result = grouped_df.collect()  # Now the entire DAG executes
    
    # Benefits of lazy evaluation:
    # 1. Entire pipeline can be optimized together
    # 2. Unnecessary operations can be eliminated
    # 3. Data doesn't move until necessary
    # 4. Better fault tolerance through lineage

# DAG lineage and fault tolerance
def demonstrate_fault_tolerance():
    # Each RDD/DataFrame maintains lineage information
    df1 = spark.read.csv("input.csv")
    df2 = df1.filter(col("age") > 18)
    df3 = df2.groupBy("department").count()
    
    # If a partition is lost, Spark can reconstruct it
    # using the lineage information stored in the DAG
    
    # Check lineage
    print("RDD Lineage:")
    print(df3.rdd.toDebugString())
```

**DAG Visualization and Monitoring:**
```python
# Access Spark UI to visualize DAG
def monitor_dag_execution():
    # Spark UI shows:
    # 1. DAG visualization
    # 2. Stage boundaries
    # 3. Task execution details
    # 4. Shuffle operations
    
    # Programmatic access to execution plan
    df = spark.read.csv("data.csv").filter(col("age") > 25)
    
    # Get query execution details
    execution_plan = df.queryExecution
    print("Logical Plan:", execution_plan.logical)
    print("Optimized Plan:", execution_plan.optimizedPlan)
    print("Physical Plan:", execution_plan.executedPlan)
    
    # Stage information
    print("Number of stages:", execution_plan.executedPlan.execute().getNumPartitions())
```

---

### ❓ Question 42. Architecture of MongoDB, How does reads/writes work in it
#### Answer: 
MongoDB is a document-oriented NoSQL database with a distributed architecture designed for scalability, high availability, and flexible data modeling.

**MongoDB Architecture Components:**

**1. Mongod (MongoDB Daemon):**
- Primary database process
- Handles data requests, manages data access
- Maintains database files and indexes

**2. Storage Engine:**
```javascript
// WiredTiger Storage Engine (default)
{
  "storage": {
    "engine": "wiredTiger",
    "wiredTiger": {
      "engineConfig": {
        "cacheSizeGB": 1,
        "directoryForIndexes": true
      },
      "collectionConfig": {
        "blockCompressor": "snappy"
      }
    }
  }
}
```

**3. Replica Set Architecture:**
```javascript
// Replica Set Configuration
rs.initiate({
  "_id": "myReplicaSet",
  "members": [
    { "_id": 0, "host": "mongodb1:27017", "priority": 2 },
    { "_id": 1, "host": "mongodb2:27017", "priority": 1 },
    { "_id": 2, "host": "mongodb3:27017", "priority": 1, "arbiterOnly": true }
  ]
})

// Roles in Replica Set:
// Primary: Handles all writes, can handle reads
// Secondary: Replicates from primary, can handle reads
// Arbiter: Participates in elections, doesn't hold data
```

**4. Sharded Cluster Architecture:**
```javascript
// Sharded cluster components:
// - Config Servers: Store cluster metadata
// - Shard Servers: Store actual data
// - Query Routers (mongos): Route queries to appropriate shards

// Shard key example
sh.shardCollection("mydb.orders", { "customer_id": 1, "order_date": 1 })
```

**MongoDB Read Operations:**

**1. Read Process Flow:**
```javascript
// Read operation flow:
// 1. Client sends read request
// 2. Mongod receives request
// 3. Query parser analyzes query
// 4. Query optimizer creates execution plan
// 5. Storage engine retrieves data
// 6. Results returned to client

// Example read operation
db.users.find({ "age": { $gt: 25 } }).limit(10)
```

**2. Read Preferences:**
```javascript
// Primary (default): Read from primary only
db.users.find().readPref("primary")

// PrimaryPreferred: Read from primary, fallback to secondary
db.users.find().readPref("primaryPreferred")

// Secondary: Read from secondary only
db.users.find().readPref("secondary")

// SecondaryPreferred: Read from secondary, fallback to primary
db.users.find().readPref("secondaryPreferred")

// Nearest: Read from nearest member (lowest latency)
db.users.find().readPref("nearest")
```

**3. Read Concerns:**
```javascript
// Read concern levels
// local: Returns most recent data available
db.users.find().readConcern("local")

// available: Returns data that may be rolled back
db.users.find().readConcern("available")

// majority: Returns data acknowledged by majority of replica set
db.users.find().readConcern("majority")

// linearizable: Returns data that reflects all prior writes
db.users.find().readConcern("linearizable")
```

**4. Index Usage in Reads:**
```javascript
// Create indexes for efficient reads
db.users.createIndex({ "email": 1 })  // Single field index
db.users.createIndex({ "age": 1, "city": 1 })  // Compound index
db.users.createIndex({ "profile.bio": "text" })  // Text index

// Query with index usage
db.users.find({ "email": "john@example.com" }).explain("executionStats")

// Index scan vs Collection scan
// Index scan: Uses index to find documents
// Collection scan: Scans entire collection
```

**MongoDB Write Operations:**

**1. Write Process Flow:**
```javascript
// Write operation flow:
// 1. Client sends write request to primary
// 2. Primary validates the operation
// 3. Primary applies write to its data
// 4. Primary logs operation in oplog
// 5. Secondaries replicate from oplog
// 6. Acknowledgment sent to client (based on write concern)

// Example write operations
db.users.insertOne({ "name": "John", "age": 30 })
db.users.updateOne({ "_id": ObjectId("...") }, { $set: { "age": 31 } })
db.users.deleteOne({ "_id": ObjectId("...") })
```

**2. Write Concerns:**
```javascript
// Write concern levels
// w: 1 (default): Acknowledge after writing to primary
db.users.insertOne({ "name": "John" }, { writeConcern: { w: 1 } })

// w: "majority": Acknowledge after majority of replica set members
db.users.insertOne({ "name": "John" }, { writeConcern: { w: "majority" } })

// w: 0: No acknowledgment (fire and forget)
db.users.insertOne({ "name": "John" }, { writeConcern: { w: 0 } })

// j: true: Acknowledge after writing to journal
db.users.insertOne({ "name": "John" }, { writeConcern: { w: 1, j: true } })

// wtimeout: Timeout for write concern
db.users.insertOne(
  { "name": "John" }, 
  { writeConcern: { w: "majority", wtimeout: 5000 } }
)
```

**3. Transactions and ACID Properties:**
```javascript
// Multi-document transactions
const session = db.getMongo().startSession()
session.startTransaction()

try {
  db.accounts.updateOne(
    { "_id": "account1" },
    { $inc: { "balance": -100 } },
    { session: session }
  )
  
  db.accounts.updateOne(
    { "_id": "account2" },
    { $inc: { "balance": 100 } },
    { session: session }
  )
  
  session.commitTransaction()
} catch (error) {
  session.abortTransaction()
  throw error
} finally {
  session.endSession()
}
```

**4. Oplog (Operations Log):**
```javascript
// Oplog structure
{
  "ts": ...,        // Timestamp
  "t": ...,         // Term
  "h": ...,         // Hash
  "v": 2,           // Version
  "op": "i",        // Operation type (i=insert, u=update, d=delete)
  "ns": "mydb.users", // Namespace
  "o": { ... }      // Operation document
}

// View oplog
db.oplog.rs.find().sort({ $natural: -1 }).limit(5)
```

**Performance Optimization:**

**1. Connection Pooling:**
```javascript
// Connection pool configuration
const MongoClient = require('mongodb').MongoClient
const client = new MongoClient(uri, {
  maxPoolSize: 10,        // Maximum connections
  minPoolSize: 5,         // Minimum connections
  maxIdleTimeMS: 30000,   // Close connections after 30 seconds
  serverSelectionTimeoutMS: 5000,
  socketTimeoutMS: 45000,
})
```

**2. Aggregation Pipeline:**
```javascript
// Efficient aggregation for complex queries
db.orders.aggregate([
  { $match: { "status": "completed" } },
  { $group: { 
      "_id": "$customer_id", 
      "total": { $sum: "$amount" },
      "count": { $sum: 1 }
    }
  },
  { $sort: { "total": -1 } },
  { $limit: 10 }
])
```

**3. Sharding for Scale:**
```javascript
// Horizontal scaling through sharding
sh.enableSharding("mydb")
sh.shardCollection("mydb.users", { "user_id": "hashed" })

// Shard key selection criteria:
// - High cardinality
// - Even distribution
// - Query isolation
// - Monotonic vs non-monotonic
```

**Memory Management:**
```javascript
// WiredTiger cache configuration
storage:
  wiredTiger:
    engineConfig:
      cacheSizeGB: 4  // 50% of RAM by default
      
// Memory usage monitoring
db.runCommand({ "serverStatus": 1 }).wiredTiger.cache
```

**Consistency and Durability:**
```javascript
// Journal for durability
storage:
  journal:
    enabled: true
    commitIntervalMs: 100  // Journal commit interval

// Replica set for high availability
rs.status()  // Check replica set status
rs.conf()    // View replica set configuration
```

---

### ❓ Question 43. CAP theorem, how does it work in MongoDB
#### Answer: 
CAP theorem states that in a distributed system, you can only guarantee two out of three properties: Consistency, Availability, and Partition tolerance. MongoDB's design choices reflect specific trade-offs within this framework.

**CAP Theorem Fundamentals:**

**1. Consistency (C):**
- All nodes see the same data simultaneously
- Every read receives the most recent write
- Strong consistency vs eventual consistency

**2. Availability (A):**
- System remains operational and responsive
- Every request receives a response (success or failure)
- No single point of failure

**3. Partition Tolerance (P):**
- System continues operating despite network failures
- Communication breakdown between nodes
- Essential for distributed systems

**MongoDB's CAP Trade-offs:**

**1. MongoDB's Default Choice: CP (Consistency + Partition Tolerance)**
```javascript
// MongoDB prioritizes consistency over availability
// During network partition, minority nodes become read-only

// Replica set behavior during partition:
// - Primary in majority partition: Continues read/write operations
// - Primary in minority partition: Steps down, becomes read-only
// - Secondaries in minority: Become read-only

// Example: 3-node replica set with network partition
// Partition 1: Primary + 1 Secondary (majority) - Continues operations
// Partition 2: 1 Secondary (minority) - Becomes read-only
```

**2. Consistency Mechanisms:**
```javascript
// Write Concerns for Consistency
// Strong consistency: Wait for majority acknowledgment
db.users.insertOne(
  { "name": "John", "email": "john@example.com" },
  { writeConcern: { w: "majority", j: true } }
)

// Read Concerns for Consistency
// Read from majority-committed data
db.users.find({ "status": "active" }).readConcern("majority")

// Linearizable reads (strongest consistency)
db.users.findOne({ "_id": userId }, { readConcern: "linearizable" })
```

**3. Availability Configurations:**
```javascript
// Tuning for higher availability (trades some consistency)
// Read from secondaries to distribute load
db.users.find().readPref("secondaryPreferred")

// Lower write concern for faster writes (less consistency guarantee)
db.logs.insertOne(
  { "timestamp": new Date(), "message": "Log entry" },
  { writeConcern: { w: 1, j: false } }
)

// Replica set with automatic failover
rs.initiate({
  "_id": "myReplicaSet",
  "members": [
    { "_id": 0, "host": "node1:27017", "priority": 2 },
    { "_id": 1, "host": "node2:27017", "priority": 1 },
    { "_id": 2, "host": "node3:27017", "priority": 1 }
  ]
})
```

**MongoDB's Partition Tolerance:**

**1. Network Partition Scenarios:**
```javascript
// Scenario 1: Majority partition retains primary
// Nodes: A (Primary), B (Secondary), C (Secondary)
// Partition: [A, B] vs [C]
// Result: A remains primary, C becomes read-only

// Scenario 2: Minority partition with primary
// Nodes: A (Primary), B (Secondary), C (Secondary)
// Partition: [A] vs [B, C]
// Result: A steps down, B or C becomes new primary

// Checking replica set status during partition
rs.status()
rs.isMaster()  // Check primary/secondary status
```

**2. Election Process:**
```javascript
// Primary election during partition
// Requirements for becoming primary:
// 1. Must be part of majority
// 2. Must have highest priority among majority members
// 3. Must have most recent oplog entries

// Election configuration
rs.reconfig({
  "_id": "myReplicaSet",
  "members": [
    { "_id": 0, "host": "node1:27017", "priority": 2 },
    { "_id": 1, "host": "node2:27017", "priority": 1 },
    { "_id": 2, "host": "node3:27017", "priority": 0, "votes": 1 }
  ]
})
```

**Practical CAP Implementations:**

**1. Strong Consistency Setup:**
```javascript
// Maximum consistency configuration
const strongConsistencyConfig = {
  // Write concern: Wait for majority
  writeConcern: { w: "majority", j: true, wtimeout: 5000 },
  
  // Read concern: Read majority-committed data
  readConcern: "majority",
  
  // Read preference: Primary only
  readPreference: "primary"
}

// Usage example
db.criticalData.insertOne(
  { "transactionId": "tx123", "amount": 1000 },
  { writeConcern: strongConsistencyConfig.writeConcern }
)

db.criticalData.find({ "transactionId": "tx123" })
  .readConcern("majority")
  .readPref("primary")
```

**2. High Availability Setup:**
```javascript
// Availability-focused configuration
const highAvailabilityConfig = {
  // Write concern: Acknowledge from primary only
  writeConcern: { w: 1, j: false },
  
  // Read concern: Local data (fastest)
  readConcern: "local",
  
  // Read preference: Allow secondary reads
  readPreference: "secondaryPreferred"
}

// Usage for non-critical data
db.logs.insertOne(
  { "timestamp": new Date(), "level": "info", "message": "User login" },
  { writeConcern: highAvailabilityConfig.writeConcern }
)

db.analytics.find({ "date": { $gte: new Date() } })
  .readConcern("local")
  .readPref("secondaryPreferred")
```

**3. Balanced Approach:**
```javascript
// Application-level CAP handling
class DatabaseService {
  constructor() {
    this.criticalOperations = {
      writeConcern: { w: "majority", j: true },
      readConcern: "majority",
      readPreference: "primary"
    }
    
    this.normalOperations = {
      writeConcern: { w: 1, j: true },
      readConcern: "local",
      readPreference: "primaryPreferred"
    }
    
    this.analyticsOperations = {
      writeConcern: { w: 1, j: false },
      readConcern: "local",
      readPreference: "secondaryPreferred"
    }
  }
  
  async transferMoney(fromAccount, toAccount, amount) {
    // Critical operation - Strong consistency
    const session = db.getMongo().startSession()
    session.startTransaction({
      readConcern: this.criticalOperations.readConcern,
      writeConcern: this.criticalOperations.writeConcern
    })
    
    try {
      await db.accounts.updateOne(
        { "_id": fromAccount },
        { $inc: { "balance": -amount } },
        { session: session }
      )
      
      await db.accounts.updateOne(
        { "_id": toAccount },
        { $inc: { "balance": amount } },
        { session: session }
      )
      
      await session.commitTransaction()
    } catch (error) {
      await session.abortTransaction()
      throw error
    } finally {
      session.endSession()
    }
  }
  
  async logUserAction(userId, action) {
    // Non-critical operation - Favor availability
    return db.userLogs.insertOne(
      { "userId": userId, "action": action, "timestamp": new Date() },
      { writeConcern: this.analyticsOperations.writeConcern }
    )
  }
}
```

**CAP Theorem Trade-offs in Different Scenarios:**

**1. Financial System (CP - Consistency + Partition Tolerance):**
```javascript
// Banking application - Cannot tolerate inconsistency
const bankingConfig = {
  writeConcern: { w: "majority", j: true, wtimeout: 5000 },
  readConcern: "majority",
  readPreference: "primary"
}

// If partition occurs, system becomes unavailable for minority
// but maintains consistency for majority
```

**2. Social Media Feed (AP - Availability + Partition Tolerance):**
```javascript
// Social media - Can tolerate some inconsistency
const socialMediaConfig = {
  writeConcern: { w: 1, j: false },
  readConcern: "local",
  readPreference: "secondaryPreferred"
}

// During partition, users can still read/write
// but may see slightly stale data
```

**3. Real-time Analytics (Eventual Consistency):**
```javascript
// Analytics system - Uses eventual consistency
const analyticsConfig = {
  writeConcern: { w: 1, j: false },
  readConcern: "available",
  readPreference: "nearest"
}

// Optimizes for performance and availability
// Accepts that data may be eventually consistent
```

**Monitoring CAP Behavior:**
```javascript
// Monitor replica set health
rs.status()

// Check oplog lag
db.getReplicationInfo()

// Monitor write concern failures
db.runCommand({ "serverStatus": 1 }).opcounters

// Check election statistics
db.runCommand({ "replSetGetStatus": 1 }).electionId
```

**MongoDB's CAP Evolution:**
- **Pre-3.2**: Primarily CP system
- **3.2+**: Introduced configurable read/write concerns
- **4.0+**: Multi-document transactions (stronger consistency)
- **4.2+**: Distributed transactions across shards
- **Current**: Flexible CAP positioning based on application needs

MongoDB's strength lies in allowing developers to choose their position on the CAP spectrum per operation, rather than forcing a single system-wide choice.

---
### ❓ Question 44. Predicate Pushdown in Spark
#### Answer: 
Predicate pushdown is an optimization technique where filter conditions are pushed down as close to the data source as possible, reducing the amount of data that needs to be processed. In Spark, this happens automatically through the Catalyst optimizer. For example, if you have a filter condition on a column, Spark will push this filter to the data source (like Parquet files or databases) so that only relevant data is read, minimizing I/O and improving performance.

---
### ❓ Question 45. Advantages of functional programming in Scala
#### Answer:
- **Immutability**: Objects cannot be modified after creation, reducing bugs and making code thread-safe
- **Higher-order functions**: Functions can be passed as parameters and returned as values
- **Pattern matching**: Powerful way to destructure and match data
- **Type safety**: Strong static typing catches errors at compile time
- **Concurrency**: Immutable data structures make concurrent programming safer
- **Composability**: Functions can be easily combined to build complex operations
- **Lazy evaluation**: Computations are deferred until results are needed

---
### ❓ Question 46. Optimization techniques in Spark
#### Answer:
- **Caching/Persistence**: Cache frequently accessed RDDs/DataFrames in memory
- **Partitioning**: Optimize data partitioning to reduce shuffles
- **Broadcast variables**: Share large read-only data across executors
- **Coalesce/Repartition**: Optimize number of partitions
- **Avoid wide transformations**: Minimize shuffles by using narrow transformations
- **Use DataFrames/Datasets**: Leverage Catalyst optimizer instead of RDDs
- **Predicate pushdown**: Apply filters early in the pipeline
- **Column pruning**: Select only required columns
- **Bucketing**: Pre-partition data for joins
- **Tune serialization**: Use Kryo serializer for better performance

---
### ❓ Question 47. What is DAG in Spark
#### Answer:
DAG (Directed Acyclic Graph) is a logical execution plan in Spark that represents the sequence of computations to be performed on RDDs/DataFrames. It's created when actions are called and consists of stages separated by shuffle operations. Each stage contains a series of tasks that can be executed in parallel. The DAG scheduler optimizes the execution plan by combining narrow transformations into stages and scheduling them efficiently across the cluster.

---
### ❓ Question 48. Broadcast Variables in Spark
#### Answer:
Broadcast variables are read-only variables that are cached and distributed to all executor nodes in a Spark cluster. They're used to efficiently share large datasets (like lookup tables or configuration data) across all tasks without sending the data over the network repeatedly. This reduces network overhead and improves performance when the same data needs to be accessed by multiple tasks.

Example:
```scala
val broadcastVar = spark.sparkContext.broadcast(Array(1, 2, 3))
// Use broadcastVar.value in transformations
```

---
### ❓ Question 49. How Spark processing happens (driver and executors creation)
#### Answer:
1. **Application Submission**: User submits Spark application using spark-submit
2. **Driver Creation**: Cluster manager creates the driver process which contains the SparkContext
3. **Resource Request**: Driver requests resources (CPU, memory) from cluster manager
4. **Executor Launch**: Cluster manager launches executors on worker nodes based on resource availability
5. **Task Distribution**: Driver creates DAG, divides it into stages and tasks, then sends tasks to executors
6. **Execution**: Executors run tasks in parallel and send results back to driver
7. **Cleanup**: After completion, driver terminates executors and releases resources

The cluster manager (YARN, Mesos, or Standalone) acts as the intermediary between driver and executors for resource allocation and management.

---
### ❓ Question 50. Python Questions - List/Tuple/Dictionary manipulation operations
#### Answer:
Common output prediction questions:
```python
# List operations
lst = [1, 2, 3, 4, 5]
print(lst[1:4])  # Output: [2, 3, 4]
print(lst[::-1])  # Output: [5, 4, 3, 2, 1]

# Dictionary operations
d = {'a': 1, 'b': 2}
d.update({'c': 3})
print(d)  # Output: {'a': 1, 'b': 2, 'c': 3}

# Tuple operations
t = (1, 2, 3)
print(t + (4, 5))  # Output: (1, 2, 3, 4, 5)

# List comprehension
result = [x*2 for x in range(5) if x % 2 == 0]
print(result)  # Output: [0, 4, 8]
```

---
### ❓ Question 51. PySpark Query - Average revenue per product ID from 3 tables
#### Answer:
```python
# Assuming tables: products, orders, order_items
# Join tables and calculate average revenue per product
result = (orders
    .join(order_items, orders.order_id == order_items.order_id)
    .join(products, order_items.product_id == products.product_id)
    .groupBy("product_id")
    .agg(avg(col("quantity") * col("price")).alias("avg_revenue"))
    .orderBy("product_id")
)

# Alternative with window function
from pyspark.sql.window import Window
windowSpec = Window.partitionBy("product_id")
result = (joined_df
    .withColumn("avg_revenue", avg(col("revenue")).over(windowSpec))
    .select("product_id", "avg_revenue")
    .distinct()
)
```

---
### ❓ Question 52. Difference between Group By and Window Functions
#### Answer:
**Group By:**
- Aggregates data and reduces number of rows
- Returns one row per group
- Cannot access individual row details after grouping
- Example: `SELECT dept, AVG(salary) FROM emp GROUP BY dept`

**Window Functions:**
- Performs calculations across related rows without reducing row count
- Returns same number of rows as input
- Can access both aggregate and row-level data
- Supports ranking, running totals, lead/lag operations
- Example: `SELECT name, salary, AVG(salary) OVER (PARTITION BY dept) FROM emp`

Window functions are more flexible for analytics queries where you need both detail and aggregate information.

---
### ❓ Question 53. Spark-Submit configurations for better performance
#### Answer:
```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --driver-memory 2g \
  --driver-cores 2 \
  --executor-memory 4g \
  --executor-cores 4 \
  --num-executors 10 \
  --conf spark.sql.adaptive.enabled=true \
  --conf spark.sql.adaptive.coalescePartitions.enabled=true \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.sql.execution.arrow.pyspark.enabled=true \
  --conf spark.dynamicAllocation.enabled=true \
  --conf spark.dynamicAllocation.minExecutors=2 \
  --conf spark.dynamicAllocation.maxExecutors=20 \
  my_app.py
```

Key configurations: adaptive query execution, dynamic allocation, Kryo serialization, appropriate memory and core allocation based on cluster resources.

---
### ❓ Question 54. OOPS principles in Java
#### Answer:
**1. Encapsulation**: Bundling data and methods together, hiding internal implementation details using private access modifiers and providing public interfaces.

**2. Inheritance**: Creating new classes based on existing classes, promoting code reusability. Uses `extends` keyword.

**3. Polymorphism**: Same interface, different implementations. Achieved through method overriding (runtime) and method overloading (compile-time).

**4. Abstraction**: Hiding complex implementation details and showing only essential features. Implemented using abstract classes and interfaces.

Additional principles:
- **Association**: Relationship between objects
- **Composition**: "Has-a" relationship where one object contains another
- **Aggregation**: Weak "has-a" relationship where objects can exist independently

---
### ❓ Question 55. Design Patterns
#### Answer:
**Creational Patterns:**
- **Singleton**: Ensures only one instance exists (e.g., database connection)
- **Factory**: Creates objects without specifying exact classes
- **Builder**: Constructs complex objects step by step

**Structural Patterns:**
- **Adapter**: Allows incompatible interfaces to work together
- **Decorator**: Adds new functionality to objects dynamically
- **Facade**: Provides simplified interface to complex subsystem

**Behavioral Patterns:**
- **Observer**: Defines one-to-many dependency between objects
- **Strategy**: Encapsulates algorithms and makes them interchangeable
- **Command**: Encapsulates requests as objects

**Common in Big Data:**
- **MapReduce Pattern**: Parallel processing pattern
- **Pipeline Pattern**: Sequential data processing stages
- **Repository Pattern**: Abstraction layer for data access

---

### ❓ Question 56. Brief about current/last Project, explain the architecture

#### Answer:
This is a personal question that requires you to discuss your specific project experience. Structure your answer to include:
- Project overview and business problem solved
- Technologies used (e.g., Spark, Hadoop, Python, cloud platforms)
- Architecture components (data ingestion, processing, storage, visualization)
- Your specific role and contributions
- Key challenges faced and how you resolved them
- Impact/results achieved

Example structure: "I worked on a real-time data processing pipeline for customer behavior analytics. The architecture included Kafka for streaming data ingestion, Spark for real-time processing, HDFS for storage, and Tableau for visualization. My role involved designing the ETL pipeline and optimizing Spark jobs for performance."

---

### ❓ Question 57. Difference b/w mutable & immutable types in Python with eg.

#### Answer:
**Mutable types** can be changed after creation:
- Lists, dictionaries, sets, user-defined classes
- Example:
```python
my_list = [1, 2, 3]
my_list.append(4)  # Modifies original list
print(my_list)  # [1, 2, 3, 4]
```

**Immutable types** cannot be changed after creation:
- Integers, floats, strings, tuples, frozensets
- Example:
```python
my_string = "Hello"
my_string.upper()  # Returns new string, doesn't modify original
print(my_string)  # Still "Hello"
new_string = my_string.upper()  # Need to assign to new variable
```

**Key implications:**
- Mutable objects can be modified in-place
- Immutable objects require creating new objects for changes
- This affects memory usage and performance considerations

---

### ❓ Question 58. Difference b/w Spark & Hadoop processing

#### Answer:
| Aspect | Spark | Hadoop (MapReduce) |
|--------|-------|-------------------|
| **Processing Model** | In-memory processing | Disk-based processing |
| **Speed** | 100x faster for iterative algorithms | Slower due to disk I/O |
| **Data Processing** | Batch, streaming, interactive | Primarily batch processing |
| **Memory Usage** | Stores intermediate results in memory | Writes to disk between stages |
| **Fault Tolerance** | RDD lineage-based recovery | Replication-based |
| **Ease of Use** | High-level APIs (SQL, DataFrame) | Lower-level, more verbose |
| **Real-time Processing** | Native support via Spark Streaming | Requires additional tools |
| **Machine Learning** | Built-in MLlib | Requires external libraries |

**Use Cases:**
- Spark: Real-time analytics, machine learning, iterative algorithms
- Hadoop: Large-scale batch processing, data archival, cost-sensitive scenarios

---

### ❓ Question 59. How will you decrease o/p no. of files to 1 in spark

#### Answer:
Several approaches to reduce output files to 1:

**1. Coalesce (Recommended):**
```python
df.coalesce(1).write.mode("overwrite").parquet("output_path")
```

**2. Repartition:**
```python
df.repartition(1).write.mode("overwrite").parquet("output_path")
```

**3. Using repartition with specific column:**
```python
df.repartition(1, "column_name").write.mode("overwrite").parquet("output_path")
```

**Key Differences:**
- `coalesce(1)`: More efficient as it minimizes data movement
- `repartition(1)`: Triggers full shuffle, less efficient
- `coalesce` is preferred for reducing partitions
- `repartition` is better for increasing partitions or redistributing data

**Note:** Single file output may create performance bottlenecks for large datasets as all data flows through one executor.

---

### ❓ Question 60. How will you handle a huge file in Spark for processing

#### Answer:
**1. Optimal Partitioning:**
```python
# Control partition size
spark.conf.set("spark.sql.files.maxPartitionBytes", "134217728")  # 128MB
df = spark.read.option("multiline", "true").json("huge_file.json")
```

**2. Increase Executor Resources:**
```python
spark.conf.set("spark.executor.memory", "8g")
spark.conf.set("spark.executor.cores", "4")
spark.conf.set("spark.sql.adaptive.enabled", "true")
```

**3. Use Appropriate File Formats:**
- Parquet for columnar data
- Avro for row-based data
- Consider compression (snappy, gzip)

**4. Implement Lazy Evaluation:**
```python
# Chain transformations without immediate execution
df_filtered = df.filter(condition).select(columns)
df_aggregated = df_filtered.groupBy("column").agg(functions)
df_aggregated.write.mode("overwrite").parquet("output")
```

**5. Memory Management:**
- Use `persist()` or `cache()` strategically
- Choose appropriate storage levels
- Monitor memory usage via Spark UI

**6. Split Processing:**
- Process file in chunks if possible
- Use date/time partitioning for incremental processing

---

### ❓ Question 61. Have you worked on Airflow and No SQL databases

#### Answer:
This is an experience-based question. Structure your answer as:

**Apache Airflow Experience:**
- Used for workflow orchestration and scheduling
- Created DAGs for ETL pipelines
- Implemented task dependencies and error handling
- Used operators like BashOperator, PythonOperator, SparkSubmitOperator
- Configured connections and variables
- Monitored workflow execution via Airflow UI

**NoSQL Database Experience:**
- **MongoDB**: Document-based storage for semi-structured data
- **Cassandra**: Column-family database for time-series data
- **Redis**: In-memory key-value store for caching
- **HBase**: Wide-column store on Hadoop ecosystem

**Example:** "I've used Airflow to orchestrate daily ETL processes, creating DAGs that extract data from MySQL, process it using Spark, and load results into MongoDB. I've also worked with Redis for caching frequently accessed data to improve application performance."

---

### ❓ Question 62. Were your end-user Business users or Technical users

#### Answer:
This is a project-specific question. Structure your response:

**Business Users:**
- Non-technical stakeholders (analysts, managers, executives)
- Required user-friendly dashboards and reports
- Focused on business metrics and KPIs
- Needed self-service analytics capabilities
- Communication required translating technical concepts to business language

**Technical Users:**
- Data scientists, analysts, developers
- Comfortable with SQL queries and technical interfaces
- Required detailed data access and flexibility
- Could handle complex data structures and APIs
- Communication could be more technical in nature

**Example Response:** "My primary end-users were business analysts and marketing managers (business users). I designed Tableau dashboards showing customer acquisition metrics and campaign performance. This required creating intuitive visualizations and ensuring data accuracy, as they needed to make strategic decisions based on the insights. I also provided data dictionary documentation and conducted training sessions to help them understand the metrics."

---

### ❓ Question 63. If you persist the data will it improve performance in Spark

#### Answer:
**Yes, persisting data can significantly improve performance in specific scenarios:**

**When to Use Persist/Cache:**
1. **Multiple Actions on Same DataFrame:**
```python
df.cache()
df.count()  # First action triggers computation and caching
df.show()   # Uses cached data, much faster
```

2. **Iterative Algorithms:**
```python
# Machine learning algorithms that iterate over same dataset
df.persist(StorageLevel.MEMORY_AND_DISK)
for i in range(iterations):
    model.fit(df)  # Reuses cached data
```

3. **Complex Transformations:**
```python
expensive_df = df.join(other_df).filter(complex_condition).persist()
result1 = expensive_df.groupBy("col1").count()
result2 = expensive_df.groupBy("col2").sum("value")
```

**Storage Levels:**
- `MEMORY_ONLY`: Fastest but limited by available memory
- `MEMORY_AND_DISK`: Spills to disk when memory is full
- `DISK_ONLY`: Slower but handles large datasets

**When NOT to Use:**
- Single action on DataFrame
- Linear transformations with no reuse
- Memory-constrained environments

**Best Practices:**
- Use `unpersist()` to free memory when done
- Monitor storage via Spark UI
- Choose appropriate storage level based on data size and memory availability

---

### ❓ Question 64. Have you used Regex in Python, what kind of?

#### Answer:
**Yes, I've used regex extensively in Python for various data processing tasks:**

**Common Use Cases:**

**1. Data Validation:**
```python
import re
# Email validation
email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
re.match(email_pattern, email)

# Phone number validation
phone_pattern = r'^\+?1?-?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})$'
```

**2. Data Extraction:**
```python
# Extract dates from text
date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
dates = re.findall(date_pattern, text)

# Extract URLs
url_pattern = r'https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?'
```

**3. Data Cleaning:**
```python
# Remove special characters
clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

# Normalize whitespace
normalized = re.sub(r'\s+', ' ', text).strip()
```

**4. Log File Processing:**
```python
# Parse Apache log files
log_pattern = r'(\d+\.\d+\.\d+\.\d+) - - \[(.*?)\] "(.*?)" (\d+) (\d+)'
match = re.search(log_pattern, log_line)
```

**5. String Replacement:**
```python
# Replace multiple patterns
text = re.sub(r'(Mr|Mrs|Ms|Dr)\.', r'\1', text)
```

---

### ❓ Question 65. Difference between RDD, Dataframe and Dataset. Which of them has type safety feature.

#### Answer:
| Feature | RDD | DataFrame | Dataset |
|---------|-----|-----------|---------|
| **Type Safety** | No | No | **Yes** |
| **Performance** | Lowest | High | Highest |
| **Optimization** | No Catalyst | Catalyst Optimizer | Catalyst Optimizer |
| **API Level** | Low-level | High-level | High-level |
| **Schema** | No schema | Has schema | Has schema |
| **Serialization** | Java serialization | Tungsten binary | Tungsten binary |
| **Language Support** | Scala, Java, Python | All languages | Scala, Java only |

**Detailed Comparison:**

**RDD (Resilient Distributed Dataset):**
- Fundamental data structure in Spark
- Immutable, distributed collection
- No built-in optimization
- Type safety only at compile time for transformation logic

**DataFrame:**
- Built on top of RDD
- Structured data with named columns
- Catalyst optimizer for query optimization
- No compile-time type safety for data

**Dataset:**
- Combines benefits of RDD and DataFrame
- **Strong type safety at compile time**
- Best performance due to optimizations
- Only available in Scala and Java

**Example of Type Safety:**
```scala
// Dataset - Type safe
case class Person(name: String, age: Int)
val ds: Dataset[Person] = spark.read.json("people.json").as[Person]
ds.filter(_.age > 18)  // Compile-time error if 'age' doesn't exist

// DataFrame - No type safety
val df = spark.read.json("people.json")
df.filter($"age" > 18)  // Runtime error if 'age' doesn't exist
```

**Answer:** **Dataset has type safety feature**, providing compile-time type checking while maintaining high performance through Catalyst optimizer.

---

### ❓ Question 66. Explain immutability in Spark dataframe. If we dont create a new dataframe from an existing one, and add a new column to the existing dataframe and use show(), will we be able to see the new column? If now, i try to do dataframe.printSchema(), will it show the new column?

#### Answer:
**Spark DataFrame Immutability:**
DataFrames in Spark are immutable, meaning once created, they cannot be modified. Any transformation creates a new DataFrame.

**Practical Example:**
```python
# Original DataFrame
df = spark.createDataFrame([(1, "Alice"), (2, "Bob")], ["id", "name"])

# Adding column without assigning to new variable
df.withColumn("age", lit(25))  # This creates a new DataFrame but doesn't store it

# Using show() on original DataFrame
df.show()  # Will NOT show the new 'age' column
# Output: Only 'id' and 'name' columns

# Using printSchema() on original DataFrame
df.printSchema()  # Will NOT show the new 'age' column
# Output: Only shows schema for 'id' and 'name'
```

**Correct Approach:**
```python
# Must assign to new variable or overwrite existing
df_with_age = df.withColumn("age", lit(25))
df_with_age.show()  # Will show the new 'age' column

# Or overwrite existing
df = df.withColumn("age", lit(25))
df.show()  # Now shows the new 'age' column
```

**Key Points:**
- Transformations like `withColumn()`, `filter()`, `select()` return new DataFrames
- Original DataFrame remains unchanged
- Must assign transformation result to see changes
- This immutability enables fault tolerance through lineage tracking
- Lazy evaluation means transformations are not executed until an action is called

**Answer to Questions:**
- **show():** No, will not see the new column
- **printSchema():** No, will not show the new column
Both will only display the original DataFrame structure since the transformation result wasn't assigned.

---

### ❓ Question 67. Explain OOM issues in Spark

#### Answer:
**Out of Memory (OOM) Issues in Spark:**

**Common Causes:**

**1. Driver OOM:**
- `collect()` on large datasets
- Broadcasting large variables
- Accumulating large amounts of data

**2. Executor OOM:**
- Large partitions
- Skewed data distribution
- Memory-intensive operations (joins, aggregations)
- Too many cached DataFrames

**3. Specific Scenarios:**
```python
# Problematic operations
df.collect()  # Brings all data to driver
df.toPandas()  # Converts entire DataFrame to Pandas

# Large broadcasts
large_dict = {...}  # Very large dictionary
broadcast_var = spark.sparkContext.broadcast(large_dict)
```

**Solutions:**

**1. Optimize Partitioning:**
```python
# Repartition to balance data
df.repartition(200, "key_column")

# Use coalesce for reducing partitions
df.coalesce(50)
```

**2. Increase Memory Allocation:**
```python
spark.conf.set("spark.executor.memory", "8g")
spark.conf.set("spark.driver.memory", "4g")
spark.conf.set("spark.executor.memoryFraction", "0.8")
```

**3. Handle Skewed Data:**
```python
# Add salt to skewed keys
df_salted = df.withColumn("salted_key", 
    concat(col("key"), lit("_"), (rand() * 100).cast("int")))
```

**4. Optimize Caching:**
```python
# Use appropriate storage levels
df.persist(StorageLevel.MEMORY_AND_DISK_SER)

# Unpersist when done
df.unpersist()
```

**5. Use Alternative Approaches:**
```python
# Instead of collect(), use sampling
df.sample(0.01).collect()

# Use write operations instead of collect
df.write.mode("overwrite").parquet("output_path")
```

**6. Monitor and Tune:**
- Use Spark UI to monitor memory usage
- Enable adaptive query execution
- Tune garbage collection settings
- Consider using off-heap memory

---

### ❓ Question 68. If some join or heavy operation is being performed on huge datasets in spark and spark is taking too long to execute, how will you find the cause? and what can be the possible reasons for it?

#### Answer:
**Debugging Approach:**

**1. Use Spark UI for Analysis:**
- Check Stages tab for bottlenecks
- Analyze task duration and data distribution
- Monitor executor utilization
- Check for failed/retried tasks

**2. Enable Detailed Logging:**
```python
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
```

**3. Profile Data Distribution:**
```python
# Check partition sizes
df.rdd.glom().map(len).collect()

# Analyze data skew
df.groupBy("join_key").count().orderBy(desc("count")).show()
```

**Possible Reasons and Solutions:**

**1. Data Skew:**
- **Problem:** Uneven distribution of data across partitions
- **Solution:** Salt keys, use broadcast joins, or custom partitioning
```python
# Broadcast join for small tables
small_df.hint("broadcast").join(large_df, "key")
```

**2. Inappropriate Join Strategy:**
- **Problem:** Hash join on large datasets without proper partitioning
- **Solution:** Use sort-merge join or broadcast join
```python
# Force sort-merge join
df1.hint("merge").join(df2, "key")
```

**3. Memory Issues:**
- **Problem:** Insufficient memory causing spills to disk
- **Solution:** Increase executor memory, optimize partitioning
```python
spark.conf.set("spark.executor.memory", "8g")
spark.conf.set("spark.sql.shuffle.partitions", "400")
```

**4. Poor Partitioning:**
- **Problem:** Too few or too many partitions
- **Solution:** Optimal partitioning based on data size
```python
# Repartition before join
df1.repartition(200, "join_key").join(df2.repartition(200, "join_key"), "join_key")
```

**5. Network Bottlenecks:**
- **Problem:** Excessive data shuffling
- **Solution:** Pre-partition data, use bucketing
```python
# Bucketing to avoid shuffle
df.write.bucketBy(10, "join_key").saveAsTable("bucketed_table")
```

**6. Resource Constraints:**
- **Problem:** Insufficient cluster resources
- **Solution:** Scale cluster, optimize resource allocation

**Optimization Strategies:**
```python
# Enable adaptive query execution
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

# Optimize joins
df1.join(broadcast(df2), "key")  # Broadcast smaller table
```

---

### ❓ Question 69. Reverse a string in Python - Input: S = "i like this program very much", i) reverse the characters and words. ii) reverse only the characters and not the position of word.

#### Answer:
**Input:** `S = "i like this program very much"`

**Solution i) Reverse the characters and words:**
```python
def reverse_words_and_characters(s):
    # Split into words, reverse each word, then reverse the list
    words = s.split()
    reversed_words = [word[::-1] for word in words]
    return ' '.join(reversed_words[::-1])

# Alternative one-liner
def reverse_words_and_characters_oneliner(s):
    return ' '.join(word[::-1] for word in s.split()[::-1])

s = "i like this program very much"
result1 = reverse_words_and_characters(s)
print(result1)  # Output: "hcum yrev margorp siht ekil i"
```

**Solution ii) Reverse only the characters and not the position of words:**
```python
def reverse_characters_only(s):
    # Split into words, reverse each word, keep word positions
    words = s.split()
    reversed_words = [word[::-1] for word in words]
    return ' '.join(reversed_words)

s = "i like this program very much"
result2 = reverse_characters_only(s)
print(result2)  # Output: "i ekil siht margorp yrev hcum"
```

**Complete Solution:**
```python
def string_reversal(s):
    # i) Reverse characters and words
    words = s.split()
    reversed_chars_and_words = ' '.join(word[::-1] for word in words[::-1])
    
    # ii) Reverse only characters, keep word positions
    reversed_chars_only = ' '.join(word[::-1] for word in words)
    
    return reversed_chars_and_words, reversed_chars_only

s = "i like this program very much"
result1, result2 = string_reversal(s)
print(f"Original: {s}")
print(f"Reversed chars and words: {result1}")
print(f"Reversed chars only: {result2}")
```

**Output:**
```
Original: i like this program very much
Reversed chars and words: hcum yrev margorp siht ekil i
Reversed chars only: i ekil siht margorp yrev hcum
```

---

### ❓ Question 70. What is the major difference between SQL and NOSQL databases

#### Answer:
| Aspect | SQL Databases | NoSQL Databases |
|--------|---------------|-----------------|
| **Data Model** | Structured, relational tables | Various (document, key-value, column-family, graph) |
| **Schema** | Fixed schema, predefined structure | Flexible/dynamic schema |
| **ACID Properties** | Strong ACID compliance | Eventual consistency, BASE properties |
| **Scalability** | Vertical scaling (scale up) | Horizontal scaling (scale out) |
| **Query Language** | Standardized SQL | Varies by database type |
| **Data Relationships** | Strong relationships via foreign keys | Weak or no relationships |
| **Consistency** | Strong consistency | Eventual consistency |
| **Use Cases** | Complex queries, transactions, reporting | Big data, real-time applications, flexible data |

**Detailed Differences:**

**SQL Databases (RDBMS):**
- **Examples:** MySQL, PostgreSQL, Oracle, SQL Server
- **Structure:** Data stored in tables with rows and columns
- **ACID:** Atomicity, Consistency, Isolation, Durability
- **Best for:** Financial systems, e-commerce, traditional applications

**NoSQL Databases:**

**1. Document Stores (MongoDB, CouchDB):**
```json
{
  "name": "John Doe",
  "age": 30,
  "address": {
    "street": "123 Main St",
    "city": "New York"
  }
}
```

**2. Key-Value Stores (Redis, DynamoDB):**
```
Key: "user:123"
Value: {"name": "John", "email": "john@example.com"}
```

**3. Column-Family (Cassandra, HBase):**
- Wide columns, good for time-series data

**4. Graph Databases (Neo4j, Amazon Neptune):**
- Nodes and relationships, good for social networks

**When to Use:**

**SQL:**
- Complex queries and reporting
- Strong consistency requirements
- Well-defined relationships
- ACID transactions needed

**NoSQL:**
- Rapid development with changing requirements
- Horizontal scaling needs
- Big data applications
- Real-time web applications
- Flexible data models

**Example Scenario:**
- **E-commerce checkout:** SQL (need ACID for payments)
- **Product catalog:** NoSQL (flexible product attributes)
- **User sessions:** Key-value NoSQL (fast access)
- **Recommendation engine:** Graph NoSQL (relationship analysis)


