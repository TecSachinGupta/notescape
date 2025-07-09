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
