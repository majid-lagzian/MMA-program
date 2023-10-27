# Databricks notebook source
import os
import sys


import pyspark
from pyspark.ml import PipelineModel
from pyspark.ml.feature import FeatureHasher
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from synapse.ml.train import ComputeModelStatistics
from synapse.ml.lightgbm import LightGBMClassifier

print("System version: {}".format(sys.version))
print("PySpark version: {}".format(pyspark.version.__version__))

# COMMAND ----------

# DBTITLE 1,Load Data
# Load the table
data = spark.sql("select * from default.reviews_train")

#data = data.sample(False, 0.10, seed=0)

#data = data.cache()

print((data.count(), len(data.columns)))

# COMMAND ----------

data.printSchema()

# COMMAND ----------

# The count of each overall rating

from pyspark.sql.functions import col
data.groupBy("overall").count().orderBy(col("overall").asc()).show()

# COMMAND ----------

df = data.na.drop(subset=["reviewText", "label"])
df.show(5)
print((df.count(), len(df.columns)))

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

spark = SparkSession.builder.appName("RowNumberGroupByExample").getOrCreate()

asin_window_spec = Window.partitionBy("asin").orderBy("unixReviewTime")
reviewer_window_spec = Window.partitionBy("reviewerID").orderBy("unixReviewTime")

# Add row numbers within each group
df = df.withColumn("Product Reviews", row_number().over(asin_window_spec))
df = df.withColumn("Customer Reviews", row_number().over(reviewer_window_spec))

# Show the resulting DataFrame
df.show()


# COMMAND ----------

# DBTITLE 1,EDA Stuff!
import re
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import *
from pyspark.sql.types import IntegerType

from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, StringIndexer, SQLTransformer, IndexToString, VectorAssembler, Word2Vec
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, MultilayerPerceptronClassifier
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import udf, col
import pyspark.sql.functions as F



# convert text column to spark nlp document
document_assembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")


# convert document to array of tokens
tokenizer = Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("token")

sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentences")

capital_letter_counter = SQLTransformer(
    statement="SELECT *, (SELECT COUNT(*) FROM (SELECT explode(token.result) AS word FROM __THIS__) WHERE word RLIKE '[A-Z]') AS capital_letters_count FROM __THIS__"
)

# Use SQLTransformer to add a new column with the count of exclamation marks
exclamation_mark_counter = SQLTransformer(
    statement="SELECT *, LENGTH(reviewText) - LENGTH(REGEXP_REPLACE(reviewText, '!', '')) AS exclamation_marks_count FROM __THIS__"
)

# clean tokens 
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")

# remove stopwords
stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

# stems tokens to bring it to root form
stemmer = Stemmer() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("stem")


# Calculate the number of tokens and sentences using SQLTransformer
token_count_sql = SQLTransformer(
    statement="SELECT *, SIZE(token) AS num_tokens FROM __THIS__"
)

cleantoken_count_sql = SQLTransformer(
    statement="SELECT *, SIZE(cleanTokens) AS num__clean_tokens FROM __THIS__"
)

sentence_count_sql = SQLTransformer(
    statement="SELECT *, SIZE(sentences) AS num_sentences FROM __THIS__"
)


transformer_pipeline = Pipeline(
    stages=[document_assembler, 
            tokenizer,
            sentence_detector,
            capital_letter_counter,
            exclamation_mark_counter,
            normalizer,
            stopwords_cleaner, 
            stemmer, 
            token_count_sql,
            cleantoken_count_sql,
            sentence_count_sql])
            

# COMMAND ----------

transformed_df=transformer_pipeline.fit(df).transform(df)
transformed_df.show()

# COMMAND ----------

import re
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import *
from pyspark.sql.types import IntegerType, FloatType

from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, StringIndexer, SQLTransformer, IndexToString, VectorAssembler, Word2Vec
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, MultilayerPerceptronClassifier
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import udf, col
import pyspark.sql.functions as F
from textblob import TextBlob


# convert text column to spark nlp document
document_assembler2 = DocumentAssembler() \
    .setInputCol("summary") \
    .setOutputCol("sum_document")


# convert document to array of tokens
tokenizer2 = Tokenizer() \
  .setInputCols(["sum_document"]) \
  .setOutputCol("sum_token")

# Calculate the number of tokens and sentences using SQLTransformer
summary_count_sql = SQLTransformer(
    statement="SELECT *, SIZE(sum_token) AS num_summary FROM __THIS__"
)

# Use SQLTransformer to add a new column with the count of exclamation marks
summary_exclamation_counter = SQLTransformer(
    statement="SELECT *, LENGTH(summary) - LENGTH(REGEXP_REPLACE(summary, '!', '')) AS summary_exclamation FROM __THIS__"
)

transformer2_pipeline = Pipeline(
    stages=[document_assembler2,
            tokenizer2,
            summary_count_sql,
            summary_exclamation_counter])

transformed_df=transformer2_pipeline.fit(df).transform(df)
transformed_df.show()
            

# COMMAND ----------

from pyspark.sql.functions import col, min, max, mean

# Specify the columns for which you want to calculate statistics
columns_to_aggregate = ["num_tokens", "num__clean_tokens", "num_sentences"]

# Define the aggregation functions for each column
agg_exprs = [min(col(col_name)).alias(f"min_{col_name}") for col_name in columns_to_aggregate] + \
            [max(col(col_name)).alias(f"max_{col_name}") for col_name in columns_to_aggregate] + \
            [mean(col(col_name)).alias(f"mean_{col_name}") for col_name in columns_to_aggregate]

# Perform the grouping and aggregation
result = transformed_df.groupBy("label").agg(*agg_exprs)

# Show the result
result.show()

# COMMAND ----------

from pyspark.sql.functions import col, min, max, mean

# Specify the columns for which you want to calculate statistics
columns_to_aggregate = ["num_summary", "exclamation_marks_count"]

# Define the aggregation functions for each column
agg_exprs = [min(col(col_name)).alias(f"min_{col_name}") for col_name in columns_to_aggregate] + \
            [max(col(col_name)).alias(f"max_{col_name}") for col_name in columns_to_aggregate] + \
            [mean(col(col_name)).alias(f"mean_{col_name}") for col_name in columns_to_aggregate]

# Perform the grouping and aggregation
result2 = transformed_df.groupBy("label").agg(*agg_exprs)

# Show the result
result2.show()

# COMMAND ----------

from pyspark.sql.functions import col, min, max, mean

# Specify the columns for which you want to calculate statistics
columns_to_aggregate = ["summary_exclamation"]

# Define the aggregation functions for each column
agg_exprs = [min(col(col_name)).alias(f"min_{col_name}") for col_name in columns_to_aggregate] + \
            [max(col(col_name)).alias(f"max_{col_name}") for col_name in columns_to_aggregate] + \
            [mean(col(col_name)).alias(f"mean_{col_name}") for col_name in columns_to_aggregate]

# Perform the grouping and aggregation
result = transformed_df.groupBy("label").agg(*agg_exprs)

# Show the result
result.show()

# COMMAND ----------

# DBTITLE 1,Adding new columns as "year" and "month"
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from datetime import datetime

# Define a UDF to extract the year from the date string
def extract_year(date_str):
    date_obj = datetime.strptime(date_str, '%m %d, %Y')
    return date_obj.year

year_udf = udf(extract_year, IntegerType())

df = df.withColumn("year", year_udf(df["reviewTime"]))

def extract_month(date_str):
    date_obj = datetime.strptime(date_str, '%m %d, %Y')
    return date_obj.month

month_udf = udf(extract_month, IntegerType())

df = df.withColumn("month", month_udf(df["reviewTime"]))

df.show()


# COMMAND ----------

# DBTITLE 1,Adding sentiment analysis (Phase 1)
import nltk
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType, FloatType
from textblob import TextBlob

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sent = SentimentIntensityAnalyzer()

def check_subjectivity(x):
    return TextBlob(x).subjectivity

def check_polarity(x):
    return TextBlob(x).polarity

def check_compound(x):
    return sent.polarity_scores(x)['compound']

subjectivity_udf = udf(check_subjectivity, FloatType())
polarity_udf = udf(check_polarity, FloatType())
compound_udf = udf(check_compound, FloatType())

df = df.withColumn("subjectivity", subjectivity_udf(df["reviewText"]))
df = df.withColumn("polarity", polarity_udf(df["reviewText"]))
df = df.withColumn("compound", compound_udf(df["reviewText"]))

df.show()

# COMMAND ----------

# DBTITLE 1,Sentiment EDA (Phase 1)
from pyspark.sql.functions import col, min, max, mean

# Specify the columns for which you want to calculate statistics
columns_to_aggregate = ["subjectivity", "polarity", "compound"]

# Define the aggregation functions for each column
agg_exprs = [min(col(col_name)).alias(f"min_{col_name}") for col_name in columns_to_aggregate] + \
            [max(col(col_name)).alias(f"max_{col_name}") for col_name in columns_to_aggregate] + \
            [mean(col(col_name)).alias(f"mean_{col_name}") for col_name in columns_to_aggregate]

# Perform the grouping and aggregation
result = df.groupBy("label").agg(*agg_exprs)

# Show the result
result.show()

# COMMAND ----------

# DBTITLE 1,Adding sentiment analysis (Phase 2)
import nltk
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType, FloatType
from textblob import TextBlob

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sent = SentimentIntensityAnalyzer()

def check_neg(x):
    return sent.polarity_scores(x)['neg']

def check_neu(x):
    return sent.polarity_scores(x)['neu']

def check_pos(x):
    return sent.polarity_scores(x)['pos']

neg_udf = udf(check_neg, FloatType())
neu_udf = udf(check_neu, FloatType())
pos_udf = udf(check_pos, FloatType())

df = df.withColumn("neg", neg_udf(df["reviewText"]))
df = df.withColumn("neu", neu_udf(df["reviewText"]))
df = df.withColumn("pos", pos_udf(df["reviewText"]))

df.show()

# COMMAND ----------

# DBTITLE 1,Sentiment EDA (Phase 2)
from pyspark.sql.functions import col, min, max, mean

# Specify the columns for which you want to calculate statistics
columns_to_aggregate = ["neg", "neu", "pos"]

# Define the aggregation functions for each column
agg_exprs = [min(col(col_name)).alias(f"min_{col_name}") for col_name in columns_to_aggregate] + \
            [max(col(col_name)).alias(f"max_{col_name}") for col_name in columns_to_aggregate] + \
            [mean(col(col_name)).alias(f"mean_{col_name}") for col_name in columns_to_aggregate]

# Perform the grouping and aggregation
result = df.groupBy("label").agg(*agg_exprs)

# Show the result
result.show()

# COMMAND ----------

# DBTITLE 1,Split into testing/training
# set seed for reproducibility
(trainingData, testingData) = df.randomSplit([0.8, 0.2], seed = 47)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testingData.count()))

# COMMAND ----------

# Get the top 10% review time of each product
from pyspark.sql.functions import col, count, ceil, rank
from pyspark.sql.window import Window

threshold_df = df.groupBy("asin").agg(ceil(count("reviewID") * 0.1).alias("threshold"))

window_spec = Window.partitionBy('asin').orderBy(col('unixReviewTime').asc())
df = df.withColumn("reviewsOrder", rank().over(window_spec))

top_review_time_df = df.join(threshold_df, on="asin", how="inner").filter(df['reviewsOrder'] == threshold_df['threshold']).select("asin", "unixReviewTime")
top_review_time_df = top_review_time_df.withColumnRenamed("unixReviewTime", "reviewTimeThreshold")
top_review_time_df = top_review_time_df.distinct()

# COMMAND ----------

# DBTITLE 1,Even more features!
# get average rating of each product
from pyspark.sql.window import Window
from pyspark.sql import functions as F

window_spec = Window.partitionBy("asin")
avg_rating = df.withColumn("average_rating", F.avg("overall").over(window_spec))
avg_rating = avg_rating.select("asin", "average_rating").distinct()


def add_new_features(originalDf):

    newDf = originalDf

    # handle null
    newDf = newDf.na.fill(value='',subset=["reviewText"])

    # rating deviation
    newDf = newDf.join(avg_rating, on="asin", how="left_outer")
    newDf = newDf.fillna(0, subset=["average_rating"])
    newDf = newDf.withColumn("deviation_rating",  newDf["overall"]- newDf["average_rating"])

    # customer deviation
    newDf = newDf.join(customer_rating, on="reviewerID", how="left_outer")
    newDf = newDf.fillna(0, subset=["customer_rating"])
    newDf = newDf.withColumn("deviation_customer",  newDf["overall"]- newDf["customer_rating"])

    # aboslute
    #from pyspark.sql.functions import col, expr
    #newDf = newDf.train_data.withColumn("absolute_dev_rating", expr("abs(deviation_rating)"))


    # verify dummy
    newDf = newDf.withColumn("verified_dummy", (newDf["verified"] == True).cast("integer")).drop("verified")

    # time weight
    from pyspark.sql.functions import from_unixtime
    from pyspark.sql.functions import current_date, datediff
    from pyspark.sql.functions import exp

    newDf = newDf.withColumn("reviewTimestamp", from_unixtime(newDf["unixReviewTime"]))
    newDf = newDf.withColumn("days_since_review", datediff(current_date(), newDf["reviewTimestamp"]))

    def calculate_weight(days_since_review):
        return exp(-0.01 * days_since_review)
    
    newDf = newDf.withColumn("review_weight", calculate_weight(newDf["days_since_review"]))



    # Has non-useful words in summary
    from pyspark.sql.functions import col, when, regexp_extract, lower,  regexp_replace
    from pyspark.sql.types import DoubleType

    newDf = newDf.withColumn(
        "has_non_useful_summary",
        when(
            (lower(col("summary")).like("%star%")|
            lower(col("summary")).like("%...%")|
            (regexp_extract(col("summary"), "[a-zA-Z]+", 0) == "")|
            (regexp_replace(col("summary"), "[\\p{P}\\p{Z}]", "") == "")),
            1
        ).otherwise(0)
    )
    newDf = newDf.withColumn("has_non_useful_summary", col("has_non_useful_summary").cast(DoubleType()))

    # Is Top 10% of review
    from pyspark.sql.functions import col, when
    newDf = newDf.join(top_review_time_df, on="asin", how="left")
    newDf = newDf.withColumn("isTopTenReview", when((col("reviewTimeThreshold").isNotNull()) & (col("unixReviewTime") < col("reviewTimeThreshold")), 1).otherwise(0))
    newDf = newDf.fillna(1, subset=["isTopTenReview"])  # Fill NaN values with True for products not in result_df
    newDf = newDf.withColumn("isTopTenReview", col("isTopTenReview").cast(DoubleType()))

    # Check if revierName has 'amazon'
    newDf = newDf.withColumn("is_name_amazon", when(lower(col("reviewerName")).contains("amazon"), 1).otherwise(0))

    return newDf

# COMMAND ----------

df = add_new_features(df)

# COMMAND ----------

trainingData = add_new_features(trainingData)
testingData = add_new_features(testingData)

# COMMAND ----------

import re
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import *
from pyspark.sql.types import IntegerType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.clustering import LDA

from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, StringIndexer, SQLTransformer, IndexToString, VectorAssembler, Word2Vec, MinMaxScaler, VarianceThresholdSelector
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, MultilayerPerceptronClassifier
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import udf, col
import pyspark.sql.functions as F

# Initialize the lemmatization model
lemmatizer_model = LemmatizerModel.pretrained()

# convert text column to spark nlp document
document_assembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")


# convert document to array of tokens
tokenizer = Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("token")

#convert document to array of sentences
sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentences")

# convert text column to spark nlp document
document_assembler2 = DocumentAssembler() \
    .setInputCol("summary") \
    .setOutputCol("sum_document")

# convert document to array of tokens
tokenizer2 = Tokenizer() \
  .setInputCols(["sum_document"]) \
  .setOutputCol("sum_token")

# Use SQLTransformer to add a new column with the count of exclamation marks
exclamation_mark_counter = SQLTransformer(
    statement="SELECT *, LENGTH(reviewText) - LENGTH(REGEXP_REPLACE(reviewText, '!', '')) AS exclamation_marks_count FROM __THIS__"
)


# clean tokens 
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")

# remove stopwords
stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

# stems tokens to bring it to root form
#stemmer = Stemmer() \
#    .setInputCols(["cleanTokens"]) \
#   .setOutputCol("stem")

# Lemmatize tokens using the lemmatization model
lemmatizer = LemmatizerModel.pretrained() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("lemma")

# Convert custom document structure to array of tokens.
finisher = Finisher() \
    .setInputCols(["lemma"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)

# Calculate the number of tokens and sentences using SQLTransformer
token_count_sql = SQLTransformer(
    statement="SELECT *, SIZE(cleanTokens) AS num_tokens FROM __THIS__"
)

sentence_count_sql = SQLTransformer(
    statement="SELECT *, SIZE(sentences) AS num_sentences FROM __THIS__"
)

# Calculate the number of tokens and sentences using SQLTransformer
summary_count_sql = SQLTransformer(
    statement="SELECT *, SIZE(sum_token) AS num_summary FROM __THIS__"
)

# Create an SQLTransformer to add the binary variable
sentence_criteria = SQLTransformer(
    statement="SELECT *, CASE WHEN num_sentences > 470 THEN 1 ELSE 0 END AS sentence_criteria FROM __THIS__"
)

# Create an SQLTransformer to add the binary variable
summary_criteria = SQLTransformer(
    statement="SELECT *, CASE WHEN num_summary < 55 THEN 1 ELSE 0 END AS summary_criteria FROM __THIS__"
)

# Create an SQLTransformer to add the binary variable
exclamation_criteria = SQLTransformer(
    statement="SELECT *, CASE WHEN exclamation_marks_count < 230 THEN 1 ELSE 0 END AS exclamation_criteria FROM __THIS__"
)

# Generate Term Frequency
tf = CountVectorizer(inputCol="token_features", outputCol="rawFeatures", vocabSize=90000, minTF=1, minDF=50, maxDF=0.2)

hashing_tf = HashingTF(inputCol="token_features", outputCol="hashingTF_features", numFeatures=90000)

# Generate Inverse Document Frequency weighting
idf = IDF(inputCol="rawFeatures", outputCol="idfFeatures", minDocFreq=5)

# Calculate the normalized TF and add it as a new column "normalizedTF"
normalized_tf_sql = SQLTransformer(
    statement="SELECT *, TRANSFORM(rawFeatures, x -> x / SIZE(rawFeatures)) AS normalizedTF FROM __THIS__"
)

# Define the LDA model
lda = LDA(k=200, maxIter=10, featuresCol="rawFeatures", topicDistributionCol="lda_features")


# Assuming "unixReviewTime" is a numeric column
time_assembler = VectorAssembler(inputCols=["unixReviewTime"], outputCol="vectorReviewTime")

# Create a MinMaxScaler instance
min_max_scaler = MinMaxScaler(inputCol="vectorReviewTime", outputCol="scaledReviewTime")

labelIndexer = StringIndexer(inputCol="overall", outputCol="indexedScore")


# Combine all features into one final "features" column
assembler = VectorAssembler(inputCols=["verified_dummy", "indexedScore" ,"idfFeatures", "hashingTF_features", "deviation_rating", "scaledReviewTime", "num_tokens", "num_sentences", "num_summary", "Product Reviews", "Customer Reviews", "exclamation_marks_count", "polarity", "neg", "neu", "pos", "exclamation_criteria", "sentence_criteria", "summary_criteria", "year", "month", "lda_features", "review_weight", "has_non_useful_summary", "isTopTenReview", "is_name_amazon"], outputCol="features")

#selector = VarianceThresholdSelector(varianceThreshold=0.05).setFeaturesCol('vec_features').setOutputCol('features')

# Machine Learning Algorithm
#ml_alg  = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.0)
ml_alg=RandomForestClassifier(numTrees=100, impurity='gini', maxDepth=4, maxBins=32)



nlp_pipeline = Pipeline(
    stages=[document_assembler, 
            tokenizer,
            sentence_detector,
            document_assembler2, 
            tokenizer2,
            exclamation_mark_counter,
            normalizer,
            stopwords_cleaner, 
            lemmatizer, 
            finisher,
            token_count_sql,
            sentence_count_sql,
            summary_count_sql,
            sentence_criteria,
            summary_criteria,
            exclamation_criteria,
            tf,
            hashing_tf,
            idf,
            lda,
            time_assembler,
            min_max_scaler,
            labelIndexer,
            assembler])


transformed_training=nlp_pipeline.fit(trainingData).transform(trainingData)
transformed_testing=nlp_pipeline.fit(testingData).transform(testingData)
            

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# Define the hyperparameter grid to search
paramGrid = (ParamGridBuilder()
             .addGrid(ml_alg.regParam, [0.01, 0.1, 1.0])  # Specify different values for regularization parameter
             .build())

# Define the CrossValidator
crossval = CrossValidator(estimator=nlp_pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(metricName="areaUnderROC"),
                          numFolds=5)  # Number of cross-validation folds

# Fit the CrossValidator to your data
cvModel = crossval.fit(trainingData)

# Evaluate the best model on your test data
best_model = cvModel.bestModel
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
auc = evaluator.evaluate(best_model.transform(testingData))

print("Best AUC: ", auc)


# COMMAND ----------


from pyspark.ml.classification import RandomForestClassifier

# Train a RandomForest model.
rf = RandomForestClassifier(numTrees=100, maxDepth=4, maxBins=32)

ml_model = rf.fit(transformed_training)
predictions =  ml_model.transform(transformed_testing)

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier

# Train a DecisionTree model.
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

ml_model = dt.fit(transformed_training)
predictions =  ml_model.transform(transformed_testing)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
auc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
print("Test areaUnderROC   = %g" % (auc_evaluator.evaluate(predictions)))

# COMMAND ----------

from pyspark.ml.classification import MultilayerPerceptronClassifier

layers = [4, 5, 4, 3]

# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234)
nn_model = trainer.fit(transformed_training)
result =  nn_model.transform(transformed_testing)
predictionAndLabels = result.select("prediction", "label")

# COMMAND ----------

# DBTITLE 1,Spark NLP Pipeline
from pyspark.ml.feature import Word2Vec
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import RegexTokenizer, Normalizer, StopWordsCleaner, LemmatizerModel
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, IDF, StringIndexer, IndexToString, VectorAssembler, NGram
from pyspark.ml.classification import RandomForestClassifier

# Initialize the lemmatization model
lemmatizer_model = LemmatizerModel.pretrained()

# Convert text column to Spark NLP document
document_assembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")

# Convert document to an array of tokens
tokenizer = RegexTokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token") \
    .setPattern("\\W")

# Remove stopwords
#stopwords_cleaner = StopWordsCleaner() \
#    .setInputCols("normalized") \
#    .setOutputCol("cleanTokens") \
#    .setCaseSensitive(False)

# Lemmatize tokens using the lemmatization model
# lemmatizer = LemmatizerModel.pretrained() \
#    .setInputCols(["cleanTokens"]) \
#    .setOutputCol("lemma")

# Convert custom document structure to an array of tokens
finisher = Finisher() \
    .setInputCols(["token"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)

# Load a pre-trained Word2Vec model
word2vec = Word2Vec() \
    .setInputCol("token_features") \
    .setOutputCol("word2vec_embeddings") \
    .setVectorSize(100) \
    .setMinCount(5)

# Combine all features into one final "features" column
assembler = VectorAssembler(inputCols=["verified", "overall", "word2vec_embeddings"], outputCol="features")

# Machine Learning Algorithm
ml_alg  = RandomForestClassifier(numTrees=100, featureSubsetStrategy="auto", impurity='gini', maxDepth=4, maxBins=32)

nlp_pipeline = Pipeline(
    stages=[document_assembler,
            tokenizer,
            finisher,
            word2vec,
            assembler,
            ml_alg])


# COMMAND ----------

pipeline_model = nlp_pipeline.fit(trainingData)
predictions =  pipeline_model.transform(testingData)

display(predictions)

# COMMAND ----------

predictions.groupBy("label").count().show()
predictions.groupBy("prediction").count().show()

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
auc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
print("Test areaUnderROC   = %g" % (auc_evaluator.evaluate(predictions)))

# COMMAND ----------

from pyspark.ml.feature import Word2Vec
from sparknlp.base import *
from sparknlp.annotator import *
from transformers import BertTokenizer, AutoTokenizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, IDF, StringIndexer, IndexToString, VectorAssembler, SQLTransformer
from pyspark.ml.classification import RandomForestClassifier

# Initialize the lemmatization model
lemmatizer_model = LemmatizerModel.pretrained()

# Convert text column to Spark NLP document
document_assembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")

# Convert document to an array of tokens
tokenizer = RegexTokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token") \
    .setPattern("\\W")


# Convert custom document structure to an array of tokens
finisher = Finisher() \
    .setInputCols(["token"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)

# Load a pre-trained Word2Vec model
#word2vec = Word2Vec() \
  #  .setInputCol("token_features") \
   # .setOutputCol("word2vec_embeddings") \
    #.setVectorSize(100) \
    #.setMinCount(5)

embeddings = BertEmbeddings.pretrained("small_bert_L2_128", "en") \
    .setInputCols(["token", "document"]) \
    .setOutputCol("bert_embeddings")

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["bert_embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True)

explodeVectors = SQLTransformer(statement=
      "SELECT EXPLODE(finished_embeddings) AS exploded_embeddings, * FROM __THIS__")

# Combine all features into one final "features" column
assembler = VectorAssembler(inputCols=["verified", "overall", "exploded_embeddings"], outputCol="features")

# Machine Learning Algorithm
ml_alg  = RandomForestClassifier(numTrees=100, featureSubsetStrategy="auto", impurity='gini', maxDepth=4, maxBins=32)

nlp_pipeline = Pipeline(
    stages=[document_assembler,
            tokenizer,
            finisher,
            embeddings,
            embeddingsFinisher,
            explodeVectors,
            assembler,
            ml_alg])



# COMMAND ----------

pipeline_model = nlp_pipeline.fit(trainingData)
predictions =  pipeline_model.transform(testingData)

# COMMAND ----------

import pyspark.sql.functions as F
import pyspark.sql.types as T

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors, VectorUDT

def avg_vectors(bert_vectors):
  length = len(bert_vectors[0]["embeddings"])
  avg_vec = [0] * length
  for vec in bert_vectors:
    for i, x in enumerate(vec["embeddings"]):
      avg_vec[i] += x
    avg_vec[i] = avg_vec[i] / length
  return avg_vec


#create a udf
avg_vectors_udf = F.udf(avg_vectors, T.ArrayType(T.DoubleType()))
df_doc_vec = df_transformed.withColumn("doc_vector", avg_vectors_udf(F.col("embeddings")))

def dense_vector(vec):
	return Vectors.dense(vec)

dense_vector_udf = F.udf(dense_vector, VectorUDT())
training = df_doc_vec.withColumn("features", dense_vector_udf(F.col("doc_vector")))

# Machine Learning Algorithm
#ml_alg  = RandomForestClassifier(numTrees=100, featureSubsetStrategy="auto", impurity='gini', maxDepth=4, maxBins=32)
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10, regParam=0.3, elasticNetParam=0.8)
pipeline_model = lr.fit(training)


# COMMAND ----------

# Machine Learning Algorithm
ml_alg  = RandomForestClassifier(numTrees=100, featureSubsetStrategy="auto", impurity='gini', maxDepth=4, maxBins=32)
pipeline_model = ml_alg.fit(newdf)

# COMMAND ----------

from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, SQLTransformer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.linalg import Vectors, VectorUDT

# Convert text column to Spark NLP document
document_assembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")

# Convert document to an array of tokens
tokenizer = RegexTokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token") \
    .setPattern("\\W")

# Initialize the SentenceDetector
sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentences") \
    .setExplodeSentences(False)

# Load a pre-trained Word Embeddings model (e.g., Word2Vec or GloVe)
word_embeddings = WordEmbeddingsModel.pretrained() \
    .setInputCols(["document", "token"]) \
    .setOutputCol("word_embeddings")

# Initialize a SentenceEmbeddings annotator
sentence_embeddings = SentenceEmbeddings() \
    .setInputCols(["document", "word_embeddings"]) \
    .setOutputCol("sentence_embeddings") \
    .setPoolingStrategy("AVERAGE")  # You can choose a pooling strategy (e.g., AVERAGE, MAX)


# Combine all features into one final "features" column
#assembler = VectorAssembler(inputCols=["sentence_embeddings"], outputCol="features")

nlp_pipeline = Pipeline(
    stages=[document_assembler,
            tokenizer,
            sentence_detector, 
            word_embeddings,
            sentence_embeddings])


df_transformed = nlp_pipeline.fit(trainingData).transform(trainingData)

def combine_arrays_udf(array):
    combined = []

    # Iterate through the arrays of floats and concatenate them
    for arr in array:
        combined.extend(arr)

    # Create a new dense vector from the concatenated list
    return Vectors.dense(combined)

# Register the UDF with appropriate input and output types
combine_arrays = udf(combine_arrays_udf, VectorUDT())

# Apply the UDF to your DataFrame to create a new column 'features'
df_with_vectors = df_transformed.select(
    df_transformed["label"],
    combine_arrays(df_transformed["sentence_embeddings"]).alias("features")
)


# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.linalg import Vectors, VectorUDT

def avg_vectors(bert_vectors):
    if not bert_vectors:
        return [0]  # Handle the case where no vectors are provided
    length = len(bert_vectors[0]["embeddings"])
    avg_vec = [0] * length
    for vec in bert_vectors:
        if "embeddings" in vec:
            for i, x in enumerate(vec["embeddings"]):
                avg_vec[i] += x
    # Calculate the average, handling the case where length is 0
    avg_vec = [x / len(bert_vectors) if len(bert_vectors) > 0 else x for x in avg_vec]
    return avg_vec


#create a udf
avg_vectors_udf = F.udf(avg_vectors, T.ArrayType(T.DoubleType()))
df_doc_vec = df_transformed.withColumn("doc_vector", avg_vectors_udf(F.col("embeddings")))


def dense_vector(vec):
	return Vectors.dense(vec)

dense_vector_udf = F.udf(dense_vector, VectorUDT())
training = df_doc_vec.withColumn("features", dense_vector_udf(F.col("doc_vector")))


# Machine Learning Algorithm
ml_alg  = RandomForestClassifier(numTrees=100, featureSubsetStrategy="auto", impurity='gini', maxDepth=4, maxBins=32)
pipeline_model = ml_alg.fit(training)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

predictions =  pipeline_model.transform(testingData)
auc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
print("Test areaUnderROC   = %g" % (auc_evaluator.evaluate(predictions)))

# COMMAND ----------

# DBTITLE 1,Create a Data Transformation/Preprocessing Pipeline
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer

# We'll tokenize the text using a simple RegexTokenizer
regexTokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")


# Remove standard Stopwords
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")


# Vectorize the sentences using simple BOW method. Other methods are possible:
# https://spark.apache.org/docs/2.2.0/ml-features.html#feature-extractors
countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)


pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, countVectors])




# COMMAND ----------

# DBTITLE 1,Transform Training Data
# Fit the pipeline to training documents.
pipelineFit = pipeline.fit(trainingData)
trainingDataTransformed = pipelineFit.transform(trainingData)
trainingDataTransformed.show(5)

# COMMAND ----------

# DBTITLE 1,Build Logistic Regression Model
from pyspark.ml.classification import LogisticRegression

# More classification docs: https://spark.apache.org/docs/latest/ml-classification-regression.html

lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingDataTransformed)

# COMMAND ----------

# DBTITLE 1,Show Training Metrics
# Extract the summary from the returned LogisticRegressionModel instance trained
# in the earlier example
trainingSummary = lrModel.summary

print("Training Accuracy:  " + str(trainingSummary.accuracy))
print("Training Precision: " + str(trainingSummary.precisionByLabel))
print("Training Recall:    " + str(trainingSummary.recallByLabel))
print("Training FMeasure:  " + str(trainingSummary.fMeasureByLabel()))
print("Training AUC:       " + str(trainingSummary.areaUnderROC))

# COMMAND ----------

trainingSummary.roc.show()

# COMMAND ----------

# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
for objective in objectiveHistory:
    print(objective)

# COMMAND ----------

# DBTITLE 1,Transform Testing Data
testingDataTransform = pipelineFit.transform(testingData)
testingDataTransform.show(5)

# COMMAND ----------

# DBTITLE 1,Use Model to Predict Test Data; Evaluate
from pyspark.ml.evaluation import BinaryClassificationEvaluator

predictions = lrModel.transform(testingDataTransform)
predictions.show(5)

evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
print('Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

# DBTITLE 1,Make Predictions on Kaggle Test Data
# Load in the tables
test_df = spark.sql("select * from default.reviews_test")
test_df.show(5)
print((test_df.count(), len(test_df.columns)))

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.sql.functions import col

concatenated_df = data.drop("label").union(test_df)

spark = SparkSession.builder.appName("RowNumberGroupByExample").getOrCreate()

asin_window_spec = Window.partitionBy("asin").orderBy("unixReviewTime")
reviewer_window_spec = Window.partitionBy("reviewerID").orderBy("unixReviewTime")

# Add row numbers within each group
concatenated_df = concatenated_df.withColumn("Product Reviews", row_number().over(asin_window_spec))
concatenated_df = concatenated_df.withColumn("Customer Reviews", row_number().over(reviewer_window_spec))

selected_columns = concatenated_df.select("reviewID", "Product Reviews", "Customer Reviews")
test_df = test_df.join(selected_columns, on=["reviewID"], how="left")

test_df.show()


# COMMAND ----------

import nltk
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType, FloatType
from textblob import TextBlob

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sent = SentimentIntensityAnalyzer()

def check_subjectivity(x):
    return TextBlob(x).subjectivity

def check_polarity(x):
    return TextBlob(x).polarity

def check_compound(x):
    return sent.polarity_scores(x)['compound']

def check_neg(x):
    return sent.polarity_scores(x)['neg']

def check_neu(x):
    return sent.polarity_scores(x)['neu']

def check_pos(x):
    return sent.polarity_scores(x)['pos']

neg_udf = udf(check_neg, FloatType())
neu_udf = udf(check_neu, FloatType())
pos_udf = udf(check_pos, FloatType())
subjectivity_udf = udf(check_subjectivity, FloatType())
polarity_udf = udf(check_polarity, FloatType())
compound_udf = udf(check_compound, FloatType())

test_df = test_df.withColumn("neg", neg_udf(test_df["reviewText"]))
test_df = test_df.withColumn("neu", neu_udf(test_df["reviewText"]))
test_df = test_df.withColumn("pos", pos_udf(test_df["reviewText"]))
test_df = test_df.withColumn("subjectivity", subjectivity_udf(test_df["reviewText"]))
test_df = test_df.withColumn("polarity", polarity_udf(test_df["reviewText"]))
test_df = test_df.withColumn("compound", compound_udf(test_df["reviewText"]))

test_df.show()

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from datetime import datetime

# Define a UDF to extract the year from the date string
def extract_year(date_str):
    date_obj = datetime.strptime(date_str, '%m %d, %Y')
    return date_obj.year

year_udf = udf(extract_year, IntegerType())

df = df.withColumn("year", year_udf(df["reviewTime"]))
test_df = test_df.withColumn("year", year_udf(test_df["reviewTime"]))

def extract_month(date_str):
    date_obj = datetime.strptime(date_str, '%m %d, %Y')
    return date_obj.month

month_udf = udf(extract_month, IntegerType())

test_df = test_df.withColumn("month", month_udf(test_df["reviewTime"]))

df.show()

# COMMAND ----------

# Get the top 10% review time of each product
from pyspark.sql.functions import col, count, ceil, rank
from pyspark.sql.window import Window

threshold_df = concatenated_df.groupBy("asin").agg(ceil(count("reviewID") * 0.1).alias("threshold"))

window_spec = Window.partitionBy('asin').orderBy(col('unixReviewTime').asc())
concatenated_df = concatenated_df.withColumn("reviewsOrder", rank().over(window_spec))

top_review_time_df = concatenated_df.join(threshold_df, on="asin", how="inner").filter(concatenated_df['reviewsOrder'] == threshold_df['threshold']).select("asin", "unixReviewTime")
top_review_time_df = top_review_time_df.withColumnRenamed("unixReviewTime", "reviewTimeThreshold")
top_review_time_df = top_review_time_df.distinct()


window_spec = Window.partitionBy("asin")
avg_rating = concatenated_df.withColumn("average_rating", F.avg("overall").over(window_spec))
avg_rating = avg_rating.select("asin", "average_rating").distinct()

# COMMAND ----------

test_df = add_new_features(test_df)

# COMMAND ----------

def add_new_features2(originalDf):

    newDf = originalDf

    # handle null
    newDf = newDf.na.fill(value='',subset=["reviewText"])

    # customer deviation
    newDf = newDf.join(customer_rating, on="reviewerID", how="left_outer")
    newDf = newDf.fillna(0, subset=["customer_rating"])
    newDf = newDf.withColumn("deviation_customer",  newDf["overall"]- newDf["customer_rating"])

# COMMAND ----------

window_spec = Window.partitionBy("reviewerID")
customer_rating = df.withColumn("customer_rating", F.avg("overall").over(window_spec))
customer_rating = customer_rating.select("reviewerID", "customer_rating").distinct()

# COMMAND ----------

window_spec = Window.partitionBy("reviewerID")
customer_history = df.withColumn("customer_history", F.avg("label").over(window_spec))
customer_history = customer_history.select("reviewerID", "customer_history").distinct()
df = df.join(customer_history, on="reviewerID", how="left_outer")
df = df.fillna(0, subset=["customer_history"])
test_df = test_df.join(customer_history, on="reviewerID", how="left_outer")
test_df = test_df.fillna(0, subset=["customer_history"])

# COMMAND ----------

df = add_new_features2(df)

# COMMAND ----------

window_spec = Window.partitionBy("reviewerID")
customer_rating = concatenated_df.withColumn("customer_rating", F.avg("overall").over(window_spec))
customer_rating = customer_rating.select("reviewerID", "customer_rating").distinct()

# COMMAND ----------

test_df = add_new_features2(test_df)

# COMMAND ----------

import re
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import *
from pyspark.sql.types import IntegerType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.clustering import LDA

from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, StringIndexer, SQLTransformer, IndexToString, VectorAssembler, Word2Vec, MinMaxScaler, VarianceThresholdSelector
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, MultilayerPerceptronClassifier, NaiveBayes
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import udf, col
import pyspark.sql.functions as F

# Initialize the lemmatization model
lemmatizer_model = LemmatizerModel.pretrained()

# convert text column to spark nlp document
document_assembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")


# convert document to array of tokens
tokenizer = Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("token")

#convert document to array of sentences
sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentences")

# convert text column to spark nlp document
document_assembler2 = DocumentAssembler() \
    .setInputCol("summary") \
    .setOutputCol("sum_document")

# convert document to array of tokens
tokenizer2 = Tokenizer() \
  .setInputCols(["sum_document"]) \
  .setOutputCol("sum_token")

# Use SQLTransformer to add a new column with the count of exclamation marks
exclamation_mark_counter = SQLTransformer(
    statement="SELECT *, LENGTH(reviewText) - LENGTH(REGEXP_REPLACE(reviewText, '!', '')) AS exclamation_marks_count FROM __THIS__"
)


# clean tokens 
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")

# remove stopwords
stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

# stems tokens to bring it to root form
#stemmer = Stemmer() \
#    .setInputCols(["cleanTokens"]) \
#   .setOutputCol("stem")

# Lemmatize tokens using the lemmatization model
lemmatizer = LemmatizerModel.pretrained() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("lemma")

# Convert custom document structure to array of tokens.
finisher = Finisher() \
    .setInputCols(["lemma"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)

# Calculate the number of tokens and sentences using SQLTransformer
token_count_sql = SQLTransformer(
    statement="SELECT *, SIZE(cleanTokens) AS num_tokens FROM __THIS__"
)

sentence_count_sql = SQLTransformer(
    statement="SELECT *, SIZE(sentences) AS num_sentences FROM __THIS__"
)

# Calculate the number of tokens and sentences using SQLTransformer
summary_count_sql = SQLTransformer(
    statement="SELECT *, SIZE(sum_token) AS num_summary FROM __THIS__"
)

# Create an SQLTransformer to add the binary variable
sentence_criteria = SQLTransformer(
    statement="SELECT *, CASE WHEN num_sentences > 470 THEN 1 ELSE 0 END AS sentence_criteria FROM __THIS__"
)

# Create an SQLTransformer to add the binary variable
summary_criteria = SQLTransformer(
    statement="SELECT *, CASE WHEN num_summary < 55 THEN 1 ELSE 0 END AS summary_criteria FROM __THIS__"
)

# Create an SQLTransformer to add the binary variable
exclamation_criteria = SQLTransformer(
    statement="SELECT *, CASE WHEN exclamation_marks_count < 230 THEN 1 ELSE 0 END AS exclamation_criteria FROM __THIS__"
)

# Generate Term Frequency
tf = CountVectorizer(inputCol="token_features", outputCol="rawFeatures", vocabSize=10000, minTF=1, minDF=50, maxDF=0.3)

hashing_tf = HashingTF(inputCol="token_features", outputCol="hashingTF_features", numFeatures=10000)

# Generate Inverse Document Frequency weighting
idf = IDF(inputCol="rawFeatures", outputCol="idfFeatures", minDocFreq=5)

# Calculate the normalized TF and add it as a new column "normalizedTF"
normalized_tf_sql = SQLTransformer(
    statement="SELECT *, TRANSFORM(rawFeatures, x -> x / SIZE(rawFeatures)) AS normalizedTF FROM __THIS__"
)

# Define the LDA model
lda = LDA(k=100, maxIter=10, featuresCol="rawFeatures", topicDistributionCol="lda_features")


# Assuming "unixReviewTime" is a numeric column
time_assembler = VectorAssembler(inputCols=["unixReviewTime"], outputCol="vectorReviewTime")

# Create a MinMaxScaler instance
min_max_scaler = MinMaxScaler(inputCol="vectorReviewTime", outputCol="scaledReviewTime")

labelIndexer = StringIndexer(inputCol="overall", outputCol="indexedScore")
asinIndexer = StringIndexer(inputCol="asin", outputCol="indexedProduct", handleInvalid="keep")


# Combine all features into one final "features" column
assembler = VectorAssembler(inputCols=["verified_dummy", "indexedScore" ,"idfFeatures", "hashingTF_features", "deviation_rating", "deviation_customer", "scaledReviewTime", "num_tokens", "num_sentences", "num_summary", "Product Reviews", "Customer Reviews", "exclamation_marks_count", "exclamation_criteria", "sentence_criteria", "summary_criteria", "year", "month", "lda_features", "review_weight", "has_non_useful_summary", "isTopTenReview", "is_name_amazon", "indexedProduct"], outputCol="features")


# Machine Learning Algorithm
ml_alg  = LogisticRegression(maxIter=100, regParam=0.05, elasticNetParam=0.0)



nlp_pipeline = Pipeline(
    stages=[document_assembler, 
            tokenizer,
            sentence_detector,
            document_assembler2, 
            tokenizer2,
            exclamation_mark_counter,
            normalizer,
            stopwords_cleaner, 
            lemmatizer, 
            finisher,
            token_count_sql,
            sentence_count_sql,
            summary_count_sql,
            sentence_criteria,
            summary_criteria,
            exclamation_criteria,
            tf,
            hashing_tf,
            idf,
            lda,
            time_assembler,
            min_max_scaler,
            labelIndexer,
            asinIndexer,
            assembler,
            ml_alg])
            

# COMMAND ----------

pipeline_model = nlp_pipeline.fit(df)
predictions =  pipeline_model.transform(test_df)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

# Define a UDF to return the second-to-last element
probelement = udf(lambda v: float(v[1]), FloatType())

# Apply the UDF to create a new DataFrame
submission_data = predictions.select('reviewID', probelement('probability')).withColumnRenamed('<lambda>(probability)', 'label')


# COMMAND ----------

display(submission_data)

# COMMAND ----------

import re
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import *
from pyspark.sql.types import IntegerType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.clustering import LDA

from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, StringIndexer, SQLTransformer, IndexToString, VectorAssembler, Word2Vec, MinMaxScaler, VarianceThresholdSelector
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, MultilayerPerceptronClassifier, NaiveBayes
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import udf, col
import pyspark.sql.functions as F

# Initialize the lemmatization model
lemmatizer_model = LemmatizerModel.pretrained()

# convert text column to spark nlp document
document_assembler = DocumentAssembler() \
    .setInputCol("reviewText") \
    .setOutputCol("document")


# convert document to array of tokens
tokenizer = Tokenizer() \
  .setInputCols(["document"]) \
  .setOutputCol("token")

#convert document to array of sentences
sentence_detector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentences")

# convert text column to spark nlp document
document_assembler2 = DocumentAssembler() \
    .setInputCol("summary") \
    .setOutputCol("sum_document")

# convert document to array of tokens
tokenizer2 = Tokenizer() \
  .setInputCols(["sum_document"]) \
  .setOutputCol("sum_token")

# Use SQLTransformer to add a new column with the count of exclamation marks
exclamation_mark_counter = SQLTransformer(
    statement="SELECT *, LENGTH(reviewText) - LENGTH(REGEXP_REPLACE(reviewText, '!', '')) AS exclamation_marks_count FROM __THIS__"
)


# clean tokens 
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")

# remove stopwords
stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

# stems tokens to bring it to root form
#stemmer = Stemmer() \
#    .setInputCols(["cleanTokens"]) \
#   .setOutputCol("stem")

# Lemmatize tokens using the lemmatization model
lemmatizer = LemmatizerModel.pretrained() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("lemma")

# Convert custom document structure to array of tokens.
finisher = Finisher() \
    .setInputCols(["lemma"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)

# Calculate the number of tokens and sentences using SQLTransformer
token_count_sql = SQLTransformer(
    statement="SELECT *, SIZE(cleanTokens) AS num_tokens FROM __THIS__"
)

sentence_count_sql = SQLTransformer(
    statement="SELECT *, SIZE(sentences) AS num_sentences FROM __THIS__"
)

# Calculate the number of tokens and sentences using SQLTransformer
summary_count_sql = SQLTransformer(
    statement="SELECT *, SIZE(sum_token) AS num_summary FROM __THIS__"
)

# Create an SQLTransformer to add the binary variable
sentence_criteria = SQLTransformer(
    statement="SELECT *, CASE WHEN num_sentences > 470 THEN 1 ELSE 0 END AS sentence_criteria FROM __THIS__"
)

# Create an SQLTransformer to add the binary variable
summary_criteria = SQLTransformer(
    statement="SELECT *, CASE WHEN num_summary < 55 THEN 1 ELSE 0 END AS summary_criteria FROM __THIS__"
)

# Create an SQLTransformer to add the binary variable
exclamation_criteria = SQLTransformer(
    statement="SELECT *, CASE WHEN exclamation_marks_count < 230 THEN 1 ELSE 0 END AS exclamation_criteria FROM __THIS__"
)

# Generate Term Frequency
tf = CountVectorizer(inputCol="token_features", outputCol="rawFeatures", vocabSize=10000, minTF=1, minDF=50, maxDF=0.3)

hashing_tf = HashingTF(inputCol="token_features", outputCol="hashingTF_features", numFeatures=10000)

# Generate Inverse Document Frequency weighting
idf = IDF(inputCol="rawFeatures", outputCol="idfFeatures", minDocFreq=5)

# Calculate the normalized TF and add it as a new column "normalizedTF"
normalized_tf_sql = SQLTransformer(
    statement="SELECT *, TRANSFORM(rawFeatures, x -> x / SIZE(rawFeatures)) AS normalizedTF FROM __THIS__"
)


# Assuming "unixReviewTime" is a numeric column
time_assembler = VectorAssembler(inputCols=["unixReviewTime"], outputCol="vectorReviewTime")

# Create a MinMaxScaler instance
min_max_scaler = MinMaxScaler(inputCol="vectorReviewTime", outputCol="scaledReviewTime")

labelIndexer = StringIndexer(inputCol="overall", outputCol="indexedScore")
asinIndexer = StringIndexer(inputCol="asin", outputCol="indexedProduct", handleInvalid="keep")
customerIndexer = StringIndexer(inputCol="reviewerID", outputCol="indexedCustomer", handleInvalid="keep")


# Combine all features into one final "features" column
assembler = VectorAssembler(inputCols=["verified_dummy", "indexedScore" ,"idfFeatures", "hashingTF_features", "deviation_rating", "deviation_customer", "scaledReviewTime", "num_tokens", "num_sentences", "num_summary", "Product Reviews", "Customer Reviews", "exclamation_marks_count", "exclamation_criteria", "sentence_criteria", "summary_criteria", "year", "month", "review_weight", "has_non_useful_summary", "isTopTenReview", "is_name_amazon", "indexedProduct", "indexedCustomer", "customer_history"], outputCol="features")


# Machine Learning Algorithm
ml_alg  = LogisticRegression(maxIter=100, regParam=0.02, elasticNetParam=0.0)



nlp_pipeline2 = Pipeline(
    stages=[document_assembler, 
            tokenizer,
            sentence_detector,
            document_assembler2, 
            tokenizer2,
            exclamation_mark_counter,
            normalizer,
            stopwords_cleaner, 
            lemmatizer, 
            finisher,
            token_count_sql,
            sentence_count_sql,
            summary_count_sql,
            sentence_criteria,
            summary_criteria,
            exclamation_criteria,
            tf,
            hashing_tf,
            idf,
            time_assembler,
            min_max_scaler,
            labelIndexer,
            asinIndexer,
            customerIndexer,
            assembler,
            ml_alg])
            

# COMMAND ----------

pipeline_model2 = nlp_pipeline2.fit(df)
predictions2 =  pipeline_model2.transform(test_df)

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

# Define a UDF to return the second-to-last element
probelement = udf(lambda v: float(v[1]), FloatType())

# Apply the UDF to create a new DataFrame
submission_data2 = predictions2.select('reviewID', probelement('probability')).withColumnRenamed('<lambda>(probability)', 'label')


# COMMAND ----------

display(submission_data2)

# COMMAND ----------

