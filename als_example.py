import os,json,logging,time
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, count
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_spark_session():
    #load the spark session(local mode)
    ss = (SparkSession.builder
         .master("local")
         .appName("Restaurant Recommendations")
         .getOrCreate())

    ss.sparkContext.setLogLevel("WARN")
    ss.sparkContext.setCheckpointDir('checkpoint')
   
    return ss

if __name__ == '__main__':

    start_time  = time.time()
    ss = init_spark_session() #initial spark session

    final_indexed_save = os.path.join('dataset','review_vegas_als.parquet')
    reviewDF = ss.read.parquet(final_indexed_save)

    logger.error('Number of reviews for Las Vegas is {}'.format(reviewDF.count()))
    #Split data into training and testing sets
    training_set, testing_set = reviewDF.randomSplit([0.97, 0.03])
    logger.error('Size of Training set is {}'.format(training_set.count()))
    logger.error('Size of Testing set is {}'.format(testing_set.count()))
    logger.error('{} seconds has elapsed'.format(time.time() - start_time))
    #build ALS learning model
    als = ALS(rank= 8, maxIter=20, regParam=0.25, userCol="user_id_int", 
              itemCol="business_id_int", ratingCol="stars_long", coldStartStrategy="drop")
    model = als.fit(training_set)
    logger.error('Model fitting done')
    logger.error('{} seconds has elapsed'.format(time.time() - start_time))
    #save model to file
    model_save = os.path.join('dataset','als_model_vegas.parquet')
    #model.write().mode('overwrite').parquet(model_save)
    #Evaluate the modle using RMSE on test set
    predictions = model.transform(testing_set)
    logger.error('Model transformation done')
    logger.error('{} seconds has elapsed'.format(time.time() - start_time))
    logger.error('Calculating RMSE value')
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="stars_long", predictionCol="prediction")
    mae = evaluator.evaluate(predictions) 


    logger.error('RMSE is {} for rank={}, maxIter= {},reg= {}'.format(str(mae),8, 20, 0.25))
    logger.error('{} seconds has elapsed'.format(time.time() - start_time))
    