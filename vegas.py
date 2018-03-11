import os,json,logging
import time
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def init_spark_context():
    #load the spark context
    conf = SparkConf().setAppName("Restaurant Recommendations")
    #Adding additional Python modules
    conf = (conf.setMaster('local[*]')
        .set('spark.executor.memory', '2G')
        .set('spark.driver.memory', '8G')
        .set('spark.driver.maxResultSize', '5G'))
    sc = SparkContext(conf=conf)
    return sc

def init_spark_session():
    #load the spark session(local mode)
    ss = (SparkSession.builder
         .master("local")
         .appName("Restaurant Recommendations")
         .getOrCreate())

    ss.sparkContext.setLogLevel("WARN")
   
    return ss

def user_string_to_id (dataframe):
    

if __name__ == '__main__':

    start_time  = time.time()

    ss = init_spark_session()
    dataset = os.path.join('dataset','json')
    #Filter for bussiness in Vegas based on geological cordinates
    vegas_lat = 36.114647
    vegas_lon = -115.172813
    lat_range = 0.075
    lon_range = 0.075
    business_file = os.path.join(dataset, 'yelp_academic_dataset_business.json')
    logger.info("loading bussiness data")
    businessDF = ss.read.json(business_file).cache()
    businessDF.printSchema()
    businessDF_vegas = businessDF.filter('latitude between {} and {}' \
                        .format(vegas_lat - lat_range, vegas_lat + lat_range)) \
                        .filter('longitude between {} and {}' \
                        .format(vegas_lon - lon_range, vegas_lon + lon_range)).cache()
    businessDF_vegas.groupBy('city').agg({"*":"count", "review_count":"avg"})\
                    .orderBy('count(1)', ascending=False)\
                    .withColumnRenamed('count(1)', '# Business/City')\
                    .withColumnRenamed('avg(review_count)', 'Avg # Reviews/Business')\
                    .show(25) 

    businessDF.unpersist()

    logger.error('Business data filter for vegas')
    logger.error('{} seconds has elapsed'.format(time.time() - start_time))

    businessDF_vegas_save = os.path.join('dataset','business_vegas.parquet')
    businessDF_vegas.write.mode('overwrite').parquet(businessDF_vegas_save)
    #Join review set on bussiness_id with Vegas Bussiness
    review_file = os.path.join(dataset, 'yelp_academic_dataset_review.json')
    reviewDF = ss.read.json(review_file).cache()
    reviewDF.printSchema()

    reviewDF_vegas = reviewDF.select('business_id','user_id', col('stars').alias('stars_long'))\
                             .join(businessDF_vegas, 'business_id', 'right').cache()
    reviewDF.unpersist()

    businessDF_vegas.unpersist()

    logger.error('Showing review data in vegas')
    logger.error('{} seconds has elapsed'.format(time.time() - start_time))
    
    reviewDF_vegas.agg({"city":"count"}).show(25)
    reviewDF_vegas.printSchema()

    reviewDF_vegas_clean = reviewDF_vegas.select('user_id','business_id','stars_long').cache()
    reviewDF_vegas_clean.show(20)
    reviewDF_vegas_clean.groupBy('stars_long').count().show()

    reviewDF_vegas_clean.unpersist()

    logger.error('Cleaned review data in vegas')
    logger.error('{} seconds has elapsed'.format(time.time() - start_time))
    #Join user with filter review set on user_id
    user_file = os.path.join(dataset, 'yelp_academic_dataset_user.json')
    
    logger.error('{} seconds has elapsed'.format(time.time() - start_time))