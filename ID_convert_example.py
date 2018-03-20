import os,json,logging,time
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, count
from pyspark.ml.feature import StringIndexer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_spark_session():
    #load the spark session(local mode)
    ss = (SparkSession.builder
         .master("local")
         .appName("Restaurant Recommendations")
         .getOrCreate())

    #ss.sparkContext.setLogLevel("WARN")
   
    return ss

if __name__ == '__main__':

    start_time  = time.time()
    ss = init_spark_session() #initial spark session

    reviewDF_vegas_save = os.path.join('dataset','review_vegas.parquet')
    reviewDF_vegas = ss.read.parquet(reviewDF_vegas_save).cache()
    #covert user_id and bussiness_id from string to int
    indexer_user = StringIndexer(inputCol ="user_id",outputCol="user_id_int").fit(reviewDF_vegas) 
    indexer_user_save = os.path.join('model','user_ind_model')
    indexer_user.write().overwrite().save(indexer_user_save)

    indexer_business = StringIndexer(inputCol ="business_id",outputCol="business_id_int").fit(reviewDF_vegas)
    indexer_business_save = os.path.join('model', 'bus_ind_model')
    indexer_business.write().overwrite().save(indexer_business_save)
    #transform id columns to string
    indexed = indexer_user.transform(reviewDF_vegas)
    final_indexed = indexer_business.transform(indexed)
    final_indexed.show(20)
    #save fitted strtingIndexer models
    final_indexed_save = os.path.join('dataset','review_vegas_als.parquet')
    final_indexed.write.mode('overwrite').parquet(final_indexed_save)
    logger.error('index compelted save to file')
    logger.error('{} seconds has elapsed'.format(time.time() - start_time))
