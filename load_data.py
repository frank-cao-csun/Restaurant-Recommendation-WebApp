import os,json,logging
import time
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import count, array_contains, col
from pyspark.ml.feature import StringIndexer

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
    ss.sparkContext.setCheckpointDir('checkpoint')
   
    return ss

#def user_string_to_id (dataframe):
def presentInList(listCategory,category):
    for i in listCategory:
        if i == category:
            return True
    return False

def food_related(listCategory):
    for i in listCategory:
        if (i == 'Food' or i =='Restaurants'):
            return True
    return False

def review_ids_to_number(dataframe):
    #build indexer model for user_id
    indexer_user = StringIndexer(inputCol ="user_id",outputCol="user_id_num").fit(dataframe) 
    indexer_user_save = os.path.join('model','user_ind_model')
    indexer_user.write().overwrite().save(indexer_user_save)
    #build indexer model for business_id
    indexer_business = StringIndexer(inputCol ="business_id",outputCol="business_id_num",handleInvalid="skip").fit(dataframe)
    indexer_business_save = os.path.join('model', 'bus_ind_model')
    indexer_business.write().overwrite().save(indexer_business_save)
    #transform id columns to string
    indexed = indexer_user.transform(dataframe)
    final_indexed = indexer_business.transform(indexed)
    final_indexed.show(20)
    #save fitted strtingIndexer models
    final_indexed_save = os.path.join('dataset','review_vegas_als.parquet')
    final_indexed.write.mode('overwrite').parquet(final_indexed_save)
    logger.error('Indexed dataframe for ALS traing saved to review_vegas_als.parquet')
    logger.error('{} seconds has elapsed'.format(time.time() - start_time))

def business_ids_to_number(dataframe):
    indexer_business = StringIndexer(inputCol ="business_id",outputCol="business_id_num").fit(dataframe)
    indexer_business_save = os.path.join('model', 'bus_ind_model')
    indexer_business.write().overwrite().save(indexer_business_save)

if __name__ == '__main__':

    start_time  = time.time()
    ss = init_spark_session()#Initiate Spark Session
    dataset = os.path.join('dataset','json')

    #Filter for bussiness in Vegas based on geological cordinates
    vegas_lat = 36.114647
    vegas_lon = -115.172813
    lat_range = 0.075
    lon_range = 0.075
    business_file = os.path.join(dataset, 'yelp_academic_dataset_business.json')
    logger.error("loading bussiness data")
    businessDF = ss.read.json(business_file)
    businessDF.printSchema()
    businessDF_vegas = businessDF.filter('latitude between {} and {}' \
                        .format(vegas_lat - lat_range, vegas_lat + lat_range)) \
                        .filter('longitude between {} and {}' \
                        .format(vegas_lon - lon_range, vegas_lon + lon_range)).cache()
    businessDF_vegas.groupBy('city').agg({"*":"count", "review_count":"avg"})\
                    .orderBy('count(1)', ascending=False)\
                    .withColumnRenamed('count(1)', '# Business/City')\
                    .withColumnRenamed('avg(review_count)', 'Avg # Reviews/Business')\
                    .show(10) 
    logger.error('Business data filtered for vegas')
    logger.error('{} seconds has elapsed'.format(time.time() - start_time))
    #Save vegas only business to file
    businessDF_vegas_save = os.path.join('dataset','business_vegas.parquet')
    businessDF_vegas.write.mode('overwrite').parquet(businessDF_vegas_save)
    
    #Filter for food business in vegas
    businessDF_vegas_food = businessDF_vegas.where(array_contains(businessDF_vegas.categories,'Restaurants')).cache()
    businessDF_vegas_food.show(10)
    businessDF_vegas.unpersist()
    logger.error('{} rows remains for vegas food only'.format(businessDF_vegas_food.count()))
    #Save vegas good only dataframe to disk
    businessDF_vegas_food_save = os.path.join('dataset','businessDF_vegas_food.parquet')
    businessDF_vegas.write.mode('overwrite').parquet(businessDF_vegas_food_save)

    #Obtain Reviews for vegas business only by joining and selecting
    review_file = os.path.join(dataset, 'yelp_academic_dataset_review.json')
    reviewDF = ss.read.json(review_file).cache()
    logger.error('Showing review data in vegas'.format(reviewDF.count()))
    reviewDF.printSchema()
    reviewDF_vegas_food = reviewDF.select('business_id','user_id', col('stars').alias('stars_long'))\
                             .join(businessDF_vegas_food, 'business_id', 'right').cache()
    reviewDF.unpersist()
    businessDF_vegas_food.unpersist()
    logger.error('Showing {} rows of food review data in vegas'.format(reviewDF_vegas_food.count()))
    logger.error('{} seconds has elapsed'.format(time.time() - start_time))

    #Index String ids into number ids and select nessary columns for als training
    reviewDF_vegas_food_clean = reviewDF_vegas_food.select('user_id','business_id','stars_long','latitude','longitude').cache()
    review_ids_to_number(reviewDF_vegas_food_clean)
    logger.error('Cleaned review data in vegas')
    logger.error('{} seconds has elapsed'.format(time.time() - start_time))
    reviewDF_vegas_food_clean.unpersist()
    

    #load users information(not needed atm)
    #user_file = os.path.join(dataset, 'yelp_academic_dataset_user.json')  
    #logger.error('Operation complete')
    #logger.error('{} seconds has elapsed'.format(time.time() - start_time))