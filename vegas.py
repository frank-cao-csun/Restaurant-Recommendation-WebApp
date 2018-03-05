import os,json,logging
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

logging.basicConfig(level=logging.INFO)
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
   
    return ss

if __name__ == '__main__':

    ss = init_spark_session()
    dataset = os.path.join('dataset','json')

    #Filter for bussiness in Vegas
    business_file = os.path.join(dataset, 'yelp_academic_dataset_business.json')

    #Join review set on bussiness_id with Vegas Bussiness
    review_file = os.path.join(dataset, 'yelp_academic_dataset_review.json')

    #Join user with filter review set on user_id
    user_file = os.path.join(dataset, 'yelp_academic_dataset_user.json')
    